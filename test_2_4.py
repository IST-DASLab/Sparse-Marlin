import unittest

import numpy as np
import torch
import torch.nn as nn

import marlin

from marlin._semi_structured_conversions import (
    mask_creator,
)

seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)

DEV = torch.device("cuda:0")

torch.set_printoptions(sci_mode=False, profile="full")


def gen_quant4_NT(m, k, groupsize=-1):
    maxq = 2**4 - 1
    w = torch.randn((m, k), dtype=torch.half, device=DEV)
    k_sp = k // 2

    w = w.t()
    if groupsize != -1:
        w = w.reshape((-1, groupsize, m))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    w = torch.round(w / s).int()
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    ref = (w - (maxq + 1) // 2).half() * s
    if groupsize != -1:

        def reshape(w):
            w = w.reshape((groupsize, -1, m))
            w = w.permute(1, 0, 2)
            w = w.reshape((k, m)).contiguous()
            return w

        ref = reshape(ref)
        w = reshape(w)

    mask = mask_creator(w.T).cuda().bool()
    uncompress = (mask * ref.T).T

    s = s.reshape((-1, m)).contiguous()
    linear = nn.Linear(k, m)
    linear.weight.data = ref

    layer = marlin.Layer_2_4(256, 256, groupsize=groupsize)
    if groupsize == -1:
        groupsize = k
    layer.k = k
    layer.n = m
    layer.groupsize = groupsize
    layer.B = torch.empty((k_sp // 16, m * 16 // 8), dtype=torch.int, device=DEV)
    layer.meta = torch.empty((m, k // 16), dtype=torch.int16, device=DEV)
    layer.s = torch.empty((k_sp // (groupsize // 2), m), dtype=torch.half, device=DEV)
    layer.pack(linear, s, True)
    q = layer.B
    s = layer.s
    meta = layer.meta

    return uncompress, q, s, meta


class Test(unittest.TestCase):
    def run_problem(self, m, n, k, thread_k, thread_m, groupsize=-1):
        print(
            "% 5d % 6d % 6d % 4d % 4d % 4d" % (m, n, k, thread_k, thread_m, groupsize)
        )
        A = torch.randn((n, k), dtype=torch.half, device=DEV)
        B_ref, B, s, meta = gen_quant4_NT(m, k, groupsize=groupsize)
        C = torch.zeros((n, m), dtype=torch.half, device=DEV)
        C_ref = torch.matmul(A, B_ref)

        workspace = torch.zeros(m // 128 * 16, device=DEV, dtype=torch.int32)
        marlin.mul_2_4(A, B, meta, C, s, workspace, thread_k, thread_m, -1)
        torch.cuda.synchronize()

        self.assertLess(
            torch.mean(torch.abs(C - C_ref)) / torch.mean(torch.abs(C_ref)), 0.002
        )

    def test_correctness(self):
        self.run_problem(256, 16, 256, 128, 128, -1)
        self.run_problem(21504, 16, 4096, 64, 256, 128)

    def test_tiles(self):
        print()
        for m in [1, 2, 4, 8, 12, 16, 32, 64]:
            for thread_k, thread_n in [(64, 256), (128, 128)]:
                if m > 16 and thread_k == 128:
                    continue
                self.run_problem(2 * 256, m, 1024, thread_k, thread_n)

    def test_k_stages_divisibility(self):
        print()
        for k in [3 * 64 + 64 * 4 * 2 + 64 * i for i in range(1, 4)]:
            self.run_problem(2 * 256, 16, k, 64, 256)

    def test_very_few_stages(self):
        print()
        for k in [64, 128, 192]:
            self.run_problem(3 * 256, 16, k, 64, 256)

    def test_llama_shapes(self):
        print()
        return
        MODELS = {
            " 7B": [(4096, 3 * 4096), (4096, 4096), (4096, 2 * 10752), (10752, 4096)],
            "13B": [(5120, 3 * 5120), (5120, 5120), (5120, 2 * 13568), (13568, 5120)],
            "33B": [(6656, 3 * 6656), (6656, 6656), (6656, 2 * 17664), (17664, 6656)],
            "70B": [(8192, 3 * 8192), (8192, 8192), (8192, 2 * 21760), (21760, 8192)],
        }
        for _, layers in MODELS.items():
            for layer in layers:
                for thread_k, thread_m in [(128, 128)]:
                    for batch in [16]:
                        print(layer[1], batch, layer[0])
                        self.run_problem(layer[1], batch, layer[0], thread_k, thread_m)

    def test_groups(self):
        print()
        for m in [16]:
            for groupsize in [128]:
                for n, k in [(256, 512), (256, 1024), (256 * 128, 1024)]:
                    for thread_shape in [(128, 128), (64, 256)]:
                        self.run_problem(n, m, k, *thread_shape, groupsize)


gpu = torch.cuda.get_device_name(0)
if "A100" in gpu:
    SMS = 108
elif "A10" in gpu:
    SMS = 72
elif "3090" in gpu:
    SMS = 82
elif "A6000" in gpu:
    SMS = 84
else:
    SMS = -1

if __name__ == "__main__":
    unittest.main()
