#
# Copyright (C) 2024 Roberto Lopez Castro (roberto.lopez.castro@udc.es).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import torch
import torch.nn as nn


import marlin_cuda

from marlin._semi_structured_conversions import (
    sparse_semi_structured_from_dense_cutlass,
    mask_creator,
)


def mul_2_4(
    A, B, meta, C, s, workspace, num_bits, thread_k=-1, thread_m=-1, sms=-1, max_par=16
):
    """Marlin FP16x(INT4+2:4 sparsity) multiply; can be used within `torch.compile`.
    @A: `torch.int` weight matrix of original shape `(m, k)` in Marlin format; see `Layer.pack()`
    @B: `torch.half` input matrix of shape `(n, k/2)` in column-major layout
    @meta: `torch.int` metadata information for 2:4 sparsity
    @C: `torch.half` out matrix of shape `(n, m)` in column-major layout
    @s: `torch.half` scales of shape `(n / groupsize /2, m)`
    @workspace: `torch.int` tensor with at least `m / 128 * max_par` entries that are all zero
    @thread_k: `k` size of a thread_tile in `A` (can usually be left as auto -1)
    @thread_m: `m` size of a thread_tile in `A` (can usually be left as auto -1)
    @sms: number of SMs to use for the kernel (can usually be left as auto -1)
    @max_par: maximum number of batch 64 problems to solve in parallel for large input sizes
    """
    marlin_cuda.mul_2_4(
        A, B, meta, C, s, workspace, num_bits, thread_k, thread_m, sms, max_par
    )


def mul(A, B, C, s, workspace, thread_k=-1, thread_n=-1, sms=-1, max_par=16):
    """Marlin FP16xINT4 multiply; can be used within `torch.compile`.
    @A: `torch.half` input matrix of shape `(m, k)` in standard row-major layout
    @B: `torch.int` weight matrix of original shape `(k, n)` in Marlin format; see `Layer.pack()`
    @C: `torch.half` out matrix of shape `(m, n)` in standard row-major layout
    @s: `torch.half` scales of shape `(m / groupsize, n)`
    @workspace: `torch.int` tensor with at least `n / 128 * max_par` entries that are all zero
    @thread_k: `k` size of a thread_tile in `B` (can usually be left as auto -1)
    @thread_n: `n` size of a thread_tile in `B` (can usually be left as auto -1)
    @sms: number of SMs to use for the kernel (can usually be left as auto -1)
    @max_par: maximum number of batch 64 problems to solve in parallel for large input sizes
    """
    marlin_cuda.mul(A, B, C, s, workspace, thread_k, thread_n, sms, max_par)


# Precompute permutations for Marlin weight and scale shuffling
def _get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])
    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single


_perm, _scale_perm, _scale_perm_single = _get_perms()


def _get_perms_NT():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        col_o = (col * 4) % 16 + ((col * 4) // 16) * 256
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col_o + 512 * block)
        # print(perm1)
        for j in range(4):
            perm.extend([p + 1 * j for p in perm1])
        # print(perm)
    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(8):
        scale_perm_single.extend([4 * i + j for j in [0, 1, 2, 3, 32, 33, 34, 35]])
    return perm, scale_perm, scale_perm_single


_perm_t, _scale_perm_t, _scale_perm_single_t = _get_perms_NT()


class Layer(nn.Module):
    """PyTorch compatible Marlin layer; 4-bit (symmetric grouped) linear layer without bias."""

    def __init__(self, infeatures, outfeatures, groupsize=-1):
        """Create an empty Marlin layer.
        @infeatures: number of input features (must be divisible by 128)
        @outfeatures: number of output features (must be divisible by 256)
        @groupsize: quantization groupsize (must be -1 or 128)
        """
        super().__init__()
        if groupsize not in [-1, 128]:
            raise ValueError("Only groupsize -1 and 128 are supported.")
        if infeatures % 128 != 0 or outfeatures != 256 == 0:
            raise ValueError(
                "`infeatures` must be divisible by 128 and `outfeatures` by 256."
            )
        if groupsize == -1:
            groupsize = infeatures
        if infeatures % groupsize != 0:
            raise ValueError("`infeatures` must be divisible by `groupsize`.")
        self.k = infeatures
        self.n = outfeatures
        self.groupsize = groupsize
        self.register_buffer(
            "B", torch.empty((self.k // 16, self.n * 16 // 8), dtype=torch.int)
        )
        self.register_buffer(
            "s", torch.empty((self.k // groupsize, self.n), dtype=torch.half)
        )
        # 128 is currently the minimum `tile_n`, hence it gives the maximum workspace size; 16 is the default `max_par`
        self.register_buffer(
            "workspace",
            torch.zeros(self.n // 128 * 16, dtype=torch.int),
            persistent=False,
        )

    def forward(self, A):
        C = torch.empty(
            A.shape[:-1] + (self.s.shape[1],), dtype=A.dtype, device=A.device
        )
        mul(
            A.view((-1, A.shape[-1])),
            self.B,
            C.view((-1, C.shape[-1])),
            self.s,
            self.workspace,
        )
        return C

    def pack(self, linear, scales, trans=False):
        """Pack a fake-quantized linear layer into this actual Marlin representation.
        @linear: fake-quantized `torch.nn.Linear` layer to convert (must be of type `torch.half`)
        @scales: corresponding quantization scales of shape `(infeatures, groups)`
        """
        if linear.weight.dtype != torch.half:
            raise ValueError("Only `torch.half` weights are supported.")
        if trans:
            perm, scale_perm, scale_perm_single = (
                _perm_t,
                _scale_perm_t,
                _scale_perm_single_t,
            )
        else:
            perm, scale_perm, scale_perm_single = _perm, _scale_perm, _scale_perm_single
        tile = 16
        maxq = 2**4 - 1
        s = scales
        w = linear.weight.data
        if self.groupsize != self.k:
            w = w.reshape((-1, self.groupsize, self.n))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.groupsize, -1))
            s = s.reshape((1, -1))
        w = torch.round(w / s).int()
        w += (maxq + 1) // 2
        w = torch.clamp(w, 0, maxq)
        if self.groupsize != self.k:
            w = w.reshape((self.groupsize, -1, self.n))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.k, self.n)).contiguous()
            s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
        else:
            s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]

        s = s.reshape((-1, self.n)).contiguous()
        w = w.reshape((self.k // tile, tile, self.n // tile, tile))
        w = w.permute((0, 2, 1, 3))
        w = w.reshape((self.k // tile, self.n * tile))
        res = w
        res = res.reshape((-1, perm.numel()))[:, perm].reshape(res.shape)
        q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
        res = res.cpu().numpy().astype(np.uint32)
        for i in range(8):
            q |= res[:, i::8] << 4 * i

        q = torch.from_numpy(q.astype(np.int32)).to(w.device)
        self.B[:, :] = q.to(self.B.device)
        self.s[:, :] = s.to(self.s.device)


def _get_perms_2_4(num_bits):
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        col_o = col // 2
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col_o * 256 + 8 * (col % 2) + 4 * block)
        for j in range(4):
            perm.extend([p + 1 * j for p in perm1])
    perm = np.array(perm)

    if num_bits == 4:
        interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = np.array([0, 2, 1, 3])
    else:
        raise ValueError("num_bits must be 4 or 8, got {}".format(num_bits))

    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i * 8 + j for j in [0, 4, 1, 5, 2, 6, 3, 7]])
    scale_perm_single = []
    for i in range(8):
        scale_perm_single.extend([8 * i + j for j in [0, 1, 2, 3, 4, 5, 6, 7]])
    return perm, scale_perm, scale_perm_single


_perm_2_4 = {}
_scale_perm_2_4 = {}
_scale_perm_single_2_4 = {}
for num_bits in [4, 8]:
    perm_2_4, scale_perm_2_4, scale_perm_single_2_4 = _get_perms_2_4(num_bits)
    _perm_2_4[num_bits] = perm_2_4
    _scale_perm_2_4[num_bits] = scale_perm_2_4
    _scale_perm_single_2_4[num_bits] = scale_perm_single_2_4


class Layer_2_4(nn.Module):
    """PyTorch compatible Marlin 2:4 layer; 4-bit (symmetric grouped) linear layer without bias."""

    def __init__(self, infeatures, outfeatures, groupsize=-1):
        """Create an empty Marlin layer.
        @infeatures: number of input features (must be divisible by 128)
        @outfeatures: number of output features (must be divisible by 256)
        @groupsize: quantization groupsize (must be -1 or 128)
        """
        super().__init__()
        if groupsize not in [-1, 128]:
            raise ValueError("Only groupsize -1 and 128 are supported.")
        if infeatures % 128 != 0 or outfeatures != 256 == 0:
            raise ValueError(
                "`infeatures` must be divisible by 64 and `outfeatures` by 256."
            )
        if groupsize == -1:
            groupsize = infeatures
        if infeatures % groupsize != 0:
            raise ValueError("`infeatures` must be divisible by `groupsize`.")
        self.k = infeatures
        self.n = outfeatures
        self.groupsize = groupsize
        self.register_buffer(
            "B", torch.empty((self.k // 16, self.n * 16 // 8), dtype=torch.int)
        )
        self.register_buffer(
            "meta", torch.empty((self.n, self.k // 16), dtype=torch.int16)
        )
        self.register_buffer(
            "s", torch.empty((self.k // groupsize, self.n), dtype=torch.half)
        )
        # 128 is currently the minimum `tile_n`, hence it gives the maximum workspace size; 16 is the default `max_par`
        self.register_buffer(
            "workspace",
            torch.zeros(
                self.n // 128 * 16, dtype=torch.int32, device=torch.device("cuda:0")
            ),
            persistent=False,
        )

    def forward(self, A):
        C = torch.empty(
            A.shape[:-1] + (self.s.shape[1],), dtype=A.dtype, device=A.device
        )

        mul_2_4(
            A.view((-1, A.shape[-1])),
            self.B,
            self.meta,
            C.view((-1, C.shape[-1])),
            self.s,
            self.workspace,
        )
        # mul_2_4(A, self.B, self.meta, C, self.s, self.workspace)
        return C

    def pack(self, linear, scales, num_bits, trans=False):
        """Pack a fake-quantized linear layer into this actual Marlin representation.
        @linear: fake-quantized `torch.nn.Linear` layer to convert (must be of type `torch.half`)
        @scales: corresponding quantization scales of shape `(infeatures, groups)`
        """
        assert num_bits == 4 or num_bits == 8
        pack_factor = 32 // num_bits

        if linear.weight.dtype != torch.half:
            raise ValueError("Only `torch.half` weights are supported.")
        if trans:
            perm, scale_perm, scale_perm_single = (
                _perm_2_4[num_bits],
                _scale_perm_2_4[num_bits],
                _scale_perm_single_2_4[num_bits],
            )
        else:
            perm, scale_perm, scale_perm_single = _perm, _scale_perm, _scale_perm_single
        tile = 16
        maxq = (1 << num_bits) - 1
        s = scales
        w = linear.weight.data
        if self.groupsize != self.k:
            w = w.reshape((-1, self.groupsize, self.n))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.groupsize, -1))
            s = s.reshape((1, -1))
        w = torch.round(w / s).int()
        w += (maxq + 1) // 2
        w = torch.clamp(w, 0, maxq)

        if self.groupsize != self.k:
            w = w.reshape((self.groupsize, -1, self.n))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.k, self.n)).contiguous()
            s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
        else:
            s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]

        mask = mask_creator(w.T).cuda().bool()
        w = mask * w.T
        w, meta = sparse_semi_structured_from_dense_cutlass(w)
        w = w.t()
        self.k = self.k // 2
        self.groupsize = self.groupsize // 2

        s = s.reshape((-1, self.n)).contiguous()
        w = w.reshape((self.k // tile, tile, self.n // tile, tile))
        w = w.permute((0, 2, 1, 3))
        w = w.reshape((self.k // tile, self.n * tile))
        res = w
        res = res.reshape((-1, perm.numel()))[:, perm].reshape(res.shape)
        q = np.zeros((res.shape[0], res.shape[1] // pack_factor), dtype=np.uint32)
        res = res.cpu().numpy().astype(np.uint32)
        for i in range(pack_factor):
            q |= res[:, i::pack_factor] << num_bits * i

        q = torch.from_numpy(q.astype(np.int32)).to(w.device)
        self.B[:, :] = q.to(self.B.device)
        self.s[:, :] = s.to(self.s.device)
        self.meta[:, :] = meta.to(self.meta.device)


def replace_linear(module, name_filter=lambda n: True, groupsize=-1, name=""):
    """Recursively replace all `torch.nn.Linear` layers by empty Marlin 2:4 layers.
    @module: top-level module in which to perform the replacement
    @name_filter: lambda indicating if a layer should be replaced
    @groupsize: marlin groupsize
    @name: root-level name
    """
    if isinstance(module, Layer_2_4):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + "." + attr if name != "" else attr
        if isinstance(tmp, nn.Linear) and name_filter(name1):
            setattr(
                module,
                attr,
                Layer_2_4(tmp.in_features, tmp.out_features, groupsize=groupsize),
            )
    for name1, child in module.named_children():
        replace_linear(
            child,
            name_filter,
            groupsize=groupsize,
            name=name + "." + name1 if name != "" else name1,
        )
