import sys

import numpy as np
import torch
import marlin

import time

from marlin._semi_structured_conversions import (
    sparse_semi_structured_from_dense_cutlass,
    sparse_semi_structured_to_dense_cutlass,
)

def benchmark(f, warmup=1, iter=10):
    for i in range(warmup + iter):
        f()
        # We do not synchronize here in order to hide the kernel launch overhead during benchmarkining as this will also
        # happen during realistic model inference as many launches are submitted to the kernel queue.
        if i == warmup - 1:
            torch.cuda.synchronize()
            tick = time.time()
    torch.cuda.synchronize()
    res = (time.time() - tick) / iter
    # Make sure there is enough to "cool down" the GPU in between benchmarks to avoid throttling for later runs when
    # we execute many benchmarks consecutively
    time.sleep(1.)
    return res

def get_problem(m, n, k, groupsize=-1):
    if groupsize == -1:
        groupsize = k
    dev = torch.device('cuda:0')
    A = torch.randn((m, k), dtype=torch.half, device=dev)
    B = torch.randint(low=-2**31, high=2**31, size=(k * n // 8,), device=dev)
    B_ref = torch.randn((k, n), dtype=torch.half, device=dev)
    C = torch.zeros((m, n), dtype=torch.half, device=dev)
    s = torch.zeros((k // groupsize, n), dtype=torch.half, device=dev)
    torch.cuda.synchronize()
    return A, B, C, B_ref, s

def gen_2_4(m, n, k, dev):
    B = torch.randint(low=-2**31, high=2**31, size=(k * n // 8 // 2,), device=dev)
    meta = torch.ones((n * k//16,), dtype=torch.int16, device=dev)*(-4370)
    return B, meta

def get_problem_24(m, n, k, groupsize=-1):
    dev = torch.device('cuda:0')
    B, meta = gen_2_4(m, n, k, dev)
    if groupsize == -1:
        s = torch.zeros((1, n), dtype=torch.half, device=dev)
    else:
        s = torch.zeros((((k//2) // (groupsize//2)), n), dtype=torch.half, device=dev)
    torch.cuda.synchronize()
    return B, s, meta

def benchmark_dense(A, B, C):
    res = benchmark(lambda: torch.matmul(A, B, out=C))
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * C.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A.numel() + 2 * B.numel() + 2 * C.numel()) / res / 10 ** 9
    }

def benchmark_quant_24(A, B, meta, C, s, thread_k, thread_m, sms):
    workspace = torch.zeros(C.shape[1] // 128 * 16, device=torch.device('cuda:0'), dtype=torch.int32)
    #print("A:", A.shape, "B:", B.shape, "meta:", meta.shape, "C:", C.shape, "s:", s.shape)
    res = benchmark(lambda: marlin.mul_2_4(A, B, meta, C, s, workspace, thread_k, thread_m, sms))
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * C.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A.numel() + 4 * B.numel()*2 + 2 * C.numel() + 2 * s.numel()) / res / 10 ** 9
    }


def benchmark_quant(A, B, C, s, thread_k, thread_n, sms):
    workspace = torch.zeros(C.shape[1] // 128 * 16, device=torch.device('cuda:0'), dtype=torch.int32)
    #print("A:", A.shape, "B:", B.shape, "C:", C.shape, "s:", s.shape)
    res = benchmark(lambda: marlin.mul(A, B, C, s, workspace, thread_k, thread_n, sms))
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * C.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A.numel() + 4 * B.numel() + 2 * C.numel() + 2 * s.numel()) / res / 10 ** 9
    }

# Pass the SM count for known GPUs to avoid the kernel having to query this information (this is very minor)
gpu = torch.cuda.get_device_name(0)
if 'A100' in gpu:
    SMS = 108
elif 'A10' in gpu:
    SMS = 72
elif '3090' in gpu:
    SMS = 82
elif 'A6000' in gpu:
    SMS = 84
elif '4090' in gpu:
    SMS = 128
else:
    SMS = -1

MODELS = {
    #'ideal': [
    #    (4 * 256 * SMS, 256 * SMS)
    #],
    'Llama7B': [
        (4096, 3 * 4096),
        (4096, 4096),
        (4096, 2 * 10752),
        (10752, 4096)
    ],
    'Llama13B': [
        (5120, 3 * 5120),
        (5120, 5120),
        (5120, 2 * 13568),
        (13568, 5120)
    ],
    'Llama33B': [
        (6656, 3 * 6656),
        (6656, 6656),
        (6656, 2 * 17664),
        (17664, 6656)
    ],
    'Llama65B': [
        (8192, 3 * 8192),
        (8192, 8192),
        (8192, 2 * 21760),
        (21760, 8192)
    ],
    'Falcon180B': [
        # Note that parallel attention and FC allows layer fusions
        (14848, 14848 * 5 + 1024),
        (14848 * 5, 14848)
    ]
}

# Set to true in order to run a more complete benchmark sweep; the default is reproduce README experiments
ALL = True

#print("groupsize,model,batch,tot_q_s,tot_q_2_4_s,tot_q_sp,tot_q_2_4_sp")
for groupsize in [-1, 128] if ALL else [128]:
    print('groupsize=%d' % groupsize)
    print()
    for model, layers in MODELS.items():
        print(model)
        if ALL:
            batchsizes =  [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        else:
            batchsizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        for batch in batchsizes:
            if not ALL and model != 'ideal' and batch != 16:
                continue
            tot_q = {'s': 0, 'TFLOP/s': 0, 'GB/s': 0, 'speedup': 0}
            tot_q_24 = {'s': 0, 'TFLOP/s': 0, 'GB/s': 0, 'speedup': 0}
            for layer in layers:
                A, B, C, B_ref, s = get_problem(batch, layer[1], layer[0], groupsize)
                B_24, s_24, meta = get_problem_24(batch, layer[1], layer[0], groupsize)
                res_d = benchmark_dense(A, B_ref, C)
                if model == 'ideal' and batch == 16:
                    # This is a special case constructed to be optimal for a thread-shape different than the default one
                    res_q_24 = benchmark_quant_24(A, B_24, meta, C, s_24, 64, 256, SMS)
                    res_q = benchmark_quant(A, B, C, s, 64, 256, SMS)
                else:
                    res_q_24 = benchmark_quant_24(A, B_24, meta, C, s_24, -1, -1, SMS)
                    res_q = benchmark_quant(A, B, C, s, -1, -1, SMS)
                #
                res_q['speedup'] = res_d['s'] / res_q['s']
                tot_q['s'] += res_q['s']
                #
                res_q_24['speedup'] = res_d['s'] / res_q_24['s']
                tot_q_24['s'] += res_q_24['s']
                for k in tot_q:
                    if k != 's':
                        tot_q[k] += res_q[k] * res_q['s']
                for k in tot_q_24:
                    if k != 's':
                        tot_q_24[k] += res_q_24[k] * res_q_24['s']
                del A
                del B
                del C
                del B_ref
                del s
                del B_24
                del s_24
                del meta
            for k in tot_q:
                if k != 's':
                    tot_q[k] /= tot_q['s']
            for k in tot_q_24:
                if k != 's':
                    tot_q_24[k] /= tot_q_24['s']

            print('[NN]  batch=%04d: s=%.5f, TFLOP/s=%07.3f, GB/s=%08.3f, speedup=%.2f' % (
                batch,
                tot_q['s'],
                tot_q['TFLOP/s'],
                tot_q['GB/s'],
                tot_q['speedup']
            ))
            print('[2:4] batch=%04d: s=%.5f, TFLOP/s=%07.3f, GB/s=%08.3f, speedup=%.2f' % (
                batch,
                tot_q_24['s'],
                tot_q_24['TFLOP/s'],
                tot_q_24['GB/s'],
                tot_q_24['speedup']
            ))

            #print(str(groupsize)+',%s,%04d,%.5f,%.5f,%.2f,%.2f' % (
            #    model, batch, tot_q['s'], tot_q_24['s'], tot_q['speedup'], tot_q_24['speedup']))
        print()