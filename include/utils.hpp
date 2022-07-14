#include <kernel_common.hpp>

// CUDA前缀和

__global__ void __prefix_sum_warp(int *input, int n)
{
    int tx = hipThreadIdx_x;
    int tid = hipBlockIdx_x * hipBlockDim_x + tx;
    __shared__ int sum[WARP_SIZE];
    sum[tx] = tid < n ? input[tid] : 0;
    int int4_j = tx & 3, sum_index = tx & 0x3c;
    int *cur = sum + sum_index;
    if (int4_j > 1)
    {
        cur[int4_j] += cur[int4_j - 2];
    }
    __syncthreads();
    if (int4_j == 2)
    {
        cur[int4_j + 1] += cur[int4_j];
        cur[int4_j] += cur[int4_j - 1];
    }
    __syncthreads();
    if (int4_j == 1)
        cur[1] += cur[0];

    if (tx == 0)
    {
        for (int i = 4; i < WARP_SIZE; i += 4)
        {
            *(int4 *)(sum + i) += (int4){sum[i - 1], sum[i - 1], sum[i - 1], sum[i - 1]};
        }
    }
    if (tid < n)
        input[tid] = sum[tx];
}

__global__ void __merge_warp(int *input, int n)
{
    int tx = hipThreadIdx_x, pre;
#pragma unroll
    for (int i = 64; i < n; i += 64)
    {
        pre = input[i - 1];
        if (i + tx < n)
            input[i + tx] += pre;
        __syncthreads();
    }
}

void prefix_sum(int *row_nnz, int size)
{
    int warp_num = (size >> WARP_BITS) + (size & (WARP_SIZE - 1) ? 1 : 0);
    __prefix_sum_warp<<<warp_num, WARP_SIZE>>>(row_nnz, size);
    __merge_warp<<<1, WARP_SIZE>>>(row_nnz, size);
}

float bit_len(int x)
{
    int res = 0;
    while (x)
    {
        x >>= 1;
        res++;
    }
    return res;
}