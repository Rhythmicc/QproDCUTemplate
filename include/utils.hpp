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

__global__ static void __gemm_nnz_per_row(
    int n,
    const int *csr_row_ptr_A,
    const int *csr_col_ind_A,
    const int *csr_row_ptr_B,
    const int *csr_col_ind_B,
    int *row_nnz)
{
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int stride = hipBlockDim_x * hipGridDim_x;

    const int trunk_size = 2048;
    char flag[trunk_size];
    int trunk = 0;
    for (int ar = tid; ar < n; ar += stride)
    {
        while (trunk < n)
        {
            memset(flag, 0, sizeof(char) * trunk_size);

            for (int ai = csr_row_ptr_A[ar]; ai < csr_row_ptr_A[ar + 1]; ai++)
            {
                int br = csr_col_ind_A[ai];
                for (int bi = csr_row_ptr_B[br]; bi < csr_row_ptr_B[br + 1]; bi++)
                {
                    int bc = csr_col_ind_B[bi];
                    if (bc >= trunk && bc < trunk + trunk_size && !flag[bc - trunk])
                    {
                        row_nnz[ar + 1]++;
                        flag[bc - trunk] = 1;
                    }
                }
            }
            trunk += trunk_size;
        }
    }
}

void getnnz(
    int n,
    int nnz_A,
    const int *csr_row_ptr_A,
    const int *csr_col_ind_A,
    int nnz_B,
    const int *csr_row_ptr_B,
    const int *csr_col_ind_B,
    int *csr_row_ptr_C,
    size_t *nnz_C)
{
    const int threadPerBlock = 256;
    const int blockPerGrid = (n - 1) / threadPerBlock + 1;

    int last = 0;

    hipMemset(csr_row_ptr_C, '\0', (n + 1) * sizeof(int));

    __gemm_nnz_per_row<<<blockPerGrid, threadPerBlock>>>(n, csr_row_ptr_A, csr_col_ind_A, csr_row_ptr_B, csr_col_ind_B, csr_row_ptr_C);
    // prefix<<<dim3(1), dim3(1)>>>(csr_row_ptr_C, n + 1);
    prefix_sum(csr_row_ptr_C, n + 1);

    hipMemcpyAsync(nnz_C, csr_row_ptr_C + n, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpyAsync(&last, csr_row_ptr_C + n - 1, sizeof(int), hipMemcpyDeviceToHost);
}