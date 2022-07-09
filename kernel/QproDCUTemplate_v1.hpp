#include <kernel_common.hpp>


__global__ void _QproDCUTemplate() {
    return;
}

float QproDCUTemplate() {
    float gflops = 0f;
    _QproDCUTemplate<<<1,1>>>();
    return gflops;
}