#include "cuda_include/common.cuh"
__global__ void KernelWarmUp()
{}
int warmUp()
{
    KernelWarmUp<<<1,1>>>();
    return 0;
}