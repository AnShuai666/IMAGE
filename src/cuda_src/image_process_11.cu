/**
 * @desc    image_process.cpp文件实现
 * @author  杨丰拓
 * @date    2019-05-16
*/
#include <cuda_runtime.h>
#include <cstdio>
#include "cuda_include/common.cuh"
template <typename T>
static void gpu_cpu2zero1(T *cpu,T *gpu,size_t bytes)
{
    memset(cpu, 0, bytes);
    cudaMemset(gpu,0,bytes);
}
///功能：图像变换
/*  函数名                           线程块大小       耗费时间
 *  kernel_byte2float2	            689.561us	    [32,4,1]
 *  kernel_byte2float8	            681.097us	    [32,4,1]**
 */
///核函数
/* 调用示例
 * dim3 block(x,y,1);
 * dim3 grid((width*channels-1+x*2)/(x*2),(height-1+y)/y,1);
 * kernel_byte2float2<<<grid,block>>>(d_out, d_in, width, height, channels);
 */
__global__ void kernel_byte2float2(float * dst,unsigned char *src,int const w,int const h,int const c)
{
    int x=threadIdx.x+blockIdx.x*blockDim.x*2;
    int y=threadIdx.y+blockIdx.y*blockDim.y;

    if(x<w*c&&y<h)
    {//此法寄存器使用较多但速度较快
        int idx=y*w*c+x;
        int idx1=idx+blockDim.x;
        float value=src[idx];
        float value1=src[idx1];

        dst[idx]=value/255.0f;
        dst[idx1]=value1/255.0f;
    }
}
/* 调用示例
 * dim3 block(x,y,1);
 * dim3 grid((width*channels-1+x*8)/(x*8),(height-1+y)/y,1);
 * kernel_byte2float8<<<grid,block>>>(d_out, d_in, width, height, channels);
 */
__global__ void kernel_byte2float8(float * dst,unsigned char *src,int const w,int const h,int const c)
{
    int x=threadIdx.x+blockIdx.x*blockDim.x*8;
    int y=threadIdx.y+blockIdx.y*blockDim.y;
    if(x<w*c&&y<h)
    {
        int idx=y*w*c+x;
        int idx1=idx+blockDim.x;
        int idx2=idx1+blockDim.x;
        int idx3=idx2+blockDim.x;
        int idx4=idx3+blockDim.x;
        int idx5=idx4+blockDim.x;
        int idx6=idx5+blockDim.x;
        int idx7=idx6+blockDim.x;

        float value=src[idx];
        float value1=src[idx1];
        float value2=src[idx2];
        float value3=src[idx3];
        float value4=src[idx4];
        float value5=src[idx5];
        float value6=src[idx6];
        float value7=src[idx7];

        dst[idx]=value/255.0f;
        dst[idx1]=value1/255.0f;
        dst[idx2]=value2/255.0f;
        dst[idx3]=value3/255.0f;
        dst[idx4]=value4/255.0f;
        dst[idx5]=value5/255.0f;
        dst[idx6]=value6/255.0f;
        dst[idx7]=value7/255.0f;
    }
}

int byte_to_float_image_by_cuda(float * dstImage,unsigned char *srcImage,int const width,int const height,int const channels,float * contrast)
{
    //计算存储空间字节数
    size_t const bytes_uchar=width*height*channels* sizeof(unsigned char);
    size_t const bytes_float=width*height*channels* sizeof(float);
    //声明显存指针
    unsigned char * d_in=NULL;
    float *d_out=NULL;
    //定义显存指针
    cudaMalloc(&d_in,bytes_uchar);
    cudaMalloc(&d_out,bytes_float);
    //cpu2gpu
    cudaMemcpy(d_in,srcImage,bytes_uchar,cudaMemcpyHostToDevice);
    //网格划分
    int x=32;
    int y=4;
    dim3 block(x,y,1);
    dim3 grid((width*channels-1+x*8)/(x*8),(height-1+y)/y,1);
    kernel_byte2float8 <<< grid, block >>> (d_out, d_in, width, height, channels);
    cudaMemcpy(dstImage, d_out, bytes_float, cudaMemcpyDeviceToHost);
    //compare1(dstImage, contrast, width * channels, height, true);
    //释放显存指针
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}