/**
 * @desc    image_process.cpp文件实现
 * @author  杨丰拓
 * @date    2019-05-16
*/
#include <cuda_runtime.h>
#include <cstdio>
#include "cuda_include/common.cuh"
template <typename T>
static void gpu_cpu2zero1(T *p_cpu,T *p_gpu,size_t bytes)
{
    memset(p_cpu, 0, bytes);
    cudaMemset(p_gpu,0,bytes);
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
__global__ void kernelByteToFloat2(float *p_dst,unsigned char *p_src,int const kWidth,int const kHeight,int const kChannels)
{
    int x=threadIdx.x+blockIdx.x*blockDim.x*2;
    int y=threadIdx.y+blockIdx.y*blockDim.y;

    if(x<kWidth*kChannels&&y<kHeight)
    {//此法寄存器使用较多但速度较快
        int idx=y*kWidth*kChannels+x;
        int idx1=idx+blockDim.x;
        float value =p_src[idx];
        float value1=p_src[idx1];

        p_dst[idx] =value/255.0f;
        p_dst[idx1]=value1/255.0f;
    }
}
/* 调用示例
 * dim3 block(x,y,1);
 * dim3 grid((width*channels-1+x*8)/(x*8),(height-1+y)/y,1);
 * kernel_byte2float8<<<grid,block>>>(d_out, d_in, width, height, channels);
 */
__global__ void kernelByteToFloat8(float *p_dst,unsigned char *p_src,int const kWidth,int const kHeight,int const kChannels)
{
    int x=threadIdx.x+blockIdx.x*blockDim.x*8;
    int y=threadIdx.y+blockIdx.y*blockDim.y;
    if(x<kWidth*kChannels&&y<kHeight)
    {
        int idx=y*kWidth*kChannels+x;
        int idx1=idx+blockDim.x;
        int idx2=idx1+blockDim.x;
        int idx3=idx2+blockDim.x;
        int idx4=idx3+blockDim.x;
        int idx5=idx4+blockDim.x;
        int idx6=idx5+blockDim.x;
        int idx7=idx6+blockDim.x;

        float value =p_src[idx];
        float value1=p_src[idx1];
        float value2=p_src[idx2];
        float value3=p_src[idx3];
        float value4=p_src[idx4];
        float value5=p_src[idx5];
        float value6=p_src[idx6];
        float value7=p_src[idx7];

        p_dst[idx]=value/255.0f;
        p_dst[idx1]=value1/255.0f;
        p_dst[idx2]=value2/255.0f;
        p_dst[idx3]=value3/255.0f;
        p_dst[idx4]=value4/255.0f;
        p_dst[idx5]=value5/255.0f;
        p_dst[idx6]=value6/255.0f;
        p_dst[idx7]=value7/255.0f;
    }
}

int byteToFloatImageByCuda(float * p_dstImage,unsigned char *p_srcImage,int const kWidth,int const kHeight,int const kChannels)
{
    //计算存储空间字节数
    size_t const kBytes_uchar=kWidth*kHeight*kChannels* sizeof(unsigned char);
    size_t const kBytes_float=kWidth*kHeight*kChannels* sizeof(float);
    //声明显存指针
    unsigned char * p_d_in=NULL;
    float *p_d_out=NULL;
    //定义显存指针
    cudaMalloc(&p_d_in ,kBytes_uchar);
    cudaMalloc(&p_d_out,kBytes_float);
    //cpu2gpu
    cudaMemcpy(p_d_in,p_srcImage,kBytes_uchar,cudaMemcpyHostToDevice);
    //网格划分
    int x=32;
    int y=4;
    dim3 block(x,y,1);
    dim3 grid((kWidth*kChannels-1+x*8)/(x*8),(kHeight-1+y)/y,1);
    kernelByteToFloat8 <<< grid, block >>> (p_d_out, p_d_in, kWidth, kHeight, kChannels);
    cudaMemcpy(p_dstImage,p_d_out,kBytes_float, cudaMemcpyDeviceToHost);
    //释放显存指针
    cudaFree(p_d_in);
    cudaFree(p_d_out);
    return 0;
}