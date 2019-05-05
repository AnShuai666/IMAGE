/*
 * @功能      image.hpp内TODO函数实现
 * @姓名      杨丰拓
 * @日期      2019-4-29
 * @时间      17:14
 * @邮箱
*/
#include <cuda_runtime.h>
#include <iostream>
#include "cuda_include/common.cuh"
#include "cuda_include/sharedmem.cuh"
#include <cstdio>
template <typename T>
void gpu_cpu2zero1(T *cpu,T *gpu,size_t bytes)
{
    memset(cpu, 0, bytes);
    cudaMemset(gpu,0,bytes);
}
template <typename T>
__global__ void kernel_fill_color(T * image, T *color,int const wc,int const h,int const c)
{
    int x=threadIdx.x+blockIdx.x*blockDim.x;
    int y=threadIdx.y+blockIdx.y*blockDim.y;
    int idx=y*wc+x;
    //越界判断
    if(x<wc&&y<h)
    {
        int channels=idx%c;
        image[idx]=color[channels];
    }
}

template <typename T>
__global__ void kernel_fill_color3(T * image, T *color,int const wc,int const h,int const c)
{
    int x=threadIdx.x+blockIdx.x*blockDim.x*3;
    int y=threadIdx.y+blockIdx.y*blockDim.y;
    int idx=y*wc+x;

    T local_color[4];
    for(int i=0;i<c;i++)
    {
        local_color[i]=color[i];
    }
    //越界判断
    if((x+blockDim.x*2)<wc&&y<h)
    {
        int channels=idx%c;
        image[idx]=local_color[channels];

        idx+=blockDim.x;
        channels=idx%c;
        image[idx]=local_color[channels];

        idx+=blockDim.x;
        channels=idx%c;
        image[idx]=local_color[channels];
    }
}

template <typename T>
__global__ void kernel_fill_color3_by_share(T * image, T *color,int const wc,int const h,int const c)
{
    SharedMemory<T> smem;
    T* data = smem.getPointer();
    int x=threadIdx.x+blockIdx.x*blockDim.x*3;
    int y=threadIdx.y+blockIdx.y*blockDim.y;
    int idx=y*wc+x;
    int sidx=threadIdx.y*blockDim.x+threadIdx.x;
    if(sidx<c)data[sidx]=color[sidx];
    __syncthreads();
    //越界判断
    if((x+blockDim.x*2)<wc&&y<h)
    {
        int channels;
        for(int k=0;k<3;k++)
        {
            channels=idx%c;
            image[idx]=data[channels];
            idx+=blockDim.x;
        }
    }
}

template <typename T>
__global__ void kernel_fill_color15_by_share(T * image, T *color,int const wc,int const h,int const c)
{
    SharedMemory<T> smem;
    T* data = smem.getPointer();
    int x=threadIdx.x+blockIdx.x*blockDim.x*15;
    int y=threadIdx.y+blockIdx.y*blockDim.y;
    int idx=y*wc+x;
    int sidx=threadIdx.y*blockDim.x+threadIdx.x;
    if(sidx<c)data[sidx]=color[sidx];
    __syncthreads();
    //越界判断

    if(x<wc&&y<h)
    {
        int channels;
        for(int k=0;k<15;k++)
        {
            channels=idx%c;
            image[idx]=data[channels];
            idx+=blockDim.x;
        }
    }
}

template <typename T>
int fill_color_cu(T *image,T *color,int const w,int const h,int const c,int const color_size,T *contrast)
{
    bool flag= false;
    if(c!=color_size)
    {
        std::cout<<"颜色通道不匹配"<<std::endl;
        return 0;
    }
    int wc=w*c;
    //定义显存指针
    T *d_out=NULL;
    T *d_color=NULL;
    //计算显存所需字节数
    size_t const imagebytes=w*h*c*sizeof(T);
    int const colorbytes=color_size* sizeof(T);
    //分配显存
    cudaMalloc((void**)&d_out,imagebytes);
    cudaMalloc((void**)&d_color,colorbytes);
    //cpu2gpu
    cudaMemcpy(d_color,color,colorbytes,cudaMemcpyHostToDevice);

    //线程网格划分
    int x=32;
    int y;

    /**********************************未展开***********************************/
    for ( y = 4; y <=32 ; y<<=1) {
        std::cout<<"block("<<x<<","<<y<<")"<<std::endl;
        dim3 block(x,y,1);
        dim3 grid((wc-1+x)/(x),(h-1+y)/y,1);
        gpu_cpu2zero1<T>(image,d_out,imagebytes);
        kernel_fill_color<T><<<grid,block>>>(d_out,d_color,wc,h,c);
        cudaMemcpy(image,d_out,imagebytes,cudaMemcpyDeviceToHost);
        compare1<T>(image,contrast,w*c,h,flag);
    }
    /**********************************三重展开***********************************/
    for ( y = 4; y <=32 ; y<<=1) {
        std::cout<<"block("<<x<<","<<y<<")"<<std::endl;
        dim3 block(x,y,1);
        dim3 grid((wc-1+x*3)/(x*3),(h-1+y)/y,1);
        gpu_cpu2zero1<T>(image,d_out,imagebytes);
        kernel_fill_color3<T><<<grid,block>>>(d_out,d_color,wc,h,c);
        cudaMemcpy(image,d_out,imagebytes,cudaMemcpyDeviceToHost);
        compare1<T>(image,contrast,w*c,h,flag);
    }
    /**********************************三重展开+共享内存***********************************/
    for ( y = 4; y <=32 ; y<<=1) {
        std::cout<<"block("<<x<<","<<y<<")"<<std::endl;
        dim3 block(x,y,1);
        dim3 grid((wc-1+x*3)/(x*3),(h-1+y)/y,1);
        gpu_cpu2zero1<T>(image,d_out,imagebytes);
        kernel_fill_color3_by_share<T><<<grid,block,colorbytes>>>(d_out,d_color,wc,h,c);
        cudaMemcpy(image,d_out,imagebytes,cudaMemcpyDeviceToHost);
        compare1<T>(image,contrast,w*c,h,flag);
    }
    /**********************************十五重展开+共享内存***********************************/
    for ( y = 4; y <=32 ; y<<=1) {
        std::cout<<"block("<<x<<","<<y<<")"<<std::endl;
        dim3 block(x,y,1);
        dim3 grid((wc-1+x*15)/(x*15),(h-1+y)/y,1);
        gpu_cpu2zero1<T>(image,d_out,imagebytes);
        kernel_fill_color15_by_share<T><<<grid,block,colorbytes>>>(d_out,d_color,wc,h,c);
        cudaMemcpy(image,d_out,imagebytes,cudaMemcpyDeviceToHost);
        compare1<T>(image,contrast,w*c,h,flag);
    }



    /*
    dim3 block(x,y,1);
    dim3 grid((wc-1+x*15)/(x*15),(h-1+y)/y,1);
    gpu_cpu2zero1<T>(image,d_out,imagebytes);
    fcolor15_by_share<T><<<grid,block,colorbytes>>>(d_out,d_color,wc,h,c);
    //gpu2cpu
    cudaMemcpy(image,d_out,imagebytes,cudaMemcpyDeviceToHost);*/

    //释放显存
    cudaFree(d_out);
    cudaFree(d_color);
    return 0;
}
template <typename T>
int fill_color_by_cuda(T *image,T *color,int const w,int const h,int const c,int const color_size,T *contrast)
{
    fill_color_cu<T>(image,color,w,h,c, color_size,contrast);
    return 0;
}
template <>
int fill_color_by_cuda<char>(char *image,char *color,int const w,int const h,int const c,int const color_size,char *contrast)
{
    fill_color_cu<char>(image,color,w,h,c, color_size,contrast);

    return 0;
}
template <>
int fill_color_by_cuda<float>(float  *image,float *color,int const w,int const h,int const c,int const color_size,float *contrast)
{
    fill_color_cu<float>(image,color,w,h,c, color_size,contrast);
    //compare1<float>(image,contrast,w*c,h, true);
    return 0;
}