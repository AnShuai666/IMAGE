/**
 * @desc   image_process.hpp函数实现
 * @author  杨丰拓
 * @date    2019-04-16
 * @email   yangfengtuo@163.com
*/
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
#include "MATH/Function/function.hpp"
#include "cuda_include/sharemem.cuh"
#include "cuda_runtime.h"

#include <vector>

/***********************************************************************************/

__global__ void kernelDesaturateAlpha(float *p_out,float const *p_in, const int kSize,const int kType)
{
    extern __shared__   float s[];
    int in_idx  = threadIdx.x + blockIdx.x * blockDim.x * 8 ;
    int out_idx = threadIdx.x + blockIdx.x * blockDim.x * 4 ;
    int tid=threadIdx.x;
    int stride=tid*4;
    int stride1=stride+blockDim.x*4;
    if (in_idx< kSize * 4)
    {
        s[tid]             =p_in[in_idx];
        s[tid+blockDim.x]  =p_in[in_idx+blockDim.x];
        s[tid+blockDim.x*2]=p_in[in_idx+blockDim.x*2];
        s[tid+blockDim.x*3]=p_in[in_idx+blockDim.x*3];
        s[tid+blockDim.x*4]=p_in[in_idx+blockDim.x*4];
        s[tid+blockDim.x*5]=p_in[in_idx+blockDim.x*5];
        s[tid+blockDim.x*6]=p_in[in_idx+blockDim.x*6];
        s[tid+blockDim.x*7]=p_in[in_idx+blockDim.x*7];
    }
    __syncthreads();

    if(kType==0)
    {
        p_out[out_idx]             =max(s[stride+0],max(s[stride+1],s[stride+2]));
        p_out[out_idx+blockDim.x*2]=max(s[stride1+0],max(s[stride1+1],s[stride1+2]));
    }
    if(kType==1)
    {
        float const max_v = max(s[stride+0],max(s[stride+1],s[stride+2]));
        float const min_v = min(s[stride+0],min(s[stride+1],s[stride+2]));
        p_out[out_idx]=0.5f*(max_v+min_v);
        float const max_s = max(s[stride1+0],max(s[stride1+1],s[stride1+2]));
        float const min_s = min(s[stride1+0],min(s[stride1+1],s[stride1+2]));
        p_out[out_idx+blockDim.x*2]=0.5f*(max_s+min_s);
    }
    if(kType==2)
    {
        p_out[out_idx]             =0.21f * s[stride+0]  + 0.72f * s[stride+1]  + 0.07f * s[stride+2];
        p_out[out_idx+blockDim.x*2]=0.21f * s[stride1+0] + 0.72f * s[stride1+1] + 0.07f * s[stride1+2];
    }
    if(kType==3)
    {
        p_out[out_idx]             =0.30f * s[stride+0]  + 0.59f * s[stride+1]  + 0.11f * s[stride+2];
        p_out[out_idx+blockDim.x*2]=0.30f * s[stride1+0] + 0.59f * s[stride1+1] + 0.11f * s[stride1+2];
    }
    if(kType==4)
    {
        p_out[out_idx]             =((float)(s[stride+0]  + s[stride+1]  + s[stride+2])) / 3.0f;
        p_out[out_idx+blockDim.x*2]=((float)(s[stride1+0] + s[stride1+1] + s[stride1+2])) / 3.0f;
    }
    p_out[out_idx+tid+1]             =s[stride+3];
    p_out[out_idx+blockDim.x*2+tid+1]=s[stride1+3];
}
__global__ void kernelDesaturate(float *p_out,float const *p_in, const int kSize,const int kType)
{
    extern __shared__   float s[];
    int in_idx  = threadIdx.x + blockIdx.x * blockDim.x * 6 ;
    int out_idx = threadIdx.x + blockIdx.x * blockDim.x * 2 ;
    int tid=threadIdx.x;
    int stride=tid*3;
    int stride1=stride+blockDim.x*3;

    if (in_idx< kSize * 3)
    {
        s[tid]             =p_in[in_idx];
        s[tid+blockDim.x]  =p_in[in_idx+blockDim.x];
        s[tid+blockDim.x*2]=p_in[in_idx+blockDim.x*2];
        s[tid+blockDim.x*3]=p_in[in_idx+blockDim.x*3];
        s[tid+blockDim.x*4]=p_in[in_idx+blockDim.x*4];
        s[tid+blockDim.x*5]=p_in[in_idx+blockDim.x*5];
    }
    __syncthreads();
    if(kType==0)
    {
        p_out[out_idx]           =max(s[stride+0],max(s[stride+1],s[stride+2]));
        p_out[out_idx+blockDim.x]=max(s[stride1+0],max(s[stride1+1],s[stride1+2]));
    }
    if(kType==1)
    {
        float const max_v = max(s[stride+0],max(s[stride+1],s[stride+2]));
        float const min_v = min(s[stride+0],min(s[stride+1],s[stride+2]));
        p_out[out_idx]=0.5f*(max_v+min_v);
        float const max_s = max(s[stride1+0],max(s[stride1+1],s[stride1+2]));
        float const min_s = min(s[stride1+0],min(s[stride1+1],s[stride1+2]));
        p_out[out_idx+blockDim.x]=0.5f*(max_s+min_s);
    }
    if(kType==2)
    {
        p_out[out_idx]           =0.21f * s[stride+0]  + 0.72f * s[stride+1]  + 0.07f * s[stride+2];
        p_out[out_idx+blockDim.x]=0.21f * s[stride1+0] + 0.72f * s[stride1+1] + 0.07f * s[stride1+2];
    }
    if(kType==3)
    {
        p_out[out_idx]           =0.30f * s[stride+0]  + 0.59f * s[stride+1]  + 0.11f * s[stride+2];
        p_out[out_idx+blockDim.x]=0.30f * s[stride1+0] + 0.59f * s[stride1+1] + 0.11f * s[stride1+2];
    }
    if(kType==4)
    {
        p_out[out_idx]           =((float)(s[stride+0]  + s[stride+1]  + s[stride+2])) / 3.0f;
        p_out[out_idx+blockDim.x]=((float)(s[stride1+0] + s[stride1+1] + s[stride1+2])) / 3.0f;
    }
}
/******************************************************************************************/
///功能：图片放大两倍
/*  函数名                         线程块大小       耗费时间
 *  kernelDoubleSize               3.678ms	    [32,4,1]
 *  kernelDoubleSize1              3.67ms	    [32,4,1]
 *  kernelDoubleSize2              3.532ms	    [32,4,1]**
 *  kernelDoubleSizeByShare        5.265ms	    [32,8,1]
 *  kernelDoubleSizeByShare1       4.737ms	    [64,8,1]
 *  kernelDoubleSizeByShare2       3.98ms	    [32,8,1]
 */
/******************************************************************************************/
/* 调用示例
 * dim3 block (x,y,1);
 * dim3 grid ((ow-1+x)/x,(oh-1+y)/y,1);
 * kernelDoublesize<<<grid,block>>>(d_out,d_in,ow,oh,width,channels);
 */
__global__ void kernelDoubleSize(float *p_out,float *p_in,int const kImage_x,int const kImage_y,int const kIn_width,int const kIn_Channels)
{
    int out_x = threadIdx.x + blockIdx.x * blockDim.x * kIn_Channels;
    int out_y = threadIdx.y + blockIdx.y * blockDim.y;
    for (int c = 0; c <kIn_Channels ; ++c)
    {
        int fact_x = out_x + blockDim.x * c;
        if(out_y<kImage_y && fact_x < kImage_x*kIn_Channels)
        {
            int  idx   =fact_x + out_y * kImage_x * kIn_Channels;
            bool nexty =(out_y+1)<kImage_y;
            bool nextx =(fact_x+kIn_Channels)<(kImage_x*kIn_Channels);
            int yoff[2]={kIn_Channels*kIn_width*(out_y>>1),
                         kIn_Channels*kIn_width*((out_y+nexty)>>1)};
            int xoff[2]={((fact_x / kIn_Channels) >>1)* kIn_Channels + fact_x % kIn_Channels,
                         (((fact_x/kIn_Channels)+nextx)>>1)*kIn_Channels+fact_x%kIn_Channels};
            int index[4]={yoff[0]+xoff[0],
                          yoff[0]+xoff[1],
                          yoff[1]+xoff[0],
                          yoff[1]+xoff[1]};
            p_out[idx]=0.25f*(p_in[index[0]]+p_in[index[1]]+p_in[index[2]]+p_in[index[3]]);
        }
    }
}
/* 调用示例
 * dim3 block (x,y,1);
 * dim3 grid ((ow-1+x*2)/(x*2),(oh-1+y)/y,1);
 * kernel_doublesize1<<<grid,block>>>(d_out,d_in,ow,oh,width,channels);
 */
__global__ void kernelDoubleSize1(float *p_out,float *p_in,int const kImage_x,int const kImage_y,int const kIn_width,int const kIn_Channels)
{
    int out_x = threadIdx.x + blockIdx.x * blockDim.x * kIn_Channels*2;
    int out_y = threadIdx.y + blockIdx.y * blockDim.y;
    for (int c = 0; c <kIn_Channels*2 ; ++c)
    {
        int fact_x=out_x+blockDim.x*c;
        if(out_y<kImage_y&&fact_x<kImage_x*kIn_Channels)
        {
            int  idx=fact_x+out_y*kImage_x*kIn_Channels;
            bool nexty=(out_y+1)<kImage_y;
            bool nextx=(fact_x+kIn_Channels)<(kImage_x*kIn_Channels);
            int yoff[2]={kIn_Channels*kIn_width*(out_y>>1),
                         kIn_Channels*kIn_width*((out_y+nexty)>>1)};
            int xoff[2]={((fact_x/kIn_Channels)>>1)*kIn_Channels+fact_x%kIn_Channels,
                         (((fact_x/kIn_Channels)+nextx)>>1)*kIn_Channels+fact_x%kIn_Channels};
            int index[4]={yoff[0]+xoff[0],
                          yoff[0]+xoff[1],
                          yoff[1]+xoff[0],
                          yoff[1]+xoff[1]};
            p_out[idx]=0.25f*(p_in[index[0]]+p_in[index[1]]+p_in[index[2]]+p_in[index[3]]);
        }
    }
}
/* 调用示例
 * dim3 block (x,y,1);
 * dim3 grid ((ow-1+x*3)/(x*3),(oh-1+y)/y,1);
 * kernel_doublesize2<<<grid,block>>>(d_out,d_in,ow,oh,width,channels);
*/
__global__ void kernelDoubleSize2(float *p_out,float *p_in,int const kImage_x,int const kImage_y,int const kIn_width,int const kIn_Channels)
{
    int out_x = threadIdx.x + blockIdx.x * blockDim.x * kIn_Channels*3;
    int out_y = threadIdx.y + blockIdx.y * blockDim.y;
    for (int c = 0; c <kIn_Channels*3 ; ++c)
    {
        int fact_x=out_x+blockDim.x*c;
        if(out_y<kImage_y&&fact_x<kImage_x*kIn_Channels)
        {
            int idx=fact_x+out_y*kImage_x*kIn_Channels;
            bool nexty=(out_y+1)<kImage_y;
            bool nextx=(fact_x+kIn_Channels)<(kImage_x*kIn_Channels);
            int yoff[2]={kIn_Channels*kIn_width*(out_y>>1),
                         kIn_Channels*kIn_width*((out_y+nexty)>>1)};
            int xoff[2]={((fact_x/kIn_Channels)>>1)*kIn_Channels+fact_x%kIn_Channels,
                         (((fact_x/kIn_Channels)+nextx)>>1)*kIn_Channels+fact_x%kIn_Channels};
            int index[4]={yoff[0]+xoff[0],
                          yoff[0]+xoff[1],
                          yoff[1]+xoff[0],
                          yoff[1]+xoff[1]};
            p_out[idx]=0.25f*(p_in[index[0]]+p_in[index[1]]+p_in[index[2]]+p_in[index[3]]);
        }
    }
}
/* 调用示例
 * dim3 block (x,y,1);
 * dim3 grid ((ow-1+x)/x,(oh-1+y)/y,1);
 * kernel_doublesizebyshare<<<grid,block,share_x*share_y*channels*sizeof(float)>>>(d_out,d_in,ow,oh,width,height,channels);
*/
__global__ void kernelDoubleSizeByShare(float *p_out,float *p_in,int const kOut_width,int const kOut_height,int const kIn_width,int const kIn_height,int const kIn_Channels)
{
    extern __shared__ float  data[];
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*kIn_Channels;
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;

    int share_x=(blockDim.x>>1)+1;//共享内存块x维（需乘kIn_Channels）
    int share_y=(blockDim.y>>1)+1;//共享内存块y维
    int share_fact_x=share_x*kIn_Channels;
    int share_idx_x;
    int share_idx_y= threadIdx.y;//共享内存块内y维索引
    int in_x0 = ((blockIdx.x * blockDim.x) >> 1) * kIn_Channels;
    int in_y0 = (blockIdx.y * blockDim.y) >> 1;
    int x,y,c,fact_x;

    for ( c = 0; c <kIn_Channels ; ++c)
    {
        share_idx_x = threadIdx.x + blockDim.x * c;//共享内存块内x索引
        if (share_idx_x < share_fact_x && share_idx_y < share_y)
        {
            x = min(in_x0 + share_idx_x, kIn_width * kIn_Channels - kIn_Channels + share_idx_x % kIn_Channels);
            y = min(in_y0 + share_idx_y, kIn_height - 1);
            data[share_idx_y * share_fact_x + share_idx_x] = p_in[y * kIn_width * kIn_Channels + x];
        }

    }
    __syncthreads();
    for ( c = 0; c <kIn_Channels ; ++c)
    {
        fact_x=out_x+blockDim.x*c;
        if(out_y<kOut_height && fact_x<kOut_width*kIn_Channels)
        {
            share_idx_x = threadIdx.x + blockDim.x * c;
            int yoff[2]={(share_idx_y>>1)*share_fact_x,((share_idx_y+1)>>1)*share_fact_x};
            int xoff[2]={(share_idx_x/kIn_Channels>>1)*kIn_Channels+share_idx_x%kIn_Channels,
                         ((share_idx_x/kIn_Channels+1)>>1)*kIn_Channels+share_idx_x%kIn_Channels};
            int out_idx=out_y*kOut_width*kIn_Channels+fact_x;
            int index[4]={yoff[0]+xoff[0],
                          yoff[0]+xoff[1],
                          yoff[1]+xoff[0],
                          yoff[1]+xoff[1]};
            p_out[out_idx]=0.25f*(data[index[0]]+data[index[1]]+data[index[2]]+data[index[3]]);
        }
    }
}
/* 调用示例
 * dim3 block (x,y,1);
 * dim3 grid ((kOut_width-1+x*2)/(x*2),(oh-1+y)/y,1);
 * kernel_doublesizebyshare1<<<grid,block,share_x*share_y*2*channels*sizeof(float)>>>(d_out,d_in,kOut_width,oh,width,height,channels);
*/
__global__ void kernelDoubleSizeByShare1(float *p_out,float *p_in,int const kOut_width,int const kOut_height,int const kIn_width,int const kIn_height,int const kIn_Channels)
{
    extern __shared__ float  data[];
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*kIn_Channels*2;
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;

    int share_x=(blockDim.x>>1)+1;//共享内存块x维（需乘kIn_Channels）
    int share_y=(blockDim.y>>1)+1;//共享内存块y维
    int share_fact_x=share_x*kIn_Channels*2;
    int share_idx_x;
    int share_idx_y= threadIdx.y;//共享内存块内y维索引
    int in_x0 = ((blockIdx.x * blockDim.x*2) >> 1) * kIn_Channels;
    int in_y0 = (blockIdx.y * blockDim.y) >> 1;
    int x,y,c,fact_x;

    for ( c = 0; c <kIn_Channels*2 ; ++c)
    {
        share_idx_x = threadIdx.x + blockDim.x * c;//共享内存块内x索引
        if (share_idx_x < share_fact_x && share_idx_y < share_y)
        {
            x = min(in_x0 + share_idx_x, kIn_width * kIn_Channels - kIn_Channels + share_idx_x % kIn_Channels);
            y = min(in_y0 + share_idx_y, kIn_height - 1);
            data[share_idx_y * share_fact_x + share_idx_x] = p_in[y * kIn_width * kIn_Channels + x];
        }

    }
    __syncthreads();
    for ( c = 0; c <kIn_Channels*2 ; ++c)
    {
        fact_x=out_x+blockDim.x*c;
        if(out_y<kOut_height&&fact_x<kOut_width*kIn_Channels)
        {
            share_idx_x = threadIdx.x + blockDim.x * c;
            int yoff[2]={(share_idx_y>>1)*share_fact_x,((share_idx_y+1)>>1)*share_fact_x};
            int xoff[2]={(share_idx_x/kIn_Channels>>1)*kIn_Channels+share_idx_x%kIn_Channels,
                         ((share_idx_x/kIn_Channels+1)>>1)*kIn_Channels+share_idx_x%kIn_Channels};
            int out_idx=out_y*kOut_width*kIn_Channels+fact_x;
            int index[4]={yoff[0]+xoff[0],
                          yoff[0]+xoff[1],
                          yoff[1]+xoff[0],
                          yoff[1]+xoff[1]};
            p_out[out_idx]=0.25f*(data[index[0]]+data[index[1]]+data[index[2]]+data[index[3]]);
        }
    }
}
/* 调用示例
 * dim3 block (x,y,1);
 * dim3 grid ((kOut_width-1+x*3)/(x*3),(kOut_height-1+y)/y,1);
 * kernel_doublesizebyshare2<<<grid,block,share_x*share_y*3*channels*sizeof(float)>>>(d_out,d_in,kOut_width,kOut_height,width,height,channels);
 */
__global__ void kernelDoubleSizeByShare2(float *p_out,float *p_in,int const kOut_width,int const kOut_height,int const kIn_width,int const kIn_height,int const kIn_Channels)
{
    extern __shared__ float  data[];
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*kIn_Channels*3;
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;

    int share_x=(blockDim.x>>1)+1;//共享内存块x维（需乘kIn_Channels）
    int share_y=(blockDim.y>>1)+1;//共享内存块y维
    int share_fact_x=share_x*kIn_Channels*3;
    int share_idx_x;
    int share_idx_y = threadIdx.y;//共享内存块内y维索引
    int in_x0 = ((blockIdx.x * blockDim.x*3) >> 1) * kIn_Channels;
    int in_y0 = (blockIdx.y * blockDim.y) >> 1;
    int x,y,c,fact_x;

    for ( c = 0; c <kIn_Channels*3 ; ++c)
    {
        share_idx_x = threadIdx.x + blockDim.x * c;//共享内存块内x索引
        if (share_idx_x < share_fact_x && share_idx_y < share_y)
        {
            x = min(in_x0 + share_idx_x, kIn_width * kIn_Channels - kIn_Channels + share_idx_x % kIn_Channels);
            y = min(in_y0 + share_idx_y, kIn_height - 1);
            data[share_idx_y * share_fact_x + share_idx_x] = p_in[y * kIn_width * kIn_Channels + x];
        }

    }
    __syncthreads();
    for ( c = 0; c <kIn_Channels*3 ; ++c)
    {
        fact_x=out_x+blockDim.x*c;
        if(out_y<kOut_height&&fact_x<kOut_width*kIn_Channels)
        {
            share_idx_x = threadIdx.x + blockDim.x * c;
            int yoff[2]={(share_idx_y>>1)*share_fact_x,((share_idx_y+1)>>1)*share_fact_x};
            int xoff[2]={(share_idx_x/kIn_Channels>>1)*kIn_Channels+share_idx_x%kIn_Channels,
                         ((share_idx_x/kIn_Channels+1)>>1)*kIn_Channels+share_idx_x%kIn_Channels};
            int out_idx=out_y*kOut_width*kIn_Channels+fact_x;
            int index[4]={yoff[0]+xoff[0],
                          yoff[0]+xoff[1],
                          yoff[1]+xoff[0],
                          yoff[1]+xoff[1]};
            p_out[out_idx]=0.25f*(data[index[0]]+data[index[1]]+data[index[2]]+data[index[3]]);
        }
    }
}

/******************************************************************************************/
///功能：图片缩小两倍
/*  函数名                            线程块大小       耗费时间
 *kernelHalfSize		            636.275us	    [32,8,1]
 *kernelHalfSize1                   634.383us	    [32,8,1]**
 *kernelHalfSize2                   641.6us	        [32,8,1]
 *kernelHalfSizeByShare	    	    643.698us	    [32,4,1]
 *kernelHalfSizeByShare1	  		671.245us	    [32,4,1]
 */
/******************************************************************************************/
/* 调用示例
 * dim3 block (x,y,1);
 * dim3 grid ((kOut_width-1+x)/x,(kOut_height-1+y)/y,1);
 * kernel_halfsize<<<grid,block>>>(d_out,d_in,kOut_width,kOut_height,width,height,channels);
 */
__global__ void kernelHalfSize(float *p_out,float *p_in,int const kOut_width,int const kOut_height,int const kIn_width,int const kIn_height,int const kIn_Channels)
{
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*kIn_Channels;
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;
    int stride=kIn_width*kIn_Channels;

    for(int c=0;c<kIn_Channels;c++)
    {
        int fact_x=out_x+blockDim.x*c;
        if(out_y<kOut_height&&fact_x<kOut_width*kIn_Channels) {
            int irow1 = out_y * 2 * stride;
            int irow2 = irow1 + stride * (out_y * 2 + 1 < kIn_height);
            int icol1 = (fact_x / kIn_Channels) * 2 * kIn_Channels + fact_x % kIn_Channels;
            int icol2 = min((icol1 + kIn_Channels), (kIn_width * kIn_Channels - kIn_Channels + fact_x % kIn_Channels));
            int index[4] = {irow1 + icol1,
                            irow1 + icol2,
                            irow2 + icol1,
                            irow2 + icol2};
            int out_idx = out_y * kOut_width*kIn_Channels + fact_x;
            p_out[out_idx] = 0.25f * (p_in[index[0]] + p_in[index[1]] + p_in[index[2]] + p_in[index[3]]);
        }
    }
}
/* 调用示例
 * dim3 block (x,y,1);
 * dim3 grid ((kOut_width-1+x*2)/(x*2),(kOut_height-1+y)/y,1);
 * kernel_halfsize1<<<grid,block>>>(d_out,d_in,kOut_width,kOut_height,width,height,channels);
 */
__global__ void kernelHalfSize1(float *p_out,float *p_in,int const kOut_width,int const kOut_height,int const kIn_width,int const kIn_height,int const kIn_Channels)
{
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*kIn_Channels*2;
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;
    int stride=kIn_width*kIn_Channels;

    for(int c=0;c<kIn_Channels*2;c++)
    {
        int fact_x=out_x+blockDim.x*c;
        if(out_y<kOut_height&&fact_x<kOut_width*kIn_Channels) {
            int irow1 = out_y * 2 * stride;
            int irow2 = irow1 + stride * (out_y * 2 + 1 < kIn_height);
            int icol1 = (fact_x / kIn_Channels) * 2 * kIn_Channels + fact_x % kIn_Channels;
            int icol2 = min((icol1 + kIn_Channels), (kIn_width * kIn_Channels - kIn_Channels + fact_x % kIn_Channels));
            int index[4] = {irow1 + icol1,
                            irow1 + icol2,
                            irow2 + icol1,
                            irow2 + icol2};
            int out_idx = out_y * kOut_width*kIn_Channels + fact_x;
            p_out[out_idx] = 0.25f * (p_in[index[0]] + p_in[index[1]] + p_in[index[2]] + p_in[index[3]]);
        }
    }
}
/* 调用示例
 * dim3 block (x,y,1);
 * dim3 grid ((kOut_width-1+x*3)/(x*3),(kOut_height-1+y)/y,1);
 * kernel_halfsize2<<<grid,block>>>(d_out,d_in,kOut_width,kOut_height,width,height,channels);
 */
__global__ void kernelHalfSize2(float *p_out,float *p_in,int const kOut_width,int const kOut_height,int const kIn_width,int const kIn_height,int const kIn_Channels)
{
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*kIn_Channels*3;
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;
    int stride=kIn_width*kIn_Channels;

    for(int c=0;c<kIn_Channels*3;c++)
    {
        int fact_x=out_x+blockDim.x*c;
        if(out_y<kOut_height && fact_x < kOut_width*kIn_Channels) {
            int irow1 = out_y * 2 * stride;
            int irow2 = irow1 + stride * (out_y * 2 + 1 < kIn_height);
            int icol1 = (fact_x / kIn_Channels) * 2 * kIn_Channels + fact_x % kIn_Channels;
            int icol2 = min((icol1 + kIn_Channels), (kIn_width * kIn_Channels - kIn_Channels + fact_x % kIn_Channels));
            int index[4] = {irow1 + icol1,
                            irow1 + icol2,
                            irow2 + icol1,
                            irow2 + icol2};
            int out_idx = out_y * kOut_width*kIn_Channels + fact_x;
            p_out[out_idx] = 0.25f * (p_in[index[0]] + p_in[index[1]] + p_in[index[2]] + p_in[index[3]]);
        }
    }
}
/* 调用示例
 * dim3 block (x,y,1);
 * dim3 grid ((kOut_width-1+x)/x,(kOut_height-1+y)/y,1);
 * kernel_halfsizebyshare<<<grid,block,share_x*share_y*channels* sizeof(float)>>>(d_out,d_in,kOut_width,kOut_height,width,height,channels);
 */
__global__ void kernelHalfSizeByShare(float *p_out,float *p_in,int const kOut_width,int const kOut_height,int const kIn_width,int const kIn_height,int const kIn_Channels)
{
    extern __shared__ float data[];
    int block_stride=blockDim.x*kIn_Channels;//线程块x维间隔
    int out_x=threadIdx.x+blockIdx.x*block_stride;//输出的x维起始索引
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;//输出的y位索引
    int stride=kIn_width*kIn_Channels;//输入图像的行索引的最大值

    int in_x0=blockIdx.x*block_stride*2;//输入图像x维的起始点
    int in_y0=blockIdx.y*blockDim.y*2;//输入图像y维的起始点
    int in_x1=in_x0+block_stride;
    int in_y1=in_y0+blockDim.y;

    int share_x=blockDim.x*2*kIn_Channels;//共享块内x维最大像素点个数
    for (int c = 0; c < kIn_Channels; ++c)
    {
        int fact_x_s=threadIdx.x+blockDim.x*c;
        int channel=fact_x_s%kIn_Channels;//第几个通道
        int x_s = fact_x_s + block_stride;
        int y_s0=threadIdx.y*share_x;
        int y_s1=y_s0+blockDim.y*share_x;
        int fact_iw=channel+stride-kIn_Channels;
        int x0=min(in_x0+fact_x_s,fact_iw);
        int x1=min(in_x1+fact_x_s,fact_iw);
        int y0=min(in_y0+threadIdx.y,kIn_height-1)*stride;
        int y1=min(in_y1+threadIdx.y,kIn_height-1)*stride;

        int deta=((fact_x_s/kIn_Channels)%2)*block_stride;//像素点的x坐标是否为奇数

        int x_fs0=(fact_x_s/kIn_Channels>>1)*kIn_Channels+channel+deta;//共享内存内存储第一个x坐标
        int x_fs1=(x_s/kIn_Channels>>1)*kIn_Channels+channel+deta;//共享内存内存储第二个x坐标

        data[y_s0+x_fs0]=p_in[y0+x0];
        data[y_s0+x_fs1]=p_in[y0+x1];
        data[y_s1+x_fs0]=p_in[y1+x0];
        data[y_s1+x_fs1]=p_in[y1+x1];;
    }
    __syncthreads();
    for (int c = 0; c <kIn_Channels ; ++c)
    {
        int fact_x=out_x+blockDim.x*c;

        if(out_y<kOut_height&&fact_x<kOut_width*kIn_Channels)
        {
            int srow1=threadIdx.y*2*share_x;
            int srow2=srow1+share_x;

            int scol1=threadIdx.x+blockDim.x*c;
            int scol2=scol1+block_stride;
            int index[4] = {srow1 + scol1,
                            srow1 + scol2,
                            srow2 + scol1,
                            srow2 + scol2};
            int out_idx = out_y * kOut_width*kIn_Channels + fact_x;
            p_out[out_idx] = 0.25f * (data[index[0]] + data[index[1]] + data[index[2]] + data[index[3]]);
        }
    }
}
/* 调用示例
 * dim3 block (x,y,1);
 * dim3 grid ((kOut_width-1+x*2)/(x*2),(kOut_height-1+y)/y,1);
 * kernel_halfsizebyshare1<<<grid,block,share_x*share_y*channels* sizeof(float)>>>(d_out,d_in,kOut_width,kOut_height,width,height,channels);
 */
__global__ void kernelHalfSizeByShare1(float *p_out,float *p_in,int const kOut_width,int const kOut_height,int const kIn_width,int const kIn_height,int const kIn_Channels)
{
    extern __shared__ float data[];
    int block_stride=blockDim.x*kIn_Channels*2;//线程块x维间隔
    int out_x=threadIdx.x+blockIdx.x*block_stride;//输出的x维起始索引
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;//输出的y位索引
    int stride=kIn_width*kIn_Channels;//输入图像的行索引的最大值

    int in_x0=blockIdx.x*block_stride*2;//输入图像x维的起始点
    int in_y0=blockIdx.y*blockDim.y*2;//输入图像y维的起始点
    int in_x1=in_x0+block_stride;
    int in_y1=in_y0+blockDim.y;

    int share_x=blockDim.x*4*kIn_Channels;//共享块内x维最大像素点个数
    for (int c = 0; c < kIn_Channels*2; ++c)
    {

        int fact_x_s=threadIdx.x+blockDim.x*c;
        int channel=fact_x_s%kIn_Channels;//第几个通道
        int x_s=fact_x_s+block_stride;
        int y_s0=threadIdx.y*share_x;
        int y_s1=y_s0+blockDim.y*share_x;
        int fact_iw=channel+stride-kIn_Channels;
        int x0=min(in_x0+fact_x_s,fact_iw);
        int x1=min(in_x1+fact_x_s,fact_iw);
        int y0=min(in_y0+threadIdx.y,kIn_height-1)*stride;
        int y1=min(in_y1+threadIdx.y,kIn_height-1)*stride;

        int deta=((fact_x_s/kIn_Channels)%2)*block_stride;//像素点的x坐标是否为奇数

        int x_fs0=(fact_x_s/kIn_Channels>>1)*kIn_Channels+channel+deta;//共享内存内存储第一个x坐标
        int x_fs1=(x_s/kIn_Channels>>1)*kIn_Channels+channel+deta;//共享内存内存储第二个x坐标

        data[y_s0+x_fs0]=p_in[y0+x0];
        data[y_s0+x_fs1]=p_in[y0+x1];
        data[y_s1+x_fs0]=p_in[y1+x0];
        data[y_s1+x_fs1]=p_in[y1+x1];;
    }
    __syncthreads();
    for (int c = 0; c <kIn_Channels*2 ; ++c)
    {
        int fact_x=out_x+blockDim.x*c;

        if(out_y<kOut_height&&fact_x<kOut_width*kIn_Channels)
        {
            int srow1=threadIdx.y*2*share_x;
            int srow2=srow1+share_x;

            int scol1=threadIdx.x+blockDim.x*c;
            int scol2=scol1+block_stride;
            int index[4] = {srow1 + scol1,
                            srow1 + scol2,
                            srow2 + scol1,
                            srow2 + scol2};
            int out_idx = out_y * kOut_width*kIn_Channels + fact_x;
            p_out[out_idx] = 0.25f * (data[index[0]] + data[index[1]] + data[index[2]] + data[index[3]]);
        }
    }
}
/******************************************************************************************/
///功能：高斯权值降采样
/*  函数名                            线程块大小       耗费时间
 * kernel_halfsize_gauss	        1.856ms	        [32,8,1]
 * kernel_halfsize_gauss1	        936.937us	    [32,4,1]
 */
/******************************************************************************************/
/* 调用示例
 * dim3 block(x, y, 1);
 * dim3 grid((ow - 1 + x) / (x), (kOut_height - 1 + y) / y, 1);
 * kernel_halfsize_guass << < grid, block >> > (d_out, d_in, ow, oh, width, height, channels, d_w);
 *
__global__ void kernel_halfsize_guass(float *out,float *in,int const ow,int const oh,int const iw,int const ih,int const ic,float const *w)
{
    //多余时间损耗原因为printf("%1.10f\t%1.10f\n",sum,in[row[2] + col[2]] * dw[0]);中又访问了in数组
    //注释掉printf函数后，时间与kernel_halfsize_guass1相差不多
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*ic;
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;
    int istride=iw*ic;
    float dw[3];
    dw[0]=w[0];
    dw[1]=w[1];
    dw[2]=w[2];

    for (int c = 0; c <ic ; ++c)
    {
        int fact_x=out_x+blockDim.x*c;
        if(out_y<oh&&fact_x<ow*ic)
        {
            int out_idx = out_y * ow * ic + fact_x;
            int channels = fact_x % ic;//颜色通道
            int out_xf = fact_x / ic;//输出像素点x坐标
            int ix = out_xf << 1;
            int iy = out_y << 1;
            int row[4], col[4];
            row[0] = max(0, iy - 1) * istride;
            row[1] = iy * istride;
            row[2] = min(iy + 1, (int)ih - 1) * istride;
            row[3] = min(iy + 2, (int)ih - 2) * istride;

            col[0] = max(0, ix - 1) * ic + channels;
            col[1] = ix * ic + channels;
            col[2] = min(ix + 1, (int)iw - 1) * ic + channels;
            col[3] = min(ix + 2, (int)iw - 1) * ic + channels;

            float sum = 0.0f;
            int t=6;
            if(out_idx==t);//printf("idx:%d\n",t);
            sum += in[row[0] + col[0]] * dw[2];
            if(out_idx==t)printf("%1.10f\t%1.10f\n",sum,in[row[0] + col[0]] * dw[2]);
            sum += in[row[0] + col[1]] * dw[1];
            if(out_idx==t)printf("%1.10f\t%1.10f\n",sum,in[row[0] + col[1]] * dw[1]);
            sum += in[row[0] + col[2]] * dw[1];
            if(out_idx==t)printf("%1.10f\t%1.10f\n",sum,in[row[0] + col[2]] * dw[1]);
            sum += in[row[0] + col[3]] * dw[2];
            if(out_idx==t)printf("%1.10f\t%1.10f\n",sum,in[row[0] + col[3]] * dw[2]);

            sum += in[row[1] + col[0]] * dw[1];
            if(out_idx==t)printf("%1.10f\t%1.10f\n",sum,in[row[1] + col[0]] * dw[1]);
            sum += in[row[1] + col[1]] * dw[0];
            if(out_idx==t)printf("%1.10f\t%1.10f\n",sum,in[row[1] + col[1]] * dw[0]);
            sum += in[row[1] + col[2]] * dw[0];
            if(out_idx==t)printf("%1.10f\t%1.10f\n",sum,in[row[1] + col[2]] * dw[0]);
            sum += in[row[1] + col[3]] * dw[1];
            if(out_idx==t)printf("%1.10f\t%1.10f\n",sum,in[row[1] + col[3]] * dw[1]);

            sum += in[row[2] + col[0]] * dw[1];
            if(out_idx==t)printf("%1.10f\t%1.10f\n",sum,in[row[2] + col[0]] * dw[1]);
            sum += in[row[2] + col[1]] * dw[0];
            if(out_idx==t)printf("%1.10f\t%1.10f\n",sum,in[row[2] + col[1]] * dw[0]);
            sum += in[row[2] + col[2]] * dw[0];
            if(out_idx==t)printf("%1.10f\t%1.10f\n",sum,in[row[2] + col[2]] * dw[0]);
            sum += in[row[2] + col[3]] * dw[1];
            if(out_idx==t)printf("%1.10f\t%1.10f\n",sum,in[row[2] + col[3]] * dw[1]);

            sum += in[row[3] + col[0]] * dw[2];
            if(out_idx==t)printf("%1.10f\t%1.10f\n",sum,in[row[3] + col[0]] * dw[2]);
            sum += in[row[3] + col[1]] * dw[1];
            if(out_idx==t)printf("%1.10f\t%1.10f\n",sum,in[row[3] + col[1]] * dw[1]);
            sum += in[row[3] + col[2]] * dw[1];
            if(out_idx==t)printf("%1.10f\t%1.10f\n",sum,in[row[3] + col[2]] * dw[1]);
            sum += in[row[3] + col[3]] * dw[2];
            if(out_idx==t)printf("%1.10f\t%1.10f\n",sum,in[row[3] + col[3]] * dw[2]);

            out[out_idx] = sum / (float)(4 * dw[2] + 8 * dw[1] + 4 * dw[0]);
        }
    }
}
*/
/* 调用示例
 * dim3 block(x, y, 1);
 * dim3 grid((kOut_width - 1 + x) / (x), (oh - 1 + y) / y, 1);
 * kernel_halfsize_gauss1 << < grid, block >> > (d_out, d_in, kOut_width, kOut_height, width, height, channels, d_w);
 */
__global__ void kernelHalfSizeGauss1(float *p_out,float *p_in,int const kOut_width,int const kOut_height,int const kIn_width,int const kIn_height,int const kIn_Channels,float const *p_w)
{
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*kIn_Channels;
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;
    int istride=kIn_width*kIn_Channels;
    float dw[3];
    dw[0]=p_w[0];
    dw[1]=p_w[1];
    dw[2]=p_w[2];

    for (int c = 0; c <kIn_Channels ; ++c)
    {
        int fact_x=out_x+blockDim.x*c;
        if(out_y<kOut_height&&fact_x<kOut_width*kIn_Channels)
        {
            int out_idx = out_y * kOut_width * kIn_Channels + fact_x;
            int channels = fact_x % kIn_Channels;//颜色通道
            int out_xf = fact_x / kIn_Channels;//输出像素点x坐标
            int ix = out_xf << 1;
            int iy = out_y << 1;
            int row[4], col[4];
            row[0] = max(0, iy - 1) * istride;
            row[1] = iy * istride;
            row[2] = min(iy + 1, (int)kIn_height - 1) * istride;
            row[3] = min(iy + 2, (int)kIn_height - 2) * istride;

            col[0] = max(0, ix - 1) * kIn_Channels + channels;
            col[1] = ix * kIn_Channels + channels;
            col[2] = min(ix + 1, (int)kIn_width - 1) * kIn_Channels + channels;
            col[3] = min(ix + 2, (int)kIn_width - 1) * kIn_Channels + channels;

            float sum = 0.0f;
            sum+=p_in[row[0] + col[0]] * dw[2];
            sum+=p_in[row[0] + col[1]] * dw[1];
            sum+=p_in[row[0] + col[2]] * dw[1];
            sum+=p_in[row[0] + col[3]] * dw[2];

            sum+=p_in[row[1] + col[0]] * dw[1];
            sum+=p_in[row[1] + col[1]] * dw[0];
            sum+=p_in[row[1] + col[2]] * dw[0];
            sum+=p_in[row[1] + col[3]] * dw[1];

            sum+=p_in[row[2] + col[0]] * dw[1];
            sum+=p_in[row[2] + col[1]] * dw[0];
            sum+=p_in[row[2] + col[2]] * dw[0];
            sum+=p_in[row[2] + col[3]] * dw[1];

            sum+=p_in[row[3] + col[0]] * dw[2];
            sum+=p_in[row[3] + col[1]] * dw[1];
            sum+=p_in[row[3] + col[2]] * dw[1];
            sum+=p_in[row[3] + col[3]] * dw[2];

            p_out[out_idx] = sum / (float)(4 * dw[2] + 8 * dw[1] + 4 * dw[0]);
        }
    }
}

/******************************************************************************************/
///功能：x维高斯模糊
/*  函数名                            线程块大小       耗费时间
 *  kernelGaussBlurX	             2.561ms	    [32,4,1]
 *  kernelGaussBlurX1	             2.025ms	    [32,4,1]**
 *  kernelGaussBlurX2	             2.148ms	    [32,4,1]
 */
/******************************************************************************************/
/* 调用示例
* dim3 block(x,y,1);
* dim3 grid((w*c-1+x)/(x),(h-1+y)/y,1);
* kernel_gaussBlur_x<<<grid,block,(ks+1)* sizeof(float)>>>(d_tmp,d_in,d_blur,w,h,c,ks,weight);
*/
__global__ void kernelGaussBlurX(float *const p_out,float const *const p_in,float const * const p_blur,int const kWidth,int const kHeight,int const kChannels,int const kSize,float const kWeight)
{
    extern __shared__ float data[];
    int x=threadIdx.x+blockIdx.x*blockDim.x;
    int y=threadIdx.y+blockIdx.y*blockDim.y;
    int share_idx=threadIdx.y*blockDim.x+threadIdx.x;
    if((share_idx)<(kSize+1))
    {
        data[share_idx]=p_blur[share_idx];
    }
    __syncthreads();
    int fact_x=x/kChannels;
    int channels=x%kChannels;
    int max_x=y*kWidth*kChannels;
    int out_idx=max_x+x;
    if(fact_x<kWidth&&y<kHeight)
    {
        float accum=0.0f;
        for (int i = -kSize; i <=kSize ; ++i)
        {
            int idx =max(0,min(fact_x+i,kWidth-1));//一维高斯模板在输入图像x方向上的对应坐标，以选中点为中心左右各ks个
            accum +=p_in[max_x+idx*kChannels+channels]* data[abs(i)];
        }
        p_out[out_idx]=accum / kWeight;
    }
}

/*调用示例
 * dim3 block(x,y,1);
 * dim3 grid((kWidth*c-1+x*2)/(x*2),(h-1+y)/y,1);
 * kernel_gaussBlur_x1<<<grid,block,(ks+1)* sizeof(float)>>>(d_tmp,d_in,d_blur,kWidth,h,c,ks,weight);
 */
__global__ void kernelGaussBlurX1(float *const p_out,float const *const p_in,float const * const p_blur,int const kWidth,int const kHeight,int const kChannels,int const kSize,float const kWeight)
{
    extern __shared__ float data[];
    int x=threadIdx.x+blockIdx.x*blockDim.x*2;
    int y=threadIdx.y+blockIdx.y*blockDim.y;
    int share_idx=threadIdx.y*blockDim.x+threadIdx.x;
    if((share_idx)<(kSize+1))
    {
        data[share_idx]=p_blur[share_idx];
    }
    __syncthreads();
    int fact_x=x/kChannels;
    int channels=x%kChannels;
    int max_x=y*kWidth*kChannels;
    int out_idx=max_x+x;
    if(fact_x<kWidth&&y<kHeight)
    {
        float accum=0.0f;
        for (int i = -kSize; i <=kSize ; ++i)
        {
            int idx =max(0,min(fact_x+i,kWidth-1));//一维高斯模板在输入图像x方向上的对应坐标，以选中点为中心左右各ks个
            accum +=p_in[max_x+idx*kChannels+channels]* data[abs(i)];
            //if(out_idx==10)printf("%f\t%f\n",accum,in[max_x+idx*c+channels]* data[abs(i)]);
        }
        p_out[out_idx]=accum / kWeight;
    }
    //二次展开
    int fact_x1=(x+blockDim.x)/kChannels;
    int channels1=(x+blockDim.x)%kChannels;
    int out_idx1=max_x+x+blockDim.x;
    if(fact_x1<kWidth&&y<kHeight)
    {
        float accum=0.0f;
        for (int i = -kSize; i <=kSize ; ++i)
        {
            int idx =max(0,min(fact_x1+i,kWidth-1));//一维高斯模板在输入图像x方向上的对应坐标，以选中点为中心左右各ks个
            accum +=p_in[max_x+idx*kChannels+channels1]* data[abs(i)];
        }
        p_out[out_idx1]=accum / kWeight;
    }
}

/* 调用示例
 * dim3 block(x,y,1);
 * dim3 grid((kWidth*c-1+x*3)/(x*3),(h-1+y)/y,1);
 * kernel_gaussBlur_x2<<<grid,block,(ks+1)* sizeof(float)>>>(d_tmp,d_in,d_blur,kWidth,h,c,ks,weight);
 */
__global__ void kernelGaussBlurX2(float *const p_out,float const *const p_in,float const * const p_blur,int const kWidth,int const kHeight,int const kChannels,int const kSize,float const kWeight)
{
    extern __shared__ float data[];
    int x=threadIdx.x+blockIdx.x*blockDim.x*3;
    int y=threadIdx.y+blockIdx.y*blockDim.y;
    int share_idx=threadIdx.y*blockDim.x+threadIdx.x;
    if((share_idx)<(kSize+1))
    {
        data[share_idx]=p_blur[share_idx];
    }
    __syncthreads();
    int fact_x=x/kChannels;
    int channels=x%kChannels;
    int max_x=y*kWidth*kChannels;
    int out_idx=max_x+x;
    if(fact_x<kWidth&&y<kHeight)
    {
        float accum=0.0f;
        for (int i = -kSize; i <=kSize ; ++i)
        {
            int idx =max(0,min(fact_x+i,kWidth-1));//一维高斯模板在输入图像x方向上的对应坐标，以选中点为中心左右各ks个
            accum +=p_in[max_x+idx*kChannels+channels]* data[abs(i)];
            //if(out_idx==10)printf("%f\t%f\n",accum,in[max_x+idx*c+channels]* data[abs(i)]);
        }
        p_out[out_idx]=accum / kWeight;
    }
    //二次展开
    int fact_x1=(x+blockDim.x)/kChannels;
    int channels1=(x+blockDim.x)%kChannels;
    int out_idx1=max_x+x+blockDim.x;
    if(fact_x1<kWidth&&y<kHeight)
    {
        float accum=0.0f;
        for (int i = -kSize; i <=kSize ; ++i)
        {
            int idx =max(0,min(fact_x1+i,kWidth-1));//一维高斯模板在输入图像x方向上的对应坐标，以选中点为中心左右各ks个
            accum +=p_in[max_x+idx*kChannels+channels1]* data[abs(i)];
        }
        p_out[out_idx1]=accum / kWeight;
    }
    //三次展开
    int fact_x2=(x+blockDim.x*2)/kChannels;
    int channels2=(x+blockDim.x*2)%kChannels;
    int out_idx2=max_x+x+blockDim.x*2;
    if(fact_x2<kWidth&&y<kHeight)
    {
        float accum=0.0f;
        for (int i = -kSize; i <=kSize ; ++i)
        {
            int idx =max(0,min(fact_x2+i,kWidth-1));//一维高斯模板在输入图像x方向上的对应坐标，以选中点为中心左右各ks个
            accum +=p_in[max_x+idx*kChannels+channels2]* data[abs(i)];
        }
        p_out[out_idx2]=accum / kWeight;
    }
}
/******************************************************************************************/
///功能：y维高斯模糊
/*  函数名                            线程块大小       耗费时间
 *  kernelGaussBlurY	             2.358ms	    [32,4,1]
 *  kernelGaussBlurY1	             1.875ms	    [32,4,1]
 *  kernelGaussBlurY2	             1.811ms	    [32,8,1]**
 */
/******************************************************************************************/
/* 调用示例
* dim3 block(x,y,1);
* dim3 grid((kWidth*c-1+x)/(x),(h-1+y)/y,1);
* kernel_gaussBlur_y<float><<<grid,block,(ks+1)* sizeof(float)>>>(d_tmp,d_in,d_blur,fact_W,h,ks,weight);
*/
template <typename T>
__global__ void kernelGaussBlurY(T *const p_out,T const *const p_in,T const * const p_blur,int const kFact_width,int const kHeight,int const kSize,T const kWeight)
{
    //extern __shared__ float data[];
    sharedMemory<T> smem;
    T* data = smem.p_getPointer();
    int x=threadIdx.x+blockIdx.x*blockDim.x;
    int y=threadIdx.y+blockIdx.y*blockDim.y;
    int share_idx=threadIdx.y*blockDim.x+threadIdx.x;
    if((share_idx)<(kSize+1))
    {
        data[share_idx]=p_blur[share_idx];
    }
    __syncthreads();
    //一次展开
    int out_idx=y*kFact_width+x;
    if(x<kFact_width&&y<kHeight)
    {
        T accum=0.0f;
        for (int i = -kSize; i <=kSize ; ++i)
        {
            int idx =max(0,min(y+i,kHeight-1));//一维高斯模板在输入图像y方向上的对应坐标，以选中点为中心上下各ks个
            accum +=p_in[idx*kFact_width+x]* data[abs(i)];
        }
        p_out[out_idx]=accum / kWeight;
    }
}

/*调用示例
 * dim3 block(x,y,1);
 * dim3 grid((w*c-1+x*2)/(x*2),(h-1+y)/y,1);
 * kernel_gaussBlur_y1<float><<<grid,block,(ks+1)* sizeof(float)>>>(d_tmp,d_in,d_blur,kFact_width,h,ks,weight);
 */
template <typename T>
__global__ void kernelGaussBlurY1(T *const p_out,T const *const p_in,T const * const p_blur,int const kFact_width,int const kHeight,int const kSize,T const kWeight)
{
    //extern __shared__ float data[];
    sharedMemory<T> smem;
    T* data = smem.p_getPointer();
    int x=threadIdx.x+blockIdx.x*blockDim.x*2;
    int y=threadIdx.y+blockIdx.y*blockDim.y;
    int share_idx=threadIdx.y*blockDim.x+threadIdx.x;
    if((share_idx)<(kSize+1))
    {
        data[share_idx]=p_blur[share_idx];
    }
    __syncthreads();
    //一次展开
    int out_idx=y*kFact_width+x;
    if(x<kFact_width&&y<kHeight)
    {
        T accum=0.0f;
        for (int i = -kSize; i <=kSize ; ++i)
        {
            int idx =max(0,min(y+i,kHeight-1));//一维高斯模板在输入图像y方向上的对应坐标，以选中点为中心上下各ks个
            accum +=p_in[idx*kFact_width+x]* data[abs(i)];
        }
        p_out[out_idx]=accum / kWeight;
    }
    //二次展开
    int x1=x+blockDim.x;
    int out_idx1=y*kFact_width+x1;
    if(x1<kFact_width&&y<kHeight)
    {
        T accum=0.0f;
        for (int i = -kSize; i <=kSize ; ++i)
        {
            int idx =max(0,min(y+i,kHeight-1));//一维高斯模板在输入图像y方向上的对应坐标，以选中点为中心上下各ks个
            accum +=p_in[idx*kFact_width+x1]* data[abs(i)];
        }
        p_out[out_idx1]=accum / kWeight;
    }
}

/* 调用示例
 * dim3 block(x,y,1);
 * dim3 grid((w*c-1+x*3)/(x*3),(h-1+y)/y,1);
 * kernel_gaussBlur_y2<float><<<grid,block,(ks+1)* sizeof(float)>>>(d_tmp,d_in,d_blur,kFact_width,h,ks,weight);
 */
template <typename T>
__global__ void kernelGaussBlurY2(T *const p_out,T const *const p_in,T const * const p_blur,int const kFact_width,int const kHeight,int const kSize,T const kWeight)
{
    //extern __shared__ float data[];
    sharedMemory<T> smem;
    T* data = smem.p_getPointer();
    int x=threadIdx.x+blockIdx.x*blockDim.x*3;
    int y=threadIdx.y+blockIdx.y*blockDim.y;
    int share_idx=threadIdx.y*blockDim.x+threadIdx.x;
    if((share_idx)<(kSize+1))
    {
        data[share_idx]=p_blur[share_idx];
    }
    __syncthreads();
    //一次展开
    int out_idx=y*kFact_width+x;
    if(x<kFact_width&&y<kHeight)
    {
        T accum=0.0f;
        for (int i = -kSize; i <=kSize ; ++i)
        {
            int idx =max(0,min(y+i,kHeight-1));//一维高斯模板在输入图像y方向上的对应坐标，以选中点为中心上下各ks个
            accum +=p_in[idx*kFact_width+x]* data[abs(i)];
        }
        p_out[out_idx]=accum / kWeight;
    }
    //二次展开
    int x1=x+blockDim.x;
    int out_idx1=y*kFact_width+x1;
    if(x1<kFact_width&&y<kHeight)
    {
        T accum=0.0f;
        for (int i = -kSize; i <=kSize ; ++i)
        {
            int idx =max(0,min(y+i,kHeight-1));//一维高斯模板在输入图像y方向上的对应坐标，以选中点为中心上下各ks个
            accum +=p_in[idx*kFact_width+x1]* data[abs(i)];
        }
        p_out[out_idx1]=accum / kWeight;
    }
    //三次展开
    int x2=x1+blockDim.x;
    int out_idx2=y*kFact_width+x2;
    if(x2<kFact_width&&y<kHeight)
    {
        T accum=0.0f;
        for (int i = -kSize; i <=kSize ; ++i)
        {
            int idx =max(0,min(y+i,kHeight-1));//一维高斯模板在输入图像y方向上的对应坐标，以选中点为中心上下各ks个
            accum +=p_in[idx*kFact_width+x2]* data[abs(i)];
        }
        p_out[out_idx2]=accum / kWeight;
    }
}

/******************************************************************************************/
///功能：求图像差
/*  函数名                            线程块大小       耗费时间
 *  kernelSubtract	                  1.554ms	    [32,4,1]
 *  kernelSubtract1	                  1.541ms	    [32,8,1]
 *  kernelSubtract2	                  1.537ms	    [32,4,1]
 */
/******************************************************************************************/
/* 调用示例
* dim3 block(x,y,1);
* dim3 grid((w*c-1+x)/(x),(h-1+y)/(y),1);
* kernel_subtract<<<grid,block>>>(d_out,d_in1,d_in2,wc,h);
*/
__global__ void kernelSubtract(float *const p_out,float const * const p_in1,float const * const p_in2,int const kWidth_channels,int const kHeight)
{
    int x=threadIdx.x+blockIdx.x*blockDim.x;
    int y=threadIdx.y+blockIdx.y*blockDim.y;
    int idx=y*kWidth_channels+x;
    float a = 0.0f;
    if(x<kWidth_channels&&y<kHeight) {
        a = p_in1[idx];
        a -= p_in2[idx];
        p_out[idx] = a;
    }
}

/* 调用示例
* dim3 block(x,y,1);
* dim3 grid((w*c-1+x*2)/(x*2),(h-1+y)/(y),1);
* kernel_subtract1<<<grid,block>>>(d_out,d_in1,d_in2,wc,h);
*/
__global__ void kernelSubtract1(float *const p_out,float const * const p_in1,float const * const p_in2,int const kWidth_channels,int const kHeight)
{
    int x=threadIdx.x+blockIdx.x*blockDim.x*2;
    int y=threadIdx.y+blockIdx.y*blockDim.y;
    float diff=0.0f;
    int idx;
    for (int i = 0; i < 2; ++i) {
        idx = y * kWidth_channels + x + blockDim.x * i;
        if (idx <= kHeight * kWidth_channels) {
            diff = p_in1[idx];
            diff -= p_in2[idx];
            p_out[idx] = diff;
        }
    }
}
/* 调用示例
* dim3 block(x,y,1);
* dim3 grid((w*c-1+x*3)/(x*3),(h-1+y)/(y),1);
* kernel_subtract2<<<grid,block>>>(d_out,d_in1,d_in2,wc,h);
*/
__global__ void kernelSubtract2(float *const p_out,float const * const p_in1,float const * const p_in2,int const kWidth_channels,int const kHeight)
{
    int x=threadIdx.x+blockIdx.x*blockDim.x*3;
    int y=threadIdx.y+blockIdx.y*blockDim.y;
    float diff=0.0f;
    int idx;
    for (int i = 0; i < 3; ++i) {
        idx = y * kWidth_channels + x + blockDim.x * i;
        if (idx <= kHeight * kWidth_channels) {
            diff = p_in1[idx];
            diff -=p_in2[idx];
            p_out[idx] = diff;
        }
    }
}

/******************************************************************************************/
///功能：图像差分
/*  函数名                            线程块大小       耗费时间
 *  kernelDifference	             1.601ms	    [32,16,1]
 *  kernelDifference1	             1.538ms	    [32,8,1]
 *  kernelDifference2	             1.534ms	    [32,4,1]**
 */
/******************************************************************************************/
/* 调用示例
* dim3 block(x,y,1);
* dim3 grid((w*c-1+x)/(x),(h-1+y)/(y),1);
* kernel_difference<<<grid,block>>>(d_out,d_in1,d_in2,wc,h);
*/
__global__ void kernelDifference(float *const p_out,float const * const p_in1,float const * const p_in2,int const kWidth_channels,int const kHeight)
{
    int x=threadIdx.x+blockIdx.x*blockDim.x;
    int y=threadIdx.y+blockIdx.y*blockDim.y;
    int idx=y*kWidth_channels+x;
    float diff = 0.0f;
    if(x<kWidth_channels&&y<kHeight) {
        diff = p_in1[idx];
        diff -= p_in2[idx];
        p_out[idx] = fabsf(diff);
    }
}
/* 调用示例
* dim3 block(x,y,1);
* dim3 grid((w*c-1+x*2)/(x*2),(h-1+y)/(y),1);
* kernel_difference1<<<grid,block>>>(d_out,d_in1,d_in2,wc,h);
*/
template <class T>
__global__ void kernelDifference1(T *const p_out,T const * const p_in1,T const * const p_in2,int const kWidth_channels,int const kHeight)
{
    int x=threadIdx.x+blockIdx.x*blockDim.x*2;
    int y=threadIdx.y+blockIdx.y*blockDim.y;
    T diff=0.0f;
    int idx;
    for (int i = 0; i < 2; ++i) {
        idx = y * kWidth_channels + x + blockDim.x * i;
        if (idx <= kHeight * kWidth_channels) {
            diff = p_in1[idx];
            diff -= p_in2[idx];
            p_out[idx] = fabsf(diff);
        }
    }
}
/* 调用示例
* dim3 block(x,y,1);
* dim3 grid((w*c-1+x*3)/(x*3),(h-1+y)/(y),1);
* kernel_difference2<<<grid,block>>>(d_out,d_in1,d_in2,wc,h);
*/
template <class T>
__global__ void kernelDifference2(T *const p_out,T const * const p_in1,T const * const p_in2,int const kWidth_channels,int const kHeight)
{
    int x=threadIdx.x+blockIdx.x*blockDim.x*3;
    int y=threadIdx.y+blockIdx.y*blockDim.y;
    T diff=0.0f;
    int idx;
    for (int i = 0; i < 3; ++i) {
        idx = y * kWidth_channels + x + blockDim.x * i;
        if (idx <= kHeight * kWidth_channels) {
            diff = p_in1[idx];
            diff -= p_in2[idx];
            p_out[idx] = fabsf(diff);
        }
    }
}
/******************************************************************************************/
///调用核函数实现加速功能
/******************************************************************************************/
void desaturateByCuda(float * const p_out_image,float const  *p_in_image,const int kPixel_amount, const int kType,const bool kAlpha)
{
    float *p_d_in=NULL;
    float *p_d_out=NULL;

    const size_t kBytes_in=kPixel_amount*(3+kAlpha)*sizeof(float);
    const size_t kBytes_out=kPixel_amount*(1+kAlpha)* sizeof(float);
    const int  kBlocksize=256;
    dim3 block(kBlocksize,1,1);
    dim3 grid((kPixel_amount-1+kBlocksize*2)/(kBlocksize*2),1,1);
    cudaMalloc(&p_d_in ,kBytes_in);
    cudaMalloc(&p_d_out,kBytes_out);
    cudaMemcpy(p_d_in,p_in_image,kBytes_in,cudaMemcpyHostToDevice);
    if(kAlpha)
    {
        kernelDesaturateAlpha<<<grid,block,kBlocksize*8* sizeof(float)>>>(p_d_out,p_d_in,kPixel_amount,kType);
    }
    else
    {
        kernelDesaturate<<<grid,block,kBlocksize*6* sizeof(float)>>>(p_d_out,p_d_in,kPixel_amount,kType);
    }
    cudaMemcpy(p_out_image,p_d_out,kBytes_out,cudaMemcpyDeviceToHost);
    cudaFree(p_d_in);
    cudaFree(p_d_out);
}

void doubleSizeByCuda(float * const p_out_image,float const  * const p_in_image,int const kWidth,int const kHeight,int const kChannels)
{
    int const kOut_width=kWidth<<1;
    int const kOut_height=kHeight<<1;
    int const kSize_in=kWidth*kHeight;
    int const kSize_out=kOut_width*kOut_height;
    size_t const kBytes_in =kSize_in *kChannels* sizeof(float);
    size_t const kBytes_out=kSize_out*kChannels* sizeof(float);

    float *p_d_in=NULL;
    float *p_d_out=NULL;
    cudaMalloc((void**)&p_d_in ,kBytes_in);
    cudaMalloc((void**)&p_d_out,kBytes_out);
    cudaMemcpy(p_d_in,p_in_image,kBytes_in,cudaMemcpyHostToDevice);

    int x=32;
    int y=4;
    dim3 block2 (x,y,1);
    dim3 grid2 ((kOut_width-1+x*3)/(x*3),(kOut_height-1+y)/y,1);
    cudaMalloc((void**)&p_d_out,kBytes_out);
    kernelDoubleSize2<<<grid2,block2>>>(p_d_out,p_d_in,kOut_width,kOut_height,kWidth,kChannels);
    cudaMemcpy(p_out_image,p_d_out,kBytes_out,cudaMemcpyDeviceToHost);
//释放分配的内存
    cudaFree(p_d_in);
    cudaFree(p_d_out);
}

void halfSizeByCuda(float * const p_out_image,float const  * const p_in_image,int const kWidth,int const kHeight,int const kChannels)
{
    int kOut_width=(kWidth+1)>>1;
    int kOut_height=(kHeight+1)>>1;
    int const kSize_in=kWidth*kHeight;
    int const kSize_out=kOut_width*kOut_height;
    size_t const kBytes_in =kSize_in *kChannels* sizeof(float);
    size_t const kBytes_out=kSize_out*kChannels* sizeof(float);

    float *p_d_in=NULL;
    float *p_d_out=NULL;
    cudaMalloc((void**)&p_d_out,kBytes_out);
    cudaMalloc((void**)&p_d_in, kBytes_in);
    cudaMemcpy(p_d_in,p_in_image,kBytes_in,cudaMemcpyHostToDevice);

    int const x=32;
    int const y=8;
    dim3 block (x,y,1);
    dim3 grid ((kOut_width-1+x*2)/(x*2),(kOut_height-1+y)/y,1);
    kernelHalfSize1<<<grid,block>>>(p_d_out,p_d_in,kOut_width,kOut_height,kWidth,kHeight,kChannels);
    cudaMemcpy(p_out_image,p_d_out,kBytes_out,cudaMemcpyDeviceToHost);

    cudaFree(p_d_in);
    cudaFree(p_d_out);
}

void halfSizeGaussianByCuda(float * const p_out_image,float const  * const p_in_image, int const kWidth,int const kHeight,int const kChannels,float sigma2)
{
    int kOut_width=(kWidth+1)>>1;
    int kOut_height=(kHeight+1)>>1;
    int const kSize_in=kWidth*kHeight;
    int const kSize_out=kOut_width*kOut_height;
    //声明+定义输入/输出图像字节数
    size_t const kBytes_in =kSize_in *kChannels* sizeof(float);
    size_t const kBytes_out=kSize_out*kChannels* sizeof(float);
    float h_w[3];
    //声明显存指针
    float *p_d_w=NULL;
    float *p_d_in=NULL;
    float *p_d_out=NULL;

    //定义权值
    h_w[0] = std::exp(-0.5f / (2.0f * sigma2));
    h_w[1] = std::exp(-2.5f / (2.0f * sigma2));
    h_w[2] = std::exp(-4.5f / (2.0f * sigma2));

    //分配显存
    cudaMalloc((void**)&p_d_w,3* sizeof(float));
    cudaMalloc((void**)&p_d_in ,kBytes_in);
    cudaMalloc((void**)&p_d_out,kBytes_out);
    //传递输入图像和权值
    cudaMemcpy(p_d_in,p_in_image,kBytes_in,cudaMemcpyHostToDevice);
    cudaMemcpy(p_d_w,h_w,3* sizeof(float),cudaMemcpyHostToDevice);

    int x=32;
    int y=4;
    //定义grid和block大小
    dim3 block(x, y, 1);
    dim3 grid((kOut_width - 1 + x) / (x), (kOut_height - 1 + y) / y, 1);
    kernelHalfSizeGauss1<<< grid, block >>> (p_d_out, p_d_in, kOut_width, kOut_height, kWidth, kHeight, kChannels, p_d_w);
    //传出输入图像
    cudaMemcpy(p_out_image, p_d_out, kBytes_out, cudaMemcpyDeviceToHost);
    //释放分配的显存
    cudaFree(p_d_w);
    cudaFree(p_d_in);
    cudaFree(p_d_out);
}

int blurGaussianByCuda(float * const p_out_image,float const  * const p_in_image, int const kWidth,int const kHeight,int const kChannels,float sigma)
{
    //声明+定义输入输出图片大小及字节数
    int const kFact_width=kWidth*kChannels;
    int const kSize_image=kWidth*kHeight;
    size_t const kBytes=kSize_image*kChannels* sizeof(float);

    int const kSize = std::ceil(sigma * 2.884f);//一维高斯核长度为ks*2+1
    std::vector<float> v_kernel(kSize + 1);//分配半个高斯核
    float weight = 0;
    for (int i = 0; i < kSize + 1; ++i)
    {
        v_kernel[i] = math::func::gaussian((float)i, sigma);//kernel[0]=1,kernel[i]=wi;
        weight += v_kernel[i]*2;
    }
    weight-=v_kernel[0];
    int const kBytes_blur=(kSize+1)*sizeof(float);

    //声明显存指针
    float *p_d_in=NULL;
    float *p_d_out=NULL;
    float *p_d_tmp=NULL;
    float *p_d_blur=NULL;

    //分配显存
    cudaMalloc((void**)&p_d_in  ,kBytes);
    cudaMalloc((void**)&p_d_tmp ,kBytes);
    cudaMalloc((void**)&p_d_out ,kBytes);
    cudaMalloc((void**)&p_d_blur,kBytes_blur);

    //数据从cpu传入gpu
    cudaMemcpy(p_d_in ,p_in_image,kBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(p_d_blur,&v_kernel[0],kBytes_blur,cudaMemcpyHostToDevice);

    int x=32;
    int y=4;
    dim3 block(x,y,1);
    dim3 grid((kFact_width-1+x*2)/(x*2),(kHeight-1+y)/y,1);

    //x维高斯模糊
    kernelGaussBlurX1<<<grid,block,(kSize+1)* sizeof(float)>>>(p_d_tmp,p_d_in,p_d_blur, kWidth, kHeight,kChannels,kSize,weight);
    //y维高斯模糊
    x=32;
    y=8;
    dim3 block1(x,y,1);
    dim3 grid1((kFact_width-1+x*3)/(x*3),(kHeight-1+y)/y,1);
    kernelGaussBlurY2<float><<<grid1,block1,(kSize+1)* sizeof(float)>>>(p_d_out,p_d_tmp,p_d_blur,kFact_width,kHeight,kSize,weight);

    //数据从gpu传回cpu
    cudaMemcpy(p_out_image,p_d_out,kBytes,cudaMemcpyDeviceToHost);
    //释放显存
    cudaFree(p_d_in);
    cudaFree(p_d_tmp);
    cudaFree(p_d_out);
    cudaFree(p_d_blur);
    return  0;
}

int blurGaussian2ByCuda(float * const p_out_image,float const  * const p_in_image, int const kWidth,int const kHeight,int const kChannels,float sigma2)
{
    float sigma = sqrt(sigma2);
    blurGaussianByCuda(p_out_image,p_in_image,kWidth,kHeight,kChannels,sigma);
    return 0;
}

int subtractByCuda(float * const p_out_image,float const  * const p_in_image1,float const  * const p_in_image2,int const kWidth,int const kHeight,int const kChannels)
{
    int const kSize=kWidth*kHeight;
    size_t const kBytes=kSize*kChannels*sizeof(float);
//定义显存指针
    float *p_d_in1=NULL;
    float *p_d_in2=NULL;
    float *p_d_out=NULL;
//分配显存指针
    cudaMalloc((void**)&p_d_in1,kBytes);
    cudaMalloc((void**)&p_d_in2,kBytes);
    cudaMalloc((void**)&p_d_out,kBytes);
//传递数据(cpu2gpu)
    cudaMemcpy(p_d_in1,p_in_image1,kBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(p_d_in2,p_in_image2,kBytes,cudaMemcpyHostToDevice);
    int x=32;
    int y=4;
    dim3 block(x,y,1);
    dim3 grid((kWidth*kChannels-1+x*3)/(x*3),(kHeight-1+y)/(y),1);
    kernelSubtract2<<<grid,block>>>(p_d_out,p_d_in1,p_d_in2,kWidth*kChannels,kHeight);
//传递数据(gpu2cpu)
    cudaMemcpy(p_out_image,p_d_out,kBytes,cudaMemcpyDeviceToHost);
//释放显存
    cudaFree(p_d_in1);
    cudaFree(p_d_in2);
    cudaFree(p_d_out);
    return 0;
}

template <class T>
int differenceByCu(T * const p_out_image,T const  * const p_in_image1,T const  * const p_in_image2,int const kWidth,int const kHeight,int const kChannels)
{
    int const kSize=kWidth*kHeight;
    size_t const kBytes=kSize*kChannels*sizeof(T);
//定义显存指针
    T *p_d_in1=NULL;
    T *p_d_in2=NULL;
    T *p_d_out=NULL;
//分配显存指针
    cudaMalloc((void**)&p_d_in1,kBytes);
    cudaMalloc((void**)&p_d_in2,kBytes);
    cudaMalloc((void**)&p_d_out,kBytes);
//传递数据(cpu2gpu)
    cudaMemcpy(p_d_in1,p_in_image1,kBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(p_d_in2,p_in_image2,kBytes,cudaMemcpyHostToDevice);
    int x=32;
    int y=4;
    dim3 block(x,y,1);
    dim3 grid((kWidth*kChannels-1+x*3)/(x*3),(kHeight-1+y)/(y),1);
    kernelDifference2<T><<<grid,block>>>(p_d_out,p_d_in1,p_d_in2,kWidth*kChannels,kHeight);
//传递数据(gpu2cpu)
    cudaMemcpy(p_out_image,p_d_out,kBytes,cudaMemcpyDeviceToHost);
//释放显存
    cudaFree(p_d_in1);
    cudaFree(p_d_in2);
    cudaFree(p_d_out);
    return 0;
}

template <typename T>
int differenceByCuda(T * const p_out_image,T const  * const p_in_image1,T const  * const p_in_image2,int const kWidth,int const kHeight,int const kChannels)
{
    return 0;
}
template<>
int differenceByCuda<float>(float * const p_out_image,float const  * const p_in_image1,float const  * const p_in_image2,int const kWidth,int const kHeight,int const kChannels)
{
    differenceByCu<float>(p_out_image,p_in_image1,p_in_image2,kWidth,kHeight,kChannels);
    return 0;
}
template<>
int differenceByCuda<char>(char * const p_out_image,char const  * const p_in_image1,char const  * const p_in_image2,int const kWidth,int const kHeight,int const kChannels)
{
    differenceByCu<char>(p_out_image,p_in_image1,p_in_image2,kWidth,kHeight,kChannels);
    return 0;
}

