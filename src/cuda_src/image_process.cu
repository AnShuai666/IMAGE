/**
 * @desc    图像处理函数加速
 * @author  杨丰拓
 * @date    2019-04-4
 * @email   yangfengtuo@163.com
*/
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

__global__ void warmup(void)
{}
/*
__global__ void kernel_desaturate_alpha(float *out,float const *in, const int size,const int type)
{
    extern __shared__   float s[];
    int in_idx = threadIdx.x  + blockIdx.x * blockDim.x * 8 ;
    int out_idx = threadIdx.x+ blockIdx.x * blockDim.x * 4 ;
    int tid=threadIdx.x;
    int stride=tid*4;
    int stride1=stride+blockDim.x*4;
    if (in_idx< size * 4)
    {
        s[tid]=in[in_idx];
        s[tid+blockDim.x]=in[in_idx+blockDim.x];
        s[tid+blockDim.x*2]=in[in_idx+blockDim.x*2];
        s[tid+blockDim.x*3]=in[in_idx+blockDim.x*3];
        s[tid+blockDim.x*4]=in[in_idx+blockDim.x*4];
        s[tid+blockDim.x*5]=in[in_idx+blockDim.x*5];
        s[tid+blockDim.x*6]=in[in_idx+blockDim.x*6];
        s[tid+blockDim.x*7]=in[in_idx+blockDim.x*7];
    }
    __syncthreads();

    if(type==0)
    {
        out[out_idx]=max(s[stride+0],max(s[stride+1],s[stride+2]));
        out[out_idx+blockDim.x*2]=max(s[stride1+0],max(s[stride1+1],s[stride1+2]));
    }
    if(type==1)
    {
        float const max_v = max(s[stride+0],max(s[stride+1],s[stride+2]));
        float const min_v = min(s[stride+0],min(s[stride+1],s[stride+2]));
        out[out_idx]=0.5f*(max_v+min_v);
        float const max_s = max(s[stride1+0],max(s[stride1+1],s[stride1+2]));
        float const min_s = min(s[stride1+0],min(s[stride1+1],s[stride1+2]));
        out[out_idx+blockDim.x*2]=0.5f*(max_s+min_s);
    }
    if(type==2)
    {
        out[out_idx]=0.21f * s[stride+0] + 0.72f * s[stride+1] + 0.07f * s[stride+2];
        out[out_idx+blockDim.x*2]=0.21f * s[stride1+0] + 0.72f * s[stride1+1] + 0.07f * s[stride1+2];
    }
    if(type==3)
    {
        out[out_idx]=0.30f * s[stride+0] + 0.59f * s[stride+1] + 0.11f * s[stride+2];
        out[out_idx+blockDim.x*2]=0.30f * s[stride1+0] + 0.59f * s[stride1+1] + 0.11f * s[stride1+2];
    }
    if(type==4)
    {
        out[out_idx]=((float)(s[stride+0] + s[stride+1] + s[stride+2])) / 3.0f;
        out[out_idx+blockDim.x*2]=((float)(s[stride1+0] + s[stride1+1] + s[stride1+2])) / 3.0f;
    }
    out[out_idx+tid+1]=s[stride+3];
    out[out_idx+blockDim.x*2+tid+1]=s[stride1+3];
}
__global__ void kernel_desaturate(float *out,float const *in, const int size,const int type)
{
    extern __shared__   float s[];
    int in_idx = threadIdx.x  + blockIdx.x * blockDim.x * 6 ;
    int out_idx = threadIdx.x+ blockIdx.x * blockDim.x * 2 ;
    int tid=threadIdx.x;
    int stride=tid*3;
    int stride1=stride+blockDim.x*3;

    if (in_idx< size * 3)
    {
        s[tid]=in[in_idx];
        s[tid+blockDim.x]=in[in_idx+blockDim.x];
        s[tid+blockDim.x*2]=in[in_idx+blockDim.x*2];
        s[tid+blockDim.x*3]=in[in_idx+blockDim.x*3];
        s[tid+blockDim.x*4]=in[in_idx+blockDim.x*4];
        s[tid+blockDim.x*5]=in[in_idx+blockDim.x*5];
    }
    __syncthreads();
    if(type==0)
    {
        out[out_idx]=max(s[stride+0],max(s[stride+1],s[stride+2]));
        out[out_idx+blockDim.x]=max(s[stride1+0],max(s[stride1+1],s[stride1+2]));
    }
    if(type==1)
    {
        float const max_v = max(s[stride+0],max(s[stride+1],s[stride+2]));
        float const min_v = min(s[stride+0],min(s[stride+1],s[stride+2]));
        out[out_idx]=0.5f*(max_v+min_v);
        float const max_s = max(s[stride1+0],max(s[stride1+1],s[stride1+2]));
        float const min_s = min(s[stride1+0],min(s[stride1+1],s[stride1+2]));
        out[out_idx+blockDim.x]=0.5f*(max_s+min_s);
    }
    if(type==2)
    {
        out[out_idx]=0.21f * s[stride+0] + 0.72f * s[stride+1] + 0.07f * s[stride+2];
        out[out_idx+blockDim.x]=0.21f * s[stride1+0] + 0.72f * s[stride1+1] + 0.07f * s[stride1+2];
    }
    if(type==3)
    {
        out[out_idx]=0.30f * s[stride+0] + 0.59f * s[stride+1] + 0.11f * s[stride+2];
        out[out_idx+blockDim.x]=0.30f * s[stride1+0] + 0.59f * s[stride1+1] + 0.11f * s[stride1+2];
    }
    if(type==4)
    {
        out[out_idx]=((float)(s[stride+0] + s[stride+1] + s[stride+2])) / 3.0f;
        out[out_idx+blockDim.x]=((float)(s[stride1+0] + s[stride1+1] + s[stride1+2])) / 3.0f;
    }

}


void desaturate_by_cuda(float  * const out_image,float const *in_image,const int pixel_amount, const int type,const bool alpha)
{
    float *d_in=NULL;
    float *d_out=NULL;

    int bytes_in=pixel_amount*(3+alpha)*sizeof(float);
    int bytes_out=pixel_amount*(1+alpha)* sizeof(float);
    const int  blocksize=256;
    dim3 block(blocksize,1,1);
    dim3 grid((pixel_amount-1+blocksize*2)/(blocksize*2),1,1);
    cudaMalloc(&d_in,bytes_in);
    cudaMalloc(&d_out,bytes_out);
    cudaMemcpy(d_in,in_image,bytes_in,cudaMemcpyHostToDevice);
    if(alpha)
    {
        kernel_desaturate_alpha<<<grid,block,blocksize*4* sizeof(float)>>>(d_out,d_in,pixel_amount,type);
    }
    else
    {
        kernel_desaturate<<<grid,block,blocksize*6* sizeof(float)>>>(d_out,d_in,pixel_amount,type);
    }
    cudaMemcpy(out_image,d_out,bytes_out,cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}
*/


/******************************************************************************************/
///功能：图片放大两倍
/*  函数名                         线程块大小       耗费时间
 *  kernel_doublesize             [32,16,1]      3.719ms(最快)
 *  kernel_doublesize_dim3        [32,8,3]       5.76ms
 *  kernel_doublesizebyshare      [32,32,1]      4.498ms
 */
/******************************************************************************************/
__global__ void kernel_doublesize(float *out,float *in,int const image_x,int const image_y,int const iw,int const ic)
{
    int out_x = threadIdx.x + blockIdx.x * blockDim.x * ic*2;
    int out_y = threadIdx.y + blockIdx.y * blockDim.y;
    for (int c = 0; c <ic*2 ; ++c)
    {
        int fact_x=out_x+blockDim.x*c;
        if(out_y<image_y&&fact_x<image_x*ic)
        {
            int idx=fact_x+out_y*image_x*ic;
            bool nexty=(out_y+1)<image_y;
            bool nextx=(fact_x+ic)<(image_x*ic);
            int yoff[2]={ic*iw*(out_y>>1),
                         ic*iw*((out_y+nexty)>>1)};
            int xoff[2]={((fact_x/ic)>>1)*ic+fact_x%ic,
                         (((fact_x/ic)+nextx)>>1)*ic+fact_x%ic};
            int index[4]={yoff[0]+xoff[0],
                          yoff[0]+xoff[1],
                          yoff[1]+xoff[0],
                          yoff[1]+xoff[1]};
            out[idx]=0.25f*(in[index[0]]+in[index[1]]+in[index[2]]+in[index[3]]);
        }
    }
}
__global__ void kernel_doublesize_dim3(float *out,float *in,int const image_x,int const image_y,int const iw)
{
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*2;
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;
    int out_z=threadIdx.z;

    if(out_x<image_x&&out_y<image_y)
    {
        int idx=out_y*image_x*blockDim.z+out_x*blockDim.z+out_z;

        const bool nexty=(out_y+1)<image_y;
        const bool nextx=(out_x+1)<image_x;
        int yoff[2]={blockDim.z*iw*(out_y>>1),blockDim.z*iw*((out_y+nexty)>>1)};
        int xoff[2]={blockDim.z*(out_x>>1),blockDim.z*((out_x+nextx)>>1)};
        int index[4]={yoff[0]+xoff[0]+out_z,
                      yoff[0]+xoff[1]+out_z,
                      yoff[1]+xoff[0]+out_z,
                      yoff[1]+xoff[1]+out_z};
        out[idx]=0.25f*(in[index[0]]+in[index[1]]+in[index[2]]+in[index[3]]);

        int idx_2=out_y*image_x*blockDim.z+(out_x+blockDim.x)*blockDim.z+out_z;
        const bool nextx_2=(out_x+blockDim.x+1)<image_x;
        int xoff_2[2]={blockDim.z*((out_x+blockDim.x)>>1),blockDim.z*((out_x+blockDim.x+nextx_2)>>1)};
        int index_2[4]={yoff[0]+xoff_2[0]+out_z,
                      yoff[0]+xoff_2[1]+out_z,
                      yoff[1]+xoff_2[0]+out_z,
                      yoff[1]+xoff_2[1]+out_z};
        out[idx_2]=0.25f*(in[index_2[0]]+in[index_2[1]]+in[index_2[2]]+in[index_2[3]]);


    }

}

__global__ void kernel_doublesizebyshare(float *out,float *in,int const ow,int const oh,int const iw,int const ih,int const ic)
{
    extern __shared__ float  data[];
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*ic*3;
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;

    int share_x=(blockDim.x>>1)+1;//共享内存块x维（需乘ic）
    int share_y=(blockDim.y>>1)+1;//共享内存块y维
    int share_fact_x=share_x*3*ic;
    int share_idx_x;
    int share_idx_y= threadIdx.y;//共享内存块内y维索引
    int in_x0 = ((blockIdx.x * blockDim.x*3) >> 1) * ic;
    int in_y0 = (blockIdx.y * blockDim.y) >> 1;
    int x,y,c,fact_x;

    for ( c = 0; c <ic*3 ; ++c)
    {
        share_idx_x = threadIdx.x + blockDim.x * c;//共享内存块内x索引
        if (share_idx_x < share_fact_x && share_idx_y < share_y)
        {
            x = min(in_x0 + share_idx_x, iw * ic - ic + share_idx_x % ic);
            y = min(in_y0 + share_idx_y, ih - 1);
            data[share_idx_y * share_fact_x + share_idx_x] = in[y * iw * ic + x];
        }

    }
    __syncthreads();
    for ( c = 0; c <ic*3 ; ++c)
    {
        fact_x=out_x+blockDim.x*c;
        if(out_y<oh&&fact_x<ow*ic)
        {
            share_idx_x = threadIdx.x + blockDim.x * c;
            int yoff[2]={(share_idx_y>>1)*share_fact_x,((share_idx_y+1)>>1)*share_fact_x};
            int xoff[2]={(share_idx_x/ic>>1)*ic+share_idx_x%ic,
                         ((share_idx_x/ic+1)>>1)*ic+share_idx_x%ic};
            int out_idx=out_y*ow*ic+fact_x;
            int index[4]={yoff[0]+xoff[0],
                          yoff[0]+xoff[1],
                          yoff[1]+xoff[0],
                          yoff[1]+xoff[1]};
            out[out_idx]=0.25f*(data[index[0]]+data[index[1]]+data[index[2]]+data[index[3]]);
        }
    }
    /*
    extern __shared__ float  data[];
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*ic;
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;

    int share_x=(blockDim.x>>1)+1;//共享内存块x维（需乘ic）
    int share_y=(blockDim.y>>1)+1;//共享内存块y维
    int share_idx_x;
    int share_idx_y;

    for (int c = 0; c <ic ; ++c) {
        share_idx_x = threadIdx.x + blockDim.x * c;//共享内存块内x索引
        share_idx_y = threadIdx.y;//共享内存块内y维索引

        int in_x0 = ((blockIdx.x * blockDim.x) >> 1) * ic;
        int in_y0 = (blockIdx.y * blockDim.y) >> 1;

        if (share_idx_x < (share_x * ic) && share_idx_y < share_y)
        {
            int x = min(in_x0 + share_idx_x, iw * ic - ic + share_idx_x % ic);
            int y = min(in_y0 + share_idx_y, ih - 1);
            data[share_idx_y * share_x * ic + share_idx_x] = in[y * iw * ic + x];
        }

    }
    __syncthreads();
    for (int c = 0; c <ic ; ++c) {
        int fact_x=out_x+blockDim.x*c;
        if(out_y<oh&&fact_x<ow*ic)
        {
            share_idx_x = threadIdx.x + blockDim.x * c;
            share_idx_y = threadIdx.y;

            int yoff[2]={(share_idx_y>>1)*share_x*ic,((share_idx_y+1)>>1)*share_x*ic};
            int xoff[2]={(share_idx_x/ic>>1)*ic+share_idx_x%ic,
                         ((share_idx_x/ic+1)>>1)*ic+share_idx_x%ic};
            float val[4]={data[yoff[0]+xoff[0]],
                          data[yoff[0]+xoff[1]],
                          data[yoff[1]+xoff[0]],
                          data[yoff[1]+xoff[1]]};
            int out_idx=out_y*ow*ic+fact_x;
            out[out_idx]=0.25f*(val[0]+val[1]+val[2]+val[3]);
        }
    }*/

}


/******************************************************************************************/
///功能：图片缩小两倍
/*  函数名                         线程块大小       耗费时间
 *  kernel_halfsize1              [32,4,1]      639.142us
 *  kernel_halfsizebyshare1       [32,4,1]      654.107us
 *  kernel_halfsize               [32,8,1]      639.56us
 *  kernel_halfsizebyshare        [32,4,1]      687.768us
 */
/******************************************************************************************/
__global__ void kernel_halfsizebyshare1(float *out,float *in,int const ow,int const oh,int const iw,int const ih,int const ic)
{
    extern __shared__ float data[];
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*ic*2;//输出的x维起始索引
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;//输出的y位索引
    int stride=iw*ic;//输入图像的行索引的最大值

    int in_x0=blockIdx.x*blockDim.x*2*ic*2;//输入图像x维的第一起始点
    int in_y0=blockIdx.y*blockDim.y*2;//输入图像y维的第一起始点
    int in_x1=in_x0+blockDim.x*ic*2;//输入图像x维的第二起始点
    int in_y1=in_y0+blockDim.y;//输入图像y维的第二起始点

    int share_x=blockDim.x*4;//共享块内x维最大像素点个数
    for (int c = 0; c < ic*2; ++c)
    {
        int fact_x_s=threadIdx.x+blockDim.x*c;//共享内存内第一个x的索引
        int x_s=fact_x_s+blockDim.x*ic*2;//共享内存内第二个x的索引
        int y_s0=threadIdx.y*share_x*ic;//共享内存内第一个y的索引
        int y_s1=y_s0+blockDim.y*share_x*ic;//共享内存内第二个y的索引
        int fact_iw=fact_x_s%ic+stride-ic;

        int x0=min(in_x0+fact_x_s,fact_iw);
        int x1=min(in_x1+fact_x_s,fact_iw);
        int y0=min(in_y0+threadIdx.y,ih-1)*stride;
        int y1=min(in_y1+threadIdx.y,ih-1)*stride;
        data[y_s0+fact_x_s]=in[y0+x0];
        data[y_s0+x_s]=in[y0+x1];
        data[y_s1+fact_x_s]=in[y1+x0];
        data[y_s1+x_s]=in[y1+x1];
    }
    __syncthreads();
    for (int c = 0; c <ic*2 ; ++c) {
        int fact_x=out_x+blockDim.x*c;

        if(out_y<oh&&fact_x<ow*ic)
        {
            int fact_x_s=threadIdx.x+blockDim.x*c;
            int srow1=threadIdx.y*2*share_x*ic;
            int srow2=srow1+share_x*ic;
            int scol1=(fact_x_s / ic) * 2 * ic + fact_x_s % ic;
            int scol2=scol1 + ic;
            int index[4] = {srow1 + scol1,
                            srow1 + scol2,
                            srow2 + scol1,
                            srow2 + scol2};
            int out_idx = out_y * ow*ic + fact_x;
            out[out_idx] = 0.25f * (data[index[0]] + data[index[1]] + data[index[2]] + data[index[3]]);
        }
    }
}
__global__ void kernel_halfsize(float *out,float *in,int const ow,int const oh,int const iw,int const ih,int const ic)
{
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*ic;
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;
    int stride=iw*ic;

    for(int c=0;c<ic;c++)
    {
        int fact_x=out_x+blockDim.x*c;
        if(out_y<oh&&fact_x<ow*ic) {
            int irow1 = out_y * 2 * stride;
            int irow2 = irow1 + stride * (out_y * 2 + 1 < ih);
            int icol1 = (fact_x / ic) * 2 * ic + fact_x % ic;
            int icol2 = min((icol1 + ic), (iw * ic - ic + fact_x % ic));
            int index[4] = {irow1 + icol1,
                            irow1 + icol2,
                            irow2 + icol1,
                            irow2 + icol2};
            int out_idx = out_y * ow*ic + fact_x;
            out[out_idx] = 0.25f * (in[index[0]] + in[index[1]] + in[index[2]] + in[index[3]]);
        }

    }
}
__global__ void kernel_halfsize1(float *out,float *in,int const ow,int const oh,int const iw,int const ih,int const ic)
{
    //若需要展开ic*3重循环只需修改out_x=threadIdx.x+blockIdx.x*blockDim.x*ic*3;以及for(int c=0;c<ic*3;c++)即可，同时应修改网格大小
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*ic*2;
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;
    int stride=iw*ic;

    for(int c=0;c<ic*2;c++)
    {
        int fact_x=out_x+blockDim.x*c;
        if(out_y<oh&&fact_x<ow*ic) {
            int irow1 = out_y * 2 * stride;
            int irow2 = irow1 + stride * (out_y * 2 + 1 < ih);
            int icol1 = (fact_x / ic) * 2 * ic + fact_x % ic;
            int icol2 = min((icol1 + ic), (iw * ic - ic + fact_x % ic));
            int index[4] = {irow1 + icol1,
                            irow1 + icol2,
                            irow2 + icol1,
                            irow2 + icol2};
            int out_idx = out_y * ow*ic + fact_x;
            out[out_idx] = 0.25f * (in[index[0]] + in[index[1]] + in[index[2]] + in[index[3]]);
        }

    }
}
__global__ void kernel_halfsizebyshare(float *out,float *in,int const ow,int const oh,int const iw,int const ih,int const ic)
{
    extern __shared__ float data[];
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*ic;//输出的x维起始索引
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;//输出的y位索引
    int stride=iw*ic;//输入图像的行索引的最大值

    int in_x0=blockIdx.x*blockDim.x*2*ic;//输入图像x维的起始点
    int in_y0=blockIdx.y*blockDim.y*2;//输入图像y维的起始点
    int in_x1=in_x0+blockDim.x*ic;
    int in_y1=in_y0+blockDim.y;

    int share_x=blockDim.x*2;//共享块内x维最大像素点个数
    for (int c = 0; c < ic; ++c)
    {
        int fact_x_s=threadIdx.x+blockDim.x*c;
        int x_s=fact_x_s+blockDim.x*ic;
        int y_s0=threadIdx.y*share_x*ic;
        int y_s1=y_s0+blockDim.y*share_x*ic;
        int fact_iw=fact_x_s%ic+stride-ic;
        int x0=min(in_x0+fact_x_s,fact_iw);
        int x1=min(in_x1+fact_x_s,fact_iw);
        int y0=min(in_y0+threadIdx.y,ih-1)*stride;
        int y1=min(in_y1+threadIdx.y,ih-1)*stride;
        data[y_s0+fact_x_s]=in[y0+x0];
        data[y_s0+x_s]=in[y0+x1];
        data[y_s1+fact_x_s]=in[y1+x0];
        data[y_s1+x_s]=in[y1+x1];
    }
    __syncthreads();
    for (int c = 0; c <ic ; ++c)
    {
        int fact_x=out_x+blockDim.x*c;

        if(out_y<oh&&fact_x<ow*ic)
        {
            int fact_x_s=threadIdx.x+blockDim.x*c;
            int srow1=threadIdx.y*2*share_x*ic;
            int srow2=srow1+share_x*ic;
            int scol1=(fact_x_s / ic) * 2 * ic + fact_x_s % ic;
            int scol2=scol1 + ic;
            int index[4] = {srow1 + scol1,
                            srow1 + scol2,
                            srow2 + scol1,
                            srow2 + scol2};
            int out_idx = out_y * ow*ic + fact_x;
            out[out_idx] = 0.25f * (data[index[0]] + data[index[1]] + data[index[2]] + data[index[3]]);
        }
    }
}


/******************************************************************************************/
///调用核函数实现加速功能
/******************************************************************************************/
void warm(void)
{
    warmup<<<1,1>>>();
}

void double_size_by_cuda(float * const out_image,float const  * const in_image, int const weight,int const height,int const channels)
{
    int const ow=weight<<1;
    int const oh=height<<1;
    int const size_in=weight*height;
    int const size_out=ow*oh;
    int const bytes_in=size_in*channels* sizeof(float);
    int const bytes_out=size_out*channels* sizeof(float);

    float *d_in=NULL;
    float *d_out=NULL;
    cudaMalloc((void**)&d_in,bytes_in);
    cudaMalloc((void**)&d_out,bytes_out);

    int const  x=32;
    int const  y=16;
    //int const  share_x=((x>>1)+1);
    //int const  share_y=(y>>1)+1;
    dim3 block (x,y,1);
    dim3 grid ((ow-1+x*2)/(x*2),(oh-1+y)/y,1);

    cudaMemcpy(d_in,in_image,bytes_in,cudaMemcpyHostToDevice);
    //kernel_doublesizebyshare<<<grid,block,share_x*share_y*3*channels*sizeof(float)>>>(d_out,d_in,ow,oh,weight,height,channels);
    kernel_doublesize<<<grid,block>>>(d_out,d_in,ow,oh,weight,channels);
    cudaMemcpy(out_image,d_out,bytes_out,cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

}

void halfsize_by_cuda(float * const out_image,float const  * const in_image, int const weight,int const height,int const channels)
{
    int ow=(weight+1)>>1;
    int oh=(height+1)>>1;
    int const size_in=weight*height;
    int const size_out=ow*oh;
    int const bytes_in=size_in*channels* sizeof(float);
    int const bytes_out=size_out*channels* sizeof(float);

    float *d_in=NULL;
    float *d_out=NULL;
    cudaMalloc((void**)&d_in,bytes_in);
    cudaMalloc((void**)&d_out,bytes_out);

    int const  x=32;
    int const  y=4;
    int const   share_x=x*4;
    int const   share_y=y*2;
    dim3 block (x,y,1);
    dim3 grid ((ow-1+x*2)/(x*2),(oh-1+y)/y,1);
    cudaMemcpy(d_in,in_image,bytes_in,cudaMemcpyHostToDevice);
    kernel_halfsizebyshare1<<<grid,block,share_x*share_y*channels* sizeof(float)>>>(d_out,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image,d_out,bytes_out,cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}

