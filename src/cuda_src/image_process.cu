/**
 * @desc    图像处理函数加速
 * @author  杨丰拓
 * @date    2019-04-16
 * @email   yangfengtuo@163.com
*/
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
void compare(float *const out_image, float const *const out, int const ow, int const oh, int const ic) {
    bool success = 1;
    for (int j = 0; j < oh; ++j) {
        for (int i = 0; i < ow; ++i) {
            for (int k = 0; k < ic; ++k) {
                float a = out[j * ow * ic + i * ic + k];
                float b = out_image[j * ow * ic + i * ic + k];
                if (std::abs(a - b) > 0.000001)
                    //if(a!=b)
                {
                    printf("idx:%d\t", j * ow * ic + i * ic + k);
                    printf("cpu:\t%1.18lf\tgpu:\t%1.18lf\n", a, b);
                    success = 0;
                }
            }
        }
    }
    if (success)std::cout << "gpu加速后的计算结果与cpu计算的结果一致!" << std::endl;
}

void compare_split(float const *const in_image, float *out_1, float *out_2, float *out_3, int const weight,
                   int const height) {
    bool success = 1;
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < weight; ++i) {
            float a_0 = in_image[j * weight * 3 + i * 3];
            float a_1 = in_image[j * weight * 3 + i * 3 + 1];
            float a_2 = in_image[j * weight * 3 + i * 3 + 2];

            float b_0 = out_1[j * weight + i];
            float b_1 = out_2[j * weight + i];
            float b_2 = out_3[j * weight + i];
            if (a_0 != b_0) {
                printf("idx:%d\t%f\t%f\n", j * weight + i, a_0, b_0);
                success = 0;
            }
            if (!success) {
                std::cout << "第一通道分离失败" << std::endl;
                exit(1);
            }
            if (a_1 != b_1) {
                printf("idx:%d\t%f\t%f\n", j * weight + i, a_1, b_1);
                success = 0;
            }
            if (!success) {
                std::cout << "第二通道分离失败" << std::endl;
                exit(1);
            }
            if (a_2 != b_2) {
                printf("idx:%d\t%f\t%f\n", j * weight + i, a_2, b_2);
                success = 0;
            }
            if (!success) {
                std::cout << "第三通道分离失败" << std::endl;
                exit(1);
            }
        }
    }
    if (success)std::cout << "分离通道成功" << std::endl;
}

void gpuzero(float *a, float *b, float *c, size_t const bytes) {
    cudaMemset(a, 0, bytes);
    cudaMemset(b, 0, bytes);
    cudaMemset(c, 0, bytes);
}

void cpuzero(float *a, float *b, float *c, size_t const bytes) {
    memset(a, 0, bytes);
    memset(b, 0, bytes);
    memset(c, 0, bytes);
}

void gpu2cpu3(float *h_in1, float *d_in1, float *h_in2, float *d_in2, float *h_in3, float *d_in3,
              size_t const bytes_channels) {
    cudaMemcpy(h_in1, d_in1, bytes_channels, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_in2, d_in2, bytes_channels, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_in3, d_in3, bytes_channels, cudaMemcpyDeviceToHost);
}
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
 *  kernel_doublesize               3.678ms	    [32,4,1]
 *  kernel_doublesize1              3.67ms	    [32,4,1]
 *  kernel_doublesize2              3.532ms	    [32,4,1]**
 *  kernel_doublesizebyshare        5.265ms	    [32,8,1]
 *  kernel_doublesizebyshare1       4.737ms	    [64,8,1]
 *  kernel_doublesizebyshare2       3.98ms	    [32,8,1]
 */
/******************************************************************************************/
__global__ void kernel_doublesize(float *out,float *in,int const image_x,int const image_y,int const iw,int const ic)
{
    int out_x = threadIdx.x + blockIdx.x * blockDim.x * ic;
    int out_y = threadIdx.y + blockIdx.y * blockDim.y;
    for (int c = 0; c <ic ; ++c)
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
__global__ void kernel_doublesize1(float *out,float *in,int const image_x,int const image_y,int const iw,int const ic)
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
__global__ void kernel_doublesize2(float *out,float *in,int const image_x,int const image_y,int const iw,int const ic)
{
    int out_x = threadIdx.x + blockIdx.x * blockDim.x * ic*3;
    int out_y = threadIdx.y + blockIdx.y * blockDim.y;
    for (int c = 0; c <ic*3 ; ++c)
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
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*ic;
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;

    int share_x=(blockDim.x>>1)+1;//共享内存块x维（需乘ic）
    int share_y=(blockDim.y>>1)+1;//共享内存块y维
    int share_fact_x=share_x*ic;
    int share_idx_x;
    int share_idx_y= threadIdx.y;//共享内存块内y维索引
    int in_x0 = ((blockIdx.x * blockDim.x) >> 1) * ic;
    int in_y0 = (blockIdx.y * blockDim.y) >> 1;
    int x,y,c,fact_x;

    for ( c = 0; c <ic ; ++c)
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
    for ( c = 0; c <ic ; ++c)
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
}
__global__ void kernel_doublesizebyshare1(float *out,float *in,int const ow,int const oh,int const iw,int const ih,int const ic)
{
    extern __shared__ float  data[];
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*ic*2;
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;

    int share_x=(blockDim.x>>1)+1;//共享内存块x维（需乘ic）
    int share_y=(blockDim.y>>1)+1;//共享内存块y维
    int share_fact_x=share_x*ic*2;
    int share_idx_x;
    int share_idx_y= threadIdx.y;//共享内存块内y维索引
    int in_x0 = ((blockIdx.x * blockDim.x*2) >> 1) * ic;
    int in_y0 = (blockIdx.y * blockDim.y) >> 1;
    int x,y,c,fact_x;

    for ( c = 0; c <ic*2 ; ++c)
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
    for ( c = 0; c <ic*2 ; ++c)
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
}
__global__ void kernel_doublesizebyshare2(float *out,float *in,int const ow,int const oh,int const iw,int const ih,int const ic)
{
    extern __shared__ float  data[];
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*ic*3;
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;

    int share_x=(blockDim.x>>1)+1;//共享内存块x维（需乘ic）
    int share_y=(blockDim.y>>1)+1;//共享内存块y维
    int share_fact_x=share_x*ic*3;
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
}

/******************************************************************************************/
///功能：图片缩小两倍
/*  函数名                            线程块大小       耗费时间
 *kernel_halfsize		            636.275us	    [32,8,1]
 *kernel_halfsize1                  634.383us	    [32,8,1]**
 *kernel_halfsize2                  641.6us	        [32,8,1]
 *kernel_halfsizebyshare	    	643.698us	    [32,4,1]
 *kernel_halfsizebyshare1	  		671.245us	    [32,4,1]
 */
/******************************************************************************************/

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
__global__ void kernel_halfsize2(float *out,float *in,int const ow,int const oh,int const iw,int const ih,int const ic)
{
    //若需要展开ic*3重循环只需修改out_x=threadIdx.x+blockIdx.x*blockDim.x*ic*3;以及for(int c=0;c<ic*3;c++)即可，同时应修改网格大小
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*ic*3;
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;
    int stride=iw*ic;

    for(int c=0;c<ic*3;c++)
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
    int block_stride=blockDim.x*ic;//线程块x维间隔
    int out_x=threadIdx.x+blockIdx.x*block_stride;//输出的x维起始索引
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;//输出的y位索引
    int stride=iw*ic;//输入图像的行索引的最大值

    int in_x0=blockIdx.x*block_stride*2;//输入图像x维的起始点
    int in_y0=blockIdx.y*blockDim.y*2;//输入图像y维的起始点
    int in_x1=in_x0+block_stride;
    int in_y1=in_y0+blockDim.y;

    int share_x=blockDim.x*2*ic;//共享块内x维最大像素点个数
    for (int c = 0; c < ic; ++c)
    {

        int fact_x_s=threadIdx.x+blockDim.x*c;
        int channel=fact_x_s%ic;//第几个通道
        int x_s=fact_x_s+block_stride;
        int y_s0=threadIdx.y*share_x;
        int y_s1=y_s0+blockDim.y*share_x;
        int fact_iw=channel+stride-ic;
        int x0=min(in_x0+fact_x_s,fact_iw);
        int x1=min(in_x1+fact_x_s,fact_iw);
        int y0=min(in_y0+threadIdx.y,ih-1)*stride;
        int y1=min(in_y1+threadIdx.y,ih-1)*stride;

        int deta=((fact_x_s/ic)%2)*block_stride;//像素点的x坐标是否为奇数

        int x_fs0=(fact_x_s/ic>>1)*ic+channel+deta;//共享内存内存储第一个x坐标
        int x_fs1=(x_s/ic>>1)*ic+channel+deta;//共享内存内存储第二个x坐标

        data[y_s0+x_fs0]=in[y0+x0];
        data[y_s0+x_fs1]=in[y0+x1];
        data[y_s1+x_fs0]=in[y1+x0];
        data[y_s1+x_fs1]=in[y1+x1];;
    }
    __syncthreads();
    for (int c = 0; c <ic ; ++c)
    {
        int fact_x=out_x+blockDim.x*c;

        if(out_y<oh&&fact_x<ow*ic)
        {
            int srow1=threadIdx.y*2*share_x;
            int srow2=srow1+share_x;

            int scol1=threadIdx.x+blockDim.x*c;
            int scol2=scol1+block_stride;
            int index[4] = {srow1 + scol1,
                            srow1 + scol2,
                            srow2 + scol1,
                            srow2 + scol2};
            int out_idx = out_y * ow*ic + fact_x;
            out[out_idx] = 0.25f * (data[index[0]] + data[index[1]] + data[index[2]] + data[index[3]]);
        }
    }
}

__global__ void kernel_halfsizebyshare1(float *out,float *in,int const ow,int const oh,int const iw,int const ih,int const ic)
{
    extern __shared__ float data[];
    int block_stride=blockDim.x*ic*2;//线程块x维间隔
    int out_x=threadIdx.x+blockIdx.x*block_stride;//输出的x维起始索引
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;//输出的y位索引
    int stride=iw*ic;//输入图像的行索引的最大值

    int in_x0=blockIdx.x*block_stride*2;//输入图像x维的起始点
    int in_y0=blockIdx.y*blockDim.y*2;//输入图像y维的起始点
    int in_x1=in_x0+block_stride;
    int in_y1=in_y0+blockDim.y;

    int share_x=blockDim.x*4*ic;//共享块内x维最大像素点个数
    for (int c = 0; c < ic*2; ++c)
    {

        int fact_x_s=threadIdx.x+blockDim.x*c;
        int channel=fact_x_s%ic;//第几个通道
        int x_s=fact_x_s+block_stride;
        int y_s0=threadIdx.y*share_x;
        int y_s1=y_s0+blockDim.y*share_x;
        int fact_iw=channel+stride-ic;
        int x0=min(in_x0+fact_x_s,fact_iw);
        int x1=min(in_x1+fact_x_s,fact_iw);
        int y0=min(in_y0+threadIdx.y,ih-1)*stride;
        int y1=min(in_y1+threadIdx.y,ih-1)*stride;

        int deta=((fact_x_s/ic)%2)*block_stride;//像素点的x坐标是否为奇数

        int x_fs0=(fact_x_s/ic>>1)*ic+channel+deta;//共享内存内存储第一个x坐标
        int x_fs1=(x_s/ic>>1)*ic+channel+deta;//共享内存内存储第二个x坐标

        data[y_s0+x_fs0]=in[y0+x0];
        data[y_s0+x_fs1]=in[y0+x1];
        data[y_s1+x_fs0]=in[y1+x0];
        data[y_s1+x_fs1]=in[y1+x1];;
    }
    __syncthreads();
    for (int c = 0; c <ic*2 ; ++c)
    {
        int fact_x=out_x+blockDim.x*c;

        if(out_y<oh&&fact_x<ow*ic)
        {
            int srow1=threadIdx.y*2*share_x;
            int srow2=srow1+share_x;

            int scol1=threadIdx.x+blockDim.x*c;
            int scol2=scol1+block_stride;
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
///功能：分离颜色通道
/*  函数名                            线程块大小       耗费时间
 *kernel_split		                 [32,4,1]         1.071ms
 *kernel_split1                      [32,4,1]         1.06ms
 *kernel_split2                      [32,4,1]         1.058ms
 *kernel_splitbyshare	    	     [32,8,1]         1.064ms
 *kernel_splitbyshare1	  		     [32,8,1]         1.059ms
 *kernel_splitbyshare2               [32,4,1]         1.057ms
 */
/******************************************************************************************/
/* 调用示例
 * dim3 block1(x, y, 1);
 * dim3 grid1((weight - 1 + x) / x, (height - 1 + y) / y, 1);
 * kernel_splitbyshare <<< grid1, block1, x * y * 3 * sizeof(float) >>> (d_c_0, d_c_1, d_c_2, d_in, weight, height);
 */
__global__ void kernel_splitbyshare(float *out_channels_0,float *out_channels_1,float *out_channels_2,float * in,int const weight,int const height)
{
    extern __shared__ float data[];
    int out_x=threadIdx.x+blockIdx.x*blockDim.x;
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;
    int idx=out_y*weight+out_x;
    int fidx=threadIdx.y*blockDim.x*3+threadIdx.x*3;
    int share_x=blockDim.x*3;//共享块x维长度

    int shidx0=threadIdx.y*share_x+blockDim.x*0+threadIdx.x;
    int shidx1=threadIdx.y*share_x+blockDim.x*1+threadIdx.x;
    int shidx2=threadIdx.y*share_x+blockDim.x*2+threadIdx.x;
    int inidx0=out_y*weight*3+blockIdx.x*share_x+blockDim.x*0+threadIdx.x;
    int inidx1=out_y*weight*3+blockIdx.x*share_x+blockDim.x*1+threadIdx.x;
    int inidx2=out_y*weight*3+blockIdx.x*share_x+blockDim.x*2+threadIdx.x;

    if(out_x<weight&&out_y<height)
    {
        data[shidx0]=in[inidx0];
        data[shidx1]=in[inidx1];
        data[shidx2]=in[inidx2];
        __syncthreads();
        out_channels_0[idx]=data[fidx+0];
        out_channels_1[idx]=data[fidx+1];
        out_channels_2[idx]=data[fidx+2];
    }
}

/* 调用示例
 * dim3 block3(x, y, 1);
 * dim3 grid3((weight - 1 + x*2) / (x*2), (height - 1 + y) / y, 1);
 * kernel_splitbyshare1<<<grid3, block3, x * y * 6 * sizeof(float) >>> (d_c_0, d_c_1, d_c_2, d_in, weight, height);
 */
__global__ void kernel_splitbyshare1(float *out_channels_0,float *out_channels_1,float *out_channels_2,float * in,int const weight,int const height)
{
    extern __shared__ float data[];
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*2;
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;
    int idx=out_y*weight+out_x;

    int share_x=blockDim.x*6;//共享块x维最大值
    int shsp=threadIdx.y*share_x+threadIdx.x;//共享块内索引起点（start point）
    int insp=out_y*weight*3+blockIdx.x*share_x+threadIdx.x;//输入数组内索引起点;
    int fidx=threadIdx.y*share_x+threadIdx.x*3;
    int inc=blockDim.x*3;//增量

    int shidx0=shsp+blockDim.x*0;
    int shidx1=shsp+blockDim.x*1;
    int shidx2=shsp+blockDim.x*2;
    int shidx3=shsp+blockDim.x*3;
    int shidx4=shsp+blockDim.x*4;
    int shidx5=shsp+blockDim.x*5;

    int inidx0=insp+blockDim.x*0;
    int inidx1=insp+blockDim.x*1;
    int inidx2=insp+blockDim.x*2;
    int inidx3=insp+blockDim.x*3;
    int inidx4=insp+blockDim.x*4;
    int inidx5=insp+blockDim.x*5;

    if(out_x<weight&&out_y<height)
    {
        data[shidx0]=in[inidx0];
        data[shidx1]=in[inidx1];
        data[shidx2]=in[inidx2];
        data[shidx3]=in[inidx3];
        data[shidx4]=in[inidx4];
        data[shidx5]=in[inidx5];
        __syncthreads();
        out_channels_0[idx]=data[fidx+0];
        out_channels_1[idx]=data[fidx+1];
        out_channels_2[idx]=data[fidx+2];
        out_channels_0[idx+blockDim.x]=data[fidx+inc+0];
        out_channels_1[idx+blockDim.x]=data[fidx+inc+1];
        out_channels_2[idx+blockDim.x]=data[fidx+inc+2];
    }
}

/* 调用示例
 * dim3 block4(x, y, 1);
 * dim3 grid4((weight - 1 + x*3) / (x*3), (height - 1 + y) / y, 1);
 * kernel_splitbyshare2<<<grid4, block4, x * y * 9 * sizeof(float) >>> (d_c_0, d_c_1, d_c_2, d_in, weight, height);
 */
__global__ void kernel_splitbyshare2(float *out_channels_0,float *out_channels_1,float *out_channels_2,float * in,int const weight,int const height)
{
    extern __shared__ float data[];
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*3;
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;
    int idx=out_y*weight+out_x;

    int share_x=blockDim.x*9;//共享块x维最大值
    int shsp=threadIdx.y*share_x+threadIdx.x;//共享块内索引起点（start point）
    int insp=out_y*weight*3+blockIdx.x*share_x+threadIdx.x;//输入数组内索引起点;
    int fidx=threadIdx.y*share_x+threadIdx.x*3;
    int inc=blockDim.x*3;//增量
    int inc1=blockDim.x*6;//增量

    int shidx0=shsp+blockDim.x*0;
    int shidx1=shsp+blockDim.x*1;
    int shidx2=shsp+blockDim.x*2;
    int shidx3=shsp+blockDim.x*3;
    int shidx4=shsp+blockDim.x*4;
    int shidx5=shsp+blockDim.x*5;
    int shidx6=shsp+blockDim.x*6;
    int shidx7=shsp+blockDim.x*7;
    int shidx8=shsp+blockDim.x*8;

    int inidx0=insp+blockDim.x*0;
    int inidx1=insp+blockDim.x*1;
    int inidx2=insp+blockDim.x*2;
    int inidx3=insp+blockDim.x*3;
    int inidx4=insp+blockDim.x*4;
    int inidx5=insp+blockDim.x*5;
    int inidx6=insp+blockDim.x*6;
    int inidx7=insp+blockDim.x*7;
    int inidx8=insp+blockDim.x*8;

    if(out_x<weight&&out_y<height)
    {
        data[shidx0]=in[inidx0];
        data[shidx1]=in[inidx1];
        data[shidx2]=in[inidx2];
        data[shidx3]=in[inidx3];
        data[shidx4]=in[inidx4];
        data[shidx5]=in[inidx5];
        data[shidx6]=in[inidx6];
        data[shidx7]=in[inidx7];
        data[shidx8]=in[inidx8];
        __syncthreads();
        out_channels_0[idx]=data[fidx+0];
        out_channels_1[idx]=data[fidx+1];
        out_channels_2[idx]=data[fidx+2];
        out_channels_0[idx+blockDim.x]=data[fidx+inc+0];
        out_channels_1[idx+blockDim.x]=data[fidx+inc+1];
        out_channels_2[idx+blockDim.x]=data[fidx+inc+2];
        out_channels_0[idx+blockDim.x*2]=data[fidx+inc1+0];
        out_channels_1[idx+blockDim.x*2]=data[fidx+inc1+1];
        out_channels_2[idx+blockDim.x*2]=data[fidx+inc1+2];

    }
}

/* 调用示例
 * dim3 block2(x, y, 1);
 * dim3 grid2((weight - 1 + x) / x, (height - 1 + y) / y, 1);
 * kernel_split<<< grid2, block2>>> (d_c_0, d_c_1, d_c_2, d_in, weight, height);
 */
__global__ void kernel_split(float *out_channels_0,float *out_channels_1,float *out_channels_2,float * in,int const weight,int const height)
{
    int out_x=threadIdx.x+blockIdx.x*blockDim.x;
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;
    int idx=out_y*weight+out_x;
    int inidx=out_y * weight * 3 + out_x * 3;
    if(out_x<weight&&out_y<height) {
        float a=in[inidx+0];
        float b=in[inidx+1];
        float c=in[inidx+2];
        out_channels_0[idx] = a;
        out_channels_1[idx] = b;
        out_channels_2[idx] = c;
    }
}

/* 调用示例
 * dim3 block5(x, y, 1);
 * dim3 grid5((weight - 1 + x*2) / (x*2), (height - 1 + y) / y, 1);
 * kernel_split1<<< grid5, block5>>> (d_c_0, d_c_1, d_c_2, d_in, weight, height);
 */
__global__ void kernel_split1(float *out_channels_0,float *out_channels_1,float *out_channels_2,float * in,int const weight,int const height)
{
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*2;
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;
    int idx=out_y*weight+out_x;
    int inidx=out_y * weight * 3 + out_x * 3;
    if(out_x<weight&&out_y<height) {
        float a=in[inidx+0];
        float b=in[inidx+1];
        float c=in[inidx+2];
        out_channels_0[idx] = a;
        out_channels_1[idx] = b;
        out_channels_2[idx] = c;
        a=in[inidx +blockDim.x*3+ 0];
        b=in[inidx +blockDim.x*3+ 1];
        c=in[inidx +blockDim.x*3+ 2];
        out_channels_0[idx+blockDim.x] = a;
        out_channels_1[idx+blockDim.x] = b;
        out_channels_2[idx+blockDim.x] = c;
    }
}

/* 调用示例
 * dim3 block6(x, y, 1);
 * dim3 grid6((weight - 1 + x*3) / (x*3), (height - 1 + y) / y, 1);
 * kernel_split2<<< grid6, block6>>> (d_c_0, d_c_1, d_c_2, d_in, weight, height);
 */
__global__ void kernel_split2(float *out_channels_0,float *out_channels_1,float *out_channels_2,float * in,int const weight,int const height)
{
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*3;
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;
    int idx=out_y*weight+out_x;
    int inidx=out_y * weight * 3 + out_x * 3;
    if(out_x<weight&&out_y<height) {
        float a=in[inidx+0];
        float b=in[inidx+1];
        float c=in[inidx+2];
        out_channels_0[idx] = a;
        out_channels_1[idx] = b;
        out_channels_2[idx] = c;
        a=in[inidx +blockDim.x*3+ 0];
        b=in[inidx +blockDim.x*3+ 1];
        c=in[inidx +blockDim.x*3+ 2];
        out_channels_0[idx+blockDim.x] = a;
        out_channels_1[idx+blockDim.x] = b;
        out_channels_2[idx+blockDim.x] = c;
        a=in[inidx +blockDim.x*6+ 0];
        b=in[inidx +blockDim.x*6+ 1];
        c=in[inidx +blockDim.x*6+ 2];
        out_channels_0[idx+blockDim.x*2] = a;
        out_channels_1[idx+blockDim.x*2] = b;
        out_channels_2[idx+blockDim.x*2] = c;
    }
}
/******************************************************************************************/
///调用核函数实现加速功能
/******************************************************************************************/

void warm(void)
{
    warmup<<<1,1>>>();
}

void double_size_by_cuda(float * const out_image,float const  * const in_image,int const weight,int const height,int const channels,float const * const out)
{
    int const ow=weight<<1;
    int const oh=height<<1;
    int const size_in=weight*height;
    int const size_out=ow*oh;
    size_t const bytes_in=size_in*channels* sizeof(float);
    size_t const bytes_out=size_out*channels* sizeof(float);


    float *d_in=NULL;
    float *d_out=NULL;
    cudaMalloc((void**)&d_in,bytes_in);
    cudaMalloc((void**)&d_out,bytes_out);
    cudaMemcpy(d_in,in_image,bytes_in,cudaMemcpyHostToDevice);

    int x=32;
    int y=4;
    dim3 block2 (x,y,1);
    dim3 grid2 ((ow-1+x*3)/(x*3),(oh-1+y)/y,1);
    cudaMalloc((void**)&d_out,bytes_out);
    kernel_doublesize2<<<grid2,block2>>>(d_out,d_in,ow,oh,weight,channels);
    cudaMemcpy(out_image,d_out,bytes_out,cudaMemcpyDeviceToHost);

/*
    int x=32;
    int y=16;
    dim3 block (x,y,1);
    dim3 grid ((ow-1+x)/x,(oh-1+y)/y,1);
    kernel_doublesize<<<grid,block>>>(d_out,d_in,ow,oh,weight,channels);
    cudaMemcpy(out_image,d_out,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image,out,ow,oh,channels);

//缩小block
    x=32;
    y=8;
    dim3 block1 (x,y,1);
    dim3 grid1 ((ow-1+x)/x,(oh-1+y)/y,1);
    float *d_out1=NULL;
    float *out_image1= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out1,bytes_out);
    kernel_doublesize<<<grid1,block1>>>(d_out1,d_in,ow,oh,weight,channels);
    cudaMemcpy(out_image1,d_out1,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image1,out,ow,oh,channels);
//缩小block
    x=32;
    y=4;
    dim3 block2 (x,y,1);
    dim3 grid2 ((ow-1+x)/x,(oh-1+y)/y,1);
    float *d_out2=NULL;
    float *out_image2= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out2,bytes_out);
    kernel_doublesize<<<grid2,block2>>>(d_out2,d_in,ow,oh,weight,channels);
    cudaMemcpy(out_image2,d_out2,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image2,out,ow,oh,channels);
//缩小block
    x=32;
    y=32;
    dim3 block3 (x,y,1);
    dim3 grid3 ((ow-1+x)/x,(oh-1+y)/y,1);
    float *d_out3=NULL;
    float *out_image3= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out3,bytes_out);
    kernel_doublesize<<<grid3,block3>>>(d_out3,d_in,ow,oh,weight,channels);
    cudaMemcpy(out_image3,d_out3,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image3,out,ow,oh,channels);
//放大block
    x=64;
    y=8;
    dim3 block4 (x,y,1);
    dim3 grid4 ((ow-1+x)/x,(oh-1+y)/y,1);
    float *d_out4=NULL;
    float *out_image4= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out4,bytes_out);
    kernel_doublesize<<<grid4,block4>>>(d_out4,d_in,ow,oh,weight,channels);
    cudaMemcpy(out_image4,d_out4,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image4,out,ow,oh,channels);
*/
/*
    int x=32;
    int y=16;
    dim3 block (x,y,1);
    dim3 grid ((ow-1+x*2)/(x*2),(oh-1+y)/y,1);
    kernel_doublesize1<<<grid,block>>>(d_out,d_in,ow,oh,weight,channels);
    cudaMemcpy(out_image,d_out,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image,out,ow,oh,channels);
//缩小block
    x=32;
    y=8;
    dim3 block1 (x,y,1);
    dim3 grid1 ((ow-1+x*2)/(x*2),(oh-1+y)/y,1);
    float *d_out1=NULL;
    float *out_image1= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out1,bytes_out);
    kernel_doublesize1<<<grid1,block1>>>(d_out1,d_in,ow,oh,weight,channels);
    cudaMemcpy(out_image1,d_out1,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image1,out,ow,oh,channels);
//缩小block
    x=32;
    y=4;
    dim3 block2 (x,y,1);
    dim3 grid2 ((ow-1+x*2)/(x*2),(oh-1+y)/y,1);
    float *d_out2=NULL;
    float *out_image2= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out2,bytes_out);
    kernel_doublesize1<<<grid2,block2>>>(d_out2,d_in,ow,oh,weight,channels);
    cudaMemcpy(out_image2,d_out2,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image2,out,ow,oh,channels);
//缩小block
    x=32;
    y=32;
    dim3 block3 (x,y,1);
    dim3 grid3 ((ow-1+x*2)/(x*2),(oh-1+y)/y,1);
    float *d_out3=NULL;
    float *out_image3= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out3,bytes_out);
    kernel_doublesize1<<<grid3,block3>>>(d_out3,d_in,ow,oh,weight,channels);
    cudaMemcpy(out_image3,d_out3,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image3,out,ow,oh,channels);
//放大block
    x=64;
    y=8;
    dim3 block4 (x,y,1);
    dim3 grid4 ((ow-1+x*2)/(x*2),(oh-1+y)/y,1);
    float *d_out4=NULL;
    float *out_image4= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out4,bytes_out);
    kernel_doublesize1<<<grid4,block4>>>(d_out4,d_in,ow,oh,weight,channels);
    cudaMemcpy(out_image4,d_out4,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image4,out,ow,oh,channels);
    */
/*
    int x=32;
    int y=16;
    dim3 block (x,y,1);
    dim3 grid ((ow-1+x*3)/(x*3),(oh-1+y)/y,1);
    kernel_doublesize2<<<grid,block>>>(d_out,d_in,ow,oh,weight,channels);
    cudaMemcpy(out_image,d_out,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image,out,ow,oh,channels);
//缩小block
    x=32;
    y=8;
    dim3 block1 (x,y,1);
    dim3 grid1 ((ow-1+x*3)/(x*3),(oh-1+y)/y,1);
    float *d_out1=NULL;
    float *out_image1= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out1,bytes_out);
    kernel_doublesize2<<<grid1,block1>>>(d_out1,d_in,ow,oh,weight,channels);
    cudaMemcpy(out_image1,d_out1,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image1,out,ow,oh,channels);
//缩小block
    int x=32;
    int y=4;
    dim3 block2 (x,y,1);
    dim3 grid2 ((ow-1+x*3)/(x*3),(oh-1+y)/y,1);
    float *d_out2=NULL;
    float *out_image2= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out2,bytes_out);
    kernel_doublesize2<<<grid2,block2>>>(d_out2,d_in,ow,oh,weight,channels);
    cudaMemcpy(out_image2,d_out2,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image2,out,ow,oh,channels);

//缩小block
    x=32;
    y=32;
    dim3 block3 (x,y,1);
    dim3 grid3 ((ow-1+x*3)/(x*3),(oh-1+y)/y,1);
    float *d_out3=NULL;
    float *out_image3= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out3,bytes_out);
    kernel_doublesize2<<<grid3,block3>>>(d_out3,d_in,ow,oh,weight,channels);
    cudaMemcpy(out_image3,d_out3,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image3,out,ow,oh,channels);
//放大block
    x=64;
    y=8;
    dim3 block4 (x,y,1);
    dim3 grid4 ((ow-1+x*3)/(x*3),(oh-1+y)/y,1);
    float *d_out4=NULL;
    float *out_image4= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out4,bytes_out);
    kernel_doublesize2<<<grid4,block4>>>(d_out4,d_in,ow,oh,weight,channels);
    cudaMemcpy(out_image4,d_out4,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image4,out,ow,oh,channels);
*/
/*
    int x=32;
    int y=16;
    int share_x=((x>>1)+1);
    int share_y=(y>>1)+1;
    dim3 block (x,y,1);
    dim3 grid ((ow-1+x)/x,(oh-1+y)/y,1);
    kernel_doublesizebyshare<<<grid,block,share_x*share_y*channels*sizeof(float)>>>(d_out,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image,d_out,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image,out,ow,oh,channels);

//缩小block
    x=32;
    y=8;
    share_x=((x>>1)+1);
    share_y=(y>>1)+1;
    dim3 block1 (x,y,1);
    dim3 grid1 ((ow-1+x)/x,(oh-1+y)/y,1);
    float *d_out1=NULL;
    float *out_image1= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out1,bytes_out);
    kernel_doublesizebyshare<<<grid1,block1,share_x*share_y*channels*sizeof(float)>>>(d_out1,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image1,d_out1,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image1,out,ow,oh,channels);
//缩小block
    x=32;
    y=4;
    share_x=((x>>1)+1);
    share_y=(y>>1)+1;
    dim3 block2 (x,y,1);
    dim3 grid2 ((ow-1+x)/x,(oh-1+y)/y,1);
    float *d_out2=NULL;
    float *out_image2= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out2,bytes_out);
    kernel_doublesizebyshare<<<grid2,block2,share_x*share_y*channels*sizeof(float)>>>(d_out2,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image2,d_out2,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image2,out,ow,oh,channels);
//缩小block
    x=32;
    y=32;
    share_x=((x>>1)+1);
    share_y=(y>>1)+1;
    dim3 block3 (x,y,1);
    dim3 grid3 ((ow-1+x)/x,(oh-1+y)/y,1);
    float *d_out3=NULL;
    float *out_image3= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out3,bytes_out);
    kernel_doublesizebyshare<<<grid3,block3,share_x*share_y*channels*sizeof(float)>>>(d_out3,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image3,d_out3,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image3,out,ow,oh,channels);
//放大block
    x=64;
    y=8;
    share_x=((x>>1)+1);
    share_y=(y>>1)+1;
    dim3 block4 (x,y,1);
    dim3 grid4 ((ow-1+x)/x,(oh-1+y)/y,1);
    float *d_out4=NULL;
    float *out_image4= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out4,bytes_out);
    kernel_doublesizebyshare<<<grid4,block4,share_x*share_y*channels*sizeof(float)>>>(d_out4,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image4,d_out4,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image4,out,ow,oh,channels);
*/
/*
    int x=32;
    int y=16;
    int share_x=((x>>1)+1);
    int share_y=(y>>1)+1;
    dim3 block (x,y,1);
    dim3 grid ((ow-1+x*2)/(x*2),(oh-1+y)/y,1);
    kernel_doublesizebyshare1<<<grid,block,share_x*share_y*2*channels*sizeof(float)>>>(d_out,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image,d_out,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image,out,ow,oh,channels);
//缩小block
    x=32;
    y=8;
    share_x=((x>>1)+1);
    share_y=(y>>1)+1;
    dim3 block1 (x,y,1);
    dim3 grid1 ((ow-1+x*2)/(x*2),(oh-1+y)/y,1);
    float *d_out1=NULL;
    float *out_image1= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out1,bytes_out);
    kernel_doublesizebyshare1<<<grid1,block1,share_x*share_y*2*channels*sizeof(float)>>>(d_out1,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image1,d_out1,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image1,out,ow,oh,channels);
//缩小block
    x=32;
    y=4;
    share_x=((x>>1)+1);
    share_y=(y>>1)+1;
    dim3 block2 (x,y,1);
    dim3 grid2 ((ow-1+x*2)/(x*2),(oh-1+y)/y,1);
    float *d_out2=NULL;
    float *out_image2= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out2,bytes_out);
    kernel_doublesizebyshare1<<<grid2,block2,share_x*share_y*2*channels*sizeof(float)>>>(d_out2,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image2,d_out2,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image2,out,ow,oh,channels);
//缩小block
    x=32;
    y=32;
    share_x=((x>>1)+1);
    share_y=(y>>1)+1;
    dim3 block3 (x,y,1);
    dim3 grid3 ((ow-1+x*2)/(x*2),(oh-1+y)/y,1);
    float *d_out3=NULL;
    float *out_image3= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out3,bytes_out);
    kernel_doublesizebyshare1<<<grid3,block3,share_x*share_y*2*channels*sizeof(float)>>>(d_out3,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image3,d_out3,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image3,out,ow,oh,channels);
//放大block
    x=64;
    y=8;
    share_x=((x>>1)+1);
    share_y=(y>>1)+1;
    dim3 block4 (x,y,1);
    dim3 grid4 ((ow-1+x*2)/(x*2),(oh-1+y)/y,1);
    float *d_out4=NULL;
    float *out_image4= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out4,bytes_out);
    kernel_doublesizebyshare1<<<grid4,block4,share_x*share_y*2*channels*sizeof(float)>>>(d_out4,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image4,d_out4,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image4,out,ow,oh,channels);
*/
/*
    int x=32;
    int y=16;
    int share_x=((x>>1)+1);
    int share_y=(y>>1)+1;
    dim3 block (x,y,1);
    dim3 grid ((ow-1+x*3)/(x*3),(oh-1+y)/y,1);
    kernel_doublesizebyshare2<<<grid,block,share_x*share_y*3*channels*sizeof(float)>>>(d_out,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image,d_out,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image,out,ow,oh,channels);
//缩小block
    x=32;
    y=8;
    share_x=((x>>1)+1);
    share_y=(y>>1)+1;
    dim3 block1 (x,y,1);
    dim3 grid1 ((ow-1+x*3)/(x*3),(oh-1+y)/y,1);
    float *d_out1=NULL;
    float *out_image1= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out1,bytes_out);
    kernel_doublesizebyshare2<<<grid1,block1,share_x*share_y*3*channels*sizeof(float)>>>(d_out1,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image1,d_out1,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image1,out,ow,oh,channels);
//缩小block
    x=32;
    y=4;
    share_x=((x>>1)+1);
    share_y=(y>>1)+1;
    dim3 block2 (x,y,1);
    dim3 grid2 ((ow-1+x*3)/(x*3),(oh-1+y)/y,1);
    float *d_out2=NULL;
    float *out_image2= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out2,bytes_out);
    kernel_doublesizebyshare2<<<grid2,block2,share_x*share_y*3*channels*sizeof(float)>>>(d_out2,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image2,d_out2,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image2,out,ow,oh,channels);
//缩小block
    x=32;
    y=32;
    share_x=((x>>1)+1);
    share_y=(y>>1)+1;
    dim3 block3 (x,y,1);
    dim3 grid3 ((ow-1+x*3)/(x*3),(oh-1+y)/y,1);
    float *d_out3=NULL;
    float *out_image3= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out3,bytes_out);
    kernel_doublesizebyshare2<<<grid3,block3,share_x*share_y*3*channels*sizeof(float)>>>(d_out3,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image3,d_out3,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image3,out,ow,oh,channels);
//放大block
    x=64;
    y=8;
    share_x=((x>>1)+1);
    share_y=(y>>1)+1;
    dim3 block4 (x,y,1);
    dim3 grid4 ((ow-1+x*3)/(x*3),(oh-1+y)/y,1);
    float *d_out4=NULL;
    float *out_image4= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out4,bytes_out);
    kernel_doublesizebyshare2<<<grid4,block4,share_x*share_y*3*channels*sizeof(float)>>>(d_out4,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image4,d_out4,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image4,out,ow,oh,channels);
    */
//释放分配的内存
    /*cudaFree(d_out1);
    cudaFree(d_out2);
    cudaFree(d_out3);
    cudaFree(d_out4);
    free(out_image1);
    free(out_image2);
    free(out_image3);
    free(out_image4);*/

    cudaFree(d_in);
    cudaFree(d_out);
}

void halfsize_by_cuda(float * const out_image,float const  * const in_image,int const weight,int const height,int const channels,float const  * const out)
{
    int ow=(weight+1)>>1;
    int oh=(height+1)>>1;
    int const size_in=weight*height;
    int const size_out=ow*oh;
    size_t const bytes_in=size_in*channels* sizeof(float);
    size_t const bytes_out=size_out*channels* sizeof(float);

    float *d_in=NULL;
    float *d_out=NULL;
    cudaMalloc((void**)&d_out,bytes_out);
    cudaMalloc((void**)&d_in,bytes_in);
    cudaMemcpy(d_in,in_image,bytes_in,cudaMemcpyHostToDevice);

    int const x=32;
    int const y=8;
    dim3 block (x,y,1);
    dim3 grid ((ow-1+x*2)/(x*2),(oh-1+y)/y,1);
    kernel_halfsize1<<<grid,block>>>(d_out,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image,d_out,bytes_out,cudaMemcpyDeviceToHost);
    //compare(out_image,out,ow,oh,channels);//对比运行结果
/*
    int x=32;
    int y=16;
    dim3 block (x,y,1);
    dim3 grid ((ow-1+x)/x,(oh-1+y)/y,1);
    kernel_halfsize<<<grid,block>>>(d_out,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image,d_out,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image,out,ow,oh,channels);
//缩小block
    x=32;
    y=8;
    float *d_out1=NULL;
    float *out_image1= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out1,bytes_out);
    dim3 block1 (x,y,1);
    dim3 grid1 ((ow-1+x)/x,(oh-1+y)/y,1);
    kernel_halfsize<<<grid1,block1>>>(d_out1,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image1,d_out1,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image1,out,ow,oh,channels);
//缩小block
    x=32;
    y=4;
    float *d_out2=NULL;
    float *out_image2= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out2,bytes_out);
    dim3 block2 (x,y,1);
    dim3 grid2 ((ow-1+x)/x,(oh-1+y)/y,1);
    kernel_halfsize<<<grid2,block2>>>(d_out2,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image2,d_out2,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image2,out,ow,oh,channels);
//缩小block
    x=32;
    y=32;
    float *d_out3=NULL;
    float *out_image3= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out3,bytes_out);
    dim3 block3 (x,y,1);
    dim3 grid3 ((ow-1+x)/x,(oh-1+y)/y,1);
    kernel_halfsize<<<grid3,block3>>>(d_out3,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image3,d_out3,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image3,out,ow,oh,channels);
//放大block
    x=64;
    y=8;
    float *d_out4=NULL;
    float *out_image4= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out4,bytes_out);
    dim3 block4 (x,y,1);
    dim3 grid4 ((ow-1+x)/x,(oh-1+y)/y,1);
    kernel_halfsize<<<grid4,block4>>>(d_out4,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image4,d_out4,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image4,out,ow,oh,channels);
*/
/*
    int x=32;
    int y=16;
    dim3 block (x,y,1);
    dim3 grid ((ow-1+x*2)/(x*2),(oh-1+y)/y,1);
    kernel_halfsize1<<<grid,block>>>(d_out,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image,d_out,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image,out,ow,oh,channels);
//缩小block
    x=32;
    y=8;
    float *d_out1=NULL;
    float *out_image1= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out1,bytes_out);
    dim3 block1 (x,y,1);
    dim3 grid1 ((ow-1+x*2)/(x*2),(oh-1+y)/y,1);
    kernel_halfsize1<<<grid1,block1>>>(d_out1,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image1,d_out1,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image1,out,ow,oh,channels);
//缩小block
    x=32;
    y=4;
    float *d_out2=NULL;
    float *out_image2= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out2,bytes_out);
    dim3 block2 (x,y,1);
    dim3 grid2 ((ow-1+x*2)/(x*2),(oh-1+y)/y,1);
    kernel_halfsize1<<<grid2,block2>>>(d_out2,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image2,d_out2,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image2,out,ow,oh,channels);
//缩小block
    x=32;
    y=32;
    float *d_out3=NULL;
    float *out_image3= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out3,bytes_out);
    dim3 block3 (x,y,1);
    dim3 grid3 ((ow-1+x*2)/(x*2),(oh-1+y)/y,1);
    kernel_halfsize1<<<grid3,block3>>>(d_out3,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image3,d_out3,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image3,out,ow,oh,channels);
//放大block
    x=64;
    y=8;
    float *d_out4=NULL;
    float *out_image4= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out4,bytes_out);
    dim3 block4 (x,y,1);
    dim3 grid4 ((ow-1+x*2)/(x*2),(oh-1+y)/y,1);
    kernel_halfsize1<<<grid4,block4>>>(d_out4,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image4,d_out4,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image4,out,ow,oh,channels);
*/
/*
    int x=32;
    int y=16;
    dim3 block (x,y,1);
    dim3 grid ((ow-1+x*3)/(x*3),(oh-1+y)/y,1);
    kernel_halfsize2<<<grid,block>>>(d_out,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image,d_out,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image,out,ow,oh,channels);
//缩小block
    x=32;
    y=8;
    float *d_out1=NULL;
    float *out_image1= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out1,bytes_out);
    dim3 block1 (x,y,1);
    dim3 grid1 ((ow-1+x*3)/(x*3),(oh-1+y)/y,1);
    kernel_halfsize2<<<grid1,block1>>>(d_out1,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image1,d_out1,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image1,out,ow,oh,channels);
//缩小block
    x=32;
    y=4;
    float *d_out2=NULL;
    float *out_image2= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out2,bytes_out);
    dim3 block2 (x,y,1);
    dim3 grid2 ((ow-1+x*3)/(x*3),(oh-1+y)/y,1);
    kernel_halfsize2<<<grid2,block2>>>(d_out2,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image2,d_out2,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image2,out,ow,oh,channels);
//缩小block
    x=32;
    y=32;
    float *d_out3=NULL;
    float *out_image3= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out3,bytes_out);
    dim3 block3 (x,y,1);
    dim3 grid3 ((ow-1+x*3)/(x*3),(oh-1+y)/y,1);
    kernel_halfsize2<<<grid3,block3>>>(d_out3,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image3,d_out3,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image3,out,ow,oh,channels);
//放大block
    x=64;
    y=8;
    float *d_out4=NULL;
    float *out_image4= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out4,bytes_out);
    dim3 block4 (x,y,1);
    dim3 grid4 ((ow-1+x*3)/(x*3),(oh-1+y)/y,1);
    kernel_halfsize2<<<grid4,block4>>>(d_out4,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image4,d_out4,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image4,out,ow,oh,channels);
*/
/*
    int x=32;
    int y=16;
    int share_x=x*2;
    int  share_y=y*2;
    dim3 block (x,y,1);
    dim3 grid ((ow-1+x)/x,(oh-1+y)/y,1);
    kernel_halfsizebyshare<<<grid,block,share_x*share_y*channels* sizeof(float)>>>(d_out,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image,d_out,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image,out,ow,oh,channels);
//缩小block
    x=32;
    y=8;
    share_x=x*2;
    share_y=y*2;
    float *d_out1=NULL;
    float *out_image1= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out1,bytes_out);
    dim3 block1 (x,y,1);
    dim3 grid1 ((ow-1+x)/x,(oh-1+y)/y,1);
    kernel_halfsizebyshare<<<grid1,block1,share_x*share_y*channels* sizeof(float)>>>(d_out1,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image1,d_out1,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image1,out,ow,oh,channels);
//缩小block
    x=32;
    y=4;
    share_x=x*2;
    share_y=y*2;
    float *d_out2=NULL;
    float *out_image2= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out2,bytes_out);
    dim3 block2 (x,y,1);
    dim3 grid2 ((ow-1+x)/x,(oh-1+y)/y,1);
    kernel_halfsizebyshare<<<grid2,block2,share_x*share_y*channels* sizeof(float)>>>(d_out2,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image2,d_out2,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image2,out,ow,oh,channels);
//缩小block
    x=32;
    y=32;
    share_x=x*2;
    share_y=y*2;
    float *d_out3=NULL;
    float *out_image3= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out3,bytes_out);
    dim3 block3 (x,y,1);
    dim3 grid3 ((ow-1+x)/x,(oh-1+y)/y,1);
    kernel_halfsizebyshare<<<grid3,block3,share_x*share_y*channels* sizeof(float)>>>(d_out3,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image3,d_out3,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image3,out,ow,oh,channels);
//放大block
    x=64;
    y=8;
    share_x=x*2;
    share_y=y*2;
    float *d_out4=NULL;
    float *out_image4= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out4,bytes_out);
    dim3 block4 (x,y,1);
    dim3 grid4 ((ow-1+x)/x,(oh-1+y)/y,1);
    kernel_halfsizebyshare<<<grid4,block4,share_x*share_y*channels* sizeof(float)>>>(d_out4,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image4,d_out4,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image4,out,ow,oh,channels);
*/
/*
    int x=32;
    int y=16;
    int share_x=x*4;
    int  share_y=y*2;
    dim3 block (x,y,1);
    dim3 grid ((ow-1+x*2)/(x*2),(oh-1+y)/y,1);
    kernel_halfsizebyshare1<<<grid,block,share_x*share_y*channels* sizeof(float)>>>(d_out,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image,d_out,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image,out,ow,oh,channels);
//缩小block
    x=32;
    y=8;
    share_x=x*4;
    share_y=y*2;
    float *d_out1=NULL;
    float *out_image1= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out1,bytes_out);
    dim3 block1 (x,y,1);
    dim3 grid1 ((ow-1+x*2)/(x*2),(oh-1+y)/y,1);
    kernel_halfsizebyshare1<<<grid1,block1,share_x*share_y*channels* sizeof(float)>>>(d_out1,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image1,d_out1,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image1,out,ow,oh,channels);
//缩小block
    x=32;
    y=4;
    share_x=x*4;
    share_y=y*2;
    float *d_out2=NULL;
    float *out_image2= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out2,bytes_out);
    dim3 block2 (x,y,1);
    dim3 grid2 ((ow-1+x*2)/(x*2),(oh-1+y)/y,1);
    kernel_halfsizebyshare1<<<grid2,block2,share_x*share_y*channels* sizeof(float)>>>(d_out2,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image2,d_out2,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image2,out,ow,oh,channels);
//缩小block
    //x=32;y=32;share_x=x*4;share_y=y*2;无法正常运行

//放大block
    x=64;
    y=8;
    share_x=x*4;
    share_y=y*2;
    float *d_out4=NULL;
    float *out_image4= (float *) malloc(bytes_out);
    cudaMalloc((void**)&d_out4,bytes_out);
    dim3 block4 (x,y,1);
    dim3 grid4 ((ow-1+x*2)/(x*2),(oh-1+y)/y,1);
    kernel_halfsizebyshare1<<<grid4,block4,share_x*share_y*channels* sizeof(float)>>>(d_out4,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image4,d_out4,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image4,out,ow,oh,channels);
*/

    /*cudaFree(d_out1);
    free(out_image1);
    cudaFree(d_out2);
    free(out_image2);
    cudaFree(d_out3);
    free(out_image3);
    cudaFree(d_out4);
    free(out_image4);*/

    cudaFree(d_in);
    cudaFree(d_out);
}

__global__ void kernel_halfsize_guass(float *out,float *in,int const ow,int const oh,int const iw,int const ih,int const ic,float const *w)
{

    int out_x=threadIdx.x+blockIdx.x*blockDim.x*ic;
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;
    int istride=iw*ic;

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
            int t=4;
            sum += in[row[0] + col[0]] * w[2];
            sum += in[row[0] + col[1]] * w[1];
            sum += in[row[0] + col[2]] * w[1];
            sum += in[row[0] + col[3]] * w[2];
            //if(out_idx==t)printf("gpu:%1.18f\n",sum);

            sum += in[row[1] + col[0]] * w[1];
            sum += in[row[1] + col[1]] * w[0];
            sum += in[row[1] + col[2]] * w[0];
            sum += in[row[1] + col[3]] * w[1];
            if(out_idx==t)printf("gpu:%1.18f\n",sum);

            /**/
            sum += in[row[2] + col[0]] * w[1];
            if(out_idx==t)printf("gpu:%1.18f\n",sum);
            sum += in[row[2] + col[1]] * w[0];
            if(out_idx==t)printf("gpu:%1.18f\n",sum);
            if(out_idx==t)
            {
                printf("高斯核索引:%d\t",row[2] + col[2]);
                printf("in:%1.18f\t",in[row[2] + col[2]]);
                printf("w:%1.18f\t",w[0]);
                printf("in*w:%1.18f\n",in[row[2] + col[2]] *w[0]);
            }
            sum += in[row[2] + col[2]] * w[0];
            if(out_idx==t)printf("gpu:%1.18f\n",sum);
            sum += in[row[2] + col[3]] * w[1];
            //if(out_idx==t)printf("gpu:%1.18f\n",sum);

            sum += in[row[3] + col[0]] * w[2];
            sum += in[row[3] + col[1]] * w[1];
            sum += in[row[3] + col[2]] * w[1];
            sum += in[row[3] + col[3]] * w[2];
            //if(out_idx==t)printf("gpu:%1.18f\n",sum);

            out[out_idx] = sum / (float)(4 * w[2] + 8 * w[1] + 4 * w[0]);

        }

    }
}


void halfsize_guassian_by_cuda(float * const out_image,float const  * const in_image, int const weight,int const height,int const channels,float sigma2,float const  * const out) {
/*
    int ow=(weight+1)>>1;
    int oh=(height+1)>>1;
    int const size_in=weight*height;
    int const size_out=ow*oh;
    int const bytes_in=size_in*channels* sizeof(float);
    int const bytes_out=size_out*channels* sizeof(float);

    float *d_in=NULL;
    float *d_out=NULL;
    float h_w[3];
    float *d_w=NULL;

    float3 *d_in1=NULL;
    cudaMalloc((void**)&d_in1,bytes_in);
    cudaMemcpy(d_in1,in_image,bytes_in,cudaMemcpyHostToDevice);

    h_w[0] = std::exp(-0.5 / (2.0f * sigma2));
    h_w[1] = std::exp(-2.5f / (2.0 * sigma2));
    h_w[2] = std::exp(-4.5f / (2.0f * sigma2));

    cudaMalloc((void**)&d_in,bytes_in);
    cudaMalloc((void**)&d_out,bytes_out);
    cudaMalloc((void**)&d_w,3* sizeof(float));
    cudaMemcpy(d_w,h_w,3* sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_in,in_image,bytes_in,cudaMemcpyHostToDevice);
    int const x=32;
    int const y=16;
    dim3 block (x,y,1);
    dim3 grid ((ow-1+x)/(x),(oh-1+y)/y,1);
    kernel_halfsize_guass<<<grid,block>>>(d_out,d_in,ow,oh,weight,height,channels,d_w);
    //compare(d_out,out,ow,oh,channels);
    float *d_channels_0=NULL;
    float *d_channels_1=NULL;
    float *d_channels_2=NULL;
    size_t bytes_channels=weight*height*sizeof(float);
    cudaMalloc((void**)&d_channels_0,bytes_channels);
    cudaMalloc((void**)&d_channels_1,bytes_channels);
    cudaMalloc((void**)&d_channels_2,bytes_channels);

    //split<<<grid,block,x*y*6*sizeof(float)>>>(d_channels_0,d_channels_1,d_channels_2,d_in,weight,height);
    //sp<<<grid,block>>>(d_in1,d_in);
    cudaFree(d_channels_0);
    cudaFree(d_channels_1);
    cudaFree(d_channels_2);


    cudaMemcpy(out_image,d_out,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image,out,ow,oh,channels);
    cudaFree(d_in);
    cudaFree(d_out);*/
    int ow = (weight + 1) >> 1;
    int oh = (height + 1) >> 1;
    int const size_in = weight * height;
    int const size_out = ow * oh;
    size_t const bytes_in = size_in * channels * sizeof(float);
    size_t const bytes_out = size_out * channels * sizeof(float);
    size_t const bytes_channels = size_in * sizeof(float);

    float *d_in = NULL;

    float *d_c_0 = NULL;
    float *d_c_1 = NULL;
    float *d_c_2 = NULL;

    cudaMalloc((void **) &d_in, bytes_in);
    cudaMalloc((void **) &d_c_0, bytes_channels);
    cudaMalloc((void **) &d_c_1, bytes_channels);
    cudaMalloc((void **) &d_c_2, bytes_channels);

    float *out_1, *out_2, *out_3;
    out_1 = (float *) malloc(bytes_channels);
    out_2 = (float *) malloc(bytes_channels);
    out_3 = (float *) malloc(bytes_channels);

    cudaMemcpy(d_in, in_image, bytes_in, cudaMemcpyHostToDevice);

    int x;
    int y;
    /*dim3 block0(32, 4, 1);
    dim3 grid0((weight +95) / 96, (height +3) / 4, 1);
    kernel_splitbyshare2<<<grid0, block0, 1152 * sizeof(float) >>> (d_c_0, d_c_1, d_c_2, d_in, weight, height);*/

    float *d_out=NULL;
    float h_w[3];
    float *d_w=NULL;

    h_w[0] = std::exp(-0.5 / (2.0f * sigma2));
    h_w[1] = std::exp(-2.5f / (2.0 * sigma2));
    h_w[2] = std::exp(-4.5f / (2.0f * sigma2));

    cudaMalloc((void**)&d_out,bytes_out);
    cudaMalloc((void**)&d_w,3* sizeof(float));
    cudaMemcpy(d_w,h_w,3* sizeof(float),cudaMemcpyHostToDevice);
    int const x1=32;
    int const y1=16;
    dim3 block (x1,y1,1);
    dim3 grid ((ow-1+x1)/(x1),(oh-1+y1)/y1,1);
    kernel_halfsize_guass<<<grid,block>>>(d_out,d_in,ow,oh,weight,height,channels,d_w);
    cudaMemcpy(out_image,d_out,bytes_out,cudaMemcpyDeviceToHost);
    compare(out_image,out,weight,height,channels);

    free(out_1);
    free(out_2);
    free(out_3);
    cudaFree(d_in);
    cudaFree(d_c_0);
    cudaFree(d_c_1);
    cudaFree(d_c_2);

}

  //
