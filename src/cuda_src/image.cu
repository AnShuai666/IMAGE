/*
 * @功能      image.hpp内TODO函数实现
 * @姓名      杨丰拓
 * @日期      2019-4-29
 * @时间      17:14
 * @邮箱
*/
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "cuda_include/common.cuh"
#include "cuda_include/sharedmem.cuh"
#include <cstdio>
template <typename T>
void gpu_cpu2zero1(T *cpu,T *gpu,size_t bytes)
{
    memset(cpu, 0, bytes);
    cudaMemset(gpu,0,bytes);
}


/*网格划分
for ( x = 32; x >8 ; x>>=1)
{
    for ( y = 32; y >2 ; y>>=1)
    {
    if (x * y < 128)continue;
    std::cout<<"block("<<x<<","<<y<<")"<<std::endl;
    }
}
*/
/******************************************************************************************/
///功能：填充图像
/*  函数名                           线程块大小       耗费时间
 *  kernel_fill_color	            702.651us	    [32,4,1]
 *  kernel_fill_color3	            705.469us	    [32,16,1]
 *  kernel_fill_color3_by_share	    400.097us	    [32,4,1]
 *  kernel_fill_color15_by_share	253.638us	    [32,4,1]**
 */
///核函数
/* 调用示例
 * dim3 block(x,y,1);
 * dim3 grid((wc-1+x)/x,(h-1+y)/y,1);
 * kernel_fill_color<T><<<grid,block>>>(d_out,d_color,wc,h,c);
 */
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

/* 调用示例
 * dim3 block(x,y,1);
 * dim3 grid((wc-1+x*3)/(x*3),(h-1+y)/y,1);
 * kernel_fill_color3<T><<<grid,block>>>(d_out,d_color,wc,h,c);
 */
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

/* 调用示例
 * dim3 block(x,y,1);
 * dim3 grid((wc-1+x*3)/(x*3),(h-1+y)/y,1);
 * kernel_fill_color3_by_share<T><<<grid,block,colorbytes>>>(d_out,d_color,wc,h,c);
 */
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

/* 调用示例
 * dim3 block(x,y,1);
 * dim3 grid((wc-1+x*15)/(x*15),(h-1+y)/y,1);
 * kernel_fill_color15_by_share<T><<<grid,block,colorbytes>>>(d_out,d_color,wc,h,c);
 */
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

///功能:添加颜色通道
/*  函数名                           线程块大小       耗费时间
 *  kernel_add_channels	            1.131ms	        [32,4,1]
 *  kernel_add_channels_stride	    507.197us	    [32,4,1]
 *  kernel_add_channels_stride2	    422.649us	    [32,4,1]**
 */
///核函数
/* 调用示例
 * dim3 block(x,y,1);
 * dim3 grid((w*c_add-1+x)/(x),(h-1+y)/y,1);
 * kernel_add_channels<T><<<grid,block>>>(d_out,d_in,w,h,c,num_channels,d_value,_front_back);
 */
template <typename T>
__global__ void kernel_add_channels(T *dst,T *src, int const w,int const h,int const c,int const num_channels,T * value,bool _front_back)
{
    int x=threadIdx.x+blockIdx.x*blockDim.x;//x坐标索引
    int y=threadIdx.y+blockIdx.y*blockDim.y;//y坐标索引
    int c_add=c+num_channels;
    int idx=y*w*c_add+x;//输出索引
    if(x<w*c_add&&y<h)
    {
        int channels=idx%c_add;
        int pixels=idx/c_add;
        if(_front_back)
        {
            if (channels < c) dst[idx] = src[pixels * c + channels];
            else dst[idx] = value[channels - c];
        }
        else
        {
            if (channels < num_channels) dst[idx] = value[channels];
            else dst[idx] = src[pixels * c + channels - num_channels];
        }
    }
}
/* 调用示例
 * dim3 block(x,y,1);
 * dim3 grid((w-1+x)/(x),(h-1+y)/y,1);
 * kernel_add_channels_stride<T><<<grid,block>>>(d_out,d_in,w,h,c,num_channels,d_value,_front_back);
 */
template <typename T>
__global__ void kernel_add_channels_stride(T *dst,T *src, int const w,int const h,int const c,int const num_channels,T * value,bool _front_back)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;//x坐标索引
    int y = threadIdx.y + blockIdx.y * blockDim.y;//y坐标索引
    int c_add=c+num_channels;
    int idx_out = y * w * c_add + x * c_add;//输出索引
    int idx_in = y * w * c + x * c;//输入索引
    if (x < w  && y < h)
    {
        if(_front_back)
        {
            for (int i = 0; i <c ; ++i) dst[idx_out+i]=src[idx_in+i];
            for (int j = 0; j <num_channels ; ++j)  dst[idx_out+c+j]=value[j];
        }
        else
        {
            for (int j = 0; j <num_channels ; ++j)  dst[idx_out+j]=value[j];
            for (int i = 0; i <c ; ++i) dst[idx_out+num_channels+i]=src[idx_in+i];
        }
    }
}
/* 调用示例
 * dim3 block(x,y,1);
 * dim3 grid((w-1+x*2)/(x*2),(h-1+y)/y,1);
 * kernel_add_channels_stride2<T><<<grid,block>>>(d_out,d_in,w,h,c,num_channels,d_value,_front_back);
 */
template <typename T>
__global__ void kernel_add_channels_stride2(T *dst,T *src, int const w,int const h,int const c,int const num_channels,T * value,bool _front_back)
{
    int x=threadIdx.x+blockIdx.x*blockDim.x*2;//x坐标索引
    int y=threadIdx.y+blockIdx.y*blockDim.y;//y坐标索引
    int c_add=c+num_channels;
    int idx_out=y*w*c_add+x*c_add;//输出索引
    int idx_in=y*w*c+x*c;//输入索引
    if (x < w  && y < h)
    {
        if(_front_back)
        {
            for (int i = 0; i <c ; ++i)
            {
                dst[idx_out+i]=src[idx_in+i];
                dst[idx_out+blockDim.x*c_add+i]=src[idx_in+blockDim.x*c+i];
            }
            for (int j = 0; j <num_channels ; ++j)
            {
                dst[idx_out+c+j]=value[j];
                dst[idx_out+blockDim.x*c_add+c+j]=value[j];
            }
        }
        else
        {
            for (int j = 0; j <num_channels ; ++j)
            {
                dst[idx_out+j]=value[j];
                dst[idx_out+blockDim.x*c_add+j]=value[j];
            }

            for (int i = 0; i <c ; ++i)
            {
                dst[idx_out+num_channels+i]=src[idx_in+i];
                dst[idx_out+blockDim.x*c_add+num_channels+i]=src[idx_in+blockDim.x*c+i];
            }
        }
    }
}

/******************************************************************************************/
///调用核函数实现加速功能
/******************************************************************************************/
///填充颜色通道函数
template <typename T>
int fill_color_cu(T *image,T *color,int const w,int const h,int const c,int const color_size)
{
    //bool flag= false;
    if(c!=color_size)
    {
        std::cerr<<"颜色通道不匹配"<<std::endl;
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
    int y=4;
    dim3 block(x,y,1);
    dim3 grid((wc-1+x*15)/(x*15),(h-1+y)/y,1);

    kernel_fill_color15_by_share<T><<<grid,block,colorbytes>>>(d_out,d_color,wc,h,c);

    //gpu2cpu
    cudaMemcpy(image,d_out,imagebytes,cudaMemcpyDeviceToHost);
    //compare1<T>(image,contrast,w*c,h,flag);
    //释放显存
    cudaFree(d_out);
    cudaFree(d_color);
    return 0;
}

///增加颜色通道函数
template <typename T>
int add_channels_cu(T *dst_image,T * src_image,int const w,int const h, int const c, int const num_channels,T * value,bool _front_back=true)
{
    if(num_channels<=0)
    {
        std::cerr<<"所添加的颜色通道个数小于1"<<std::endl;
        return 0;
    }
    int const wc =w*c;//输入图像实际宽度
    int const wc_add=w*(c+num_channels);//输出图像实际宽度
    //计算存储空间字节数
    size_t const bytes_value=num_channels* sizeof(T);
    size_t const bytes_src=wc*h* sizeof(T);
    size_t const bytes_dst=wc_add*h* sizeof(T);
    //声明显存指针
    T *d_in,*d_out,*d_value;
    //定义显存指针
    cudaMalloc((void**)&d_value,bytes_value);
    cudaMalloc((void**)&d_in,bytes_src);
    cudaMalloc((void**)&d_out,bytes_dst);
    //cpu2gpu
    cudaMemcpy(d_value,value,bytes_value,cudaMemcpyHostToDevice);
    cudaMemcpy(d_in,src_image,bytes_src,cudaMemcpyHostToDevice);
    //int c_add=c+num_channels;
    //网格划分
    int x=32;
    int y=4;
    dim3 block(x,y,1);
    dim3 grid((w-1+x*2)/(x*2),(h-1+y)/y,1);
    //核函数
    kernel_add_channels_stride2<T><<<grid,block>>>(d_out,d_in,w,h,c,num_channels,d_value,_front_back);
    //gpu2cpu
    cudaMemcpy(dst_image,d_out,bytes_dst,cudaMemcpyDeviceToHost);
    //compare1(dst_image,contrast,w*c,h, false);
    ///释放显存指针
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_value);
    return 0;
}






/******************************************************************************************/
///调用函数模板化
/******************************************************************************************/
///填充颜色通道函数
template <typename T>
int fill_color_by_cuda(T *image,T *color,int const w,int const h,int const c,int const color_size,T *contrast)
{
    fill_color_cu<T>(image,color,w,h,c, color_size);
    return 0;
}
template <>
int fill_color_by_cuda<char>(char *image,char *color,int const w,int const h,int const c,int const color_size,char *contrast)
{
    fill_color_cu<char>(image,color,w,h,c, color_size);
    //compare1<char>(image,contrast,w*c,h, false);
    return 0;
}
template <>
int fill_color_by_cuda<float>(float  *image,float *color,int const w,int const h,int const c,int const color_size,float *contrast)
{
    fill_color_cu<float>(image,color,w,h,c, color_size);
    //compare1<float>(image,contrast,w*c,h, true);
    return 0;
}

///增加颜色通道函数(后)
template <typename T>
int add_channels_by_cuda(T *dst_image,T  * src_image,int const w,int const h, int const c, int const num_channels,T * value,T *contrast)
{
    add_channels_cu(dst_image,src_image, w, h, c,num_channels,value);
    return  0;
}
template <>
int add_channels_by_cuda<char>(char *dst_image,char  * src_image,int const w,int const h, int const c, int const num_channels,char * value,char *contrast)
{
    add_channels_cu<char>(dst_image,src_image, w, h, c,num_channels,value);
    compare1(dst_image,contrast,w*c,h, false);
    return  0;
}
template <>
int add_channels_by_cuda<float>(float *dst_image,float  * src_image,int const w,int const h, int const c, int const num_channels,float * value,float *contrast)
{
    add_channels_cu<float>(dst_image,src_image, w, h, c,num_channels,value);
    //compare1(dst_image,contrast,w*c,h, true);
    return  0;
}

///增加颜色通道函数(前/后)
template <typename T>
int add_channels_front_by_cuda(T *dst_image,T  * src_image,int const w,int const h, int const c, vector<T> value,bool _front_back,T *contrast)
{
    add_channels_cu(dst_image,src_image, w, h, c,(int)value.size(),&value.at(0),_front_back);
    //compare1(dst_image,contrast,w*c,h, false);
    return 0;
}
template <>
int add_channels_front_by_cuda<char>(char *dst_image,char  * src_image,int const w,int const h, int const c, vector<char> value,bool _front_back,char *contrast)
{
    add_channels_cu<char>(dst_image,src_image, w, h, c,(int)value.size(),&value.at(0),_front_back);
    compare1(dst_image,contrast,w*c,h, false);
    return 0;
}
template <>
int add_channels_front_by_cuda<float>(float *dst_image,float  * src_image,int const w,int const h, int const c, vector<float> value,bool _front_back,float *contrast)
{
    add_channels_cu<float>(dst_image,src_image, w, h, c,(int)value.size(),&value.at(0),_front_back);
    compare1(dst_image,contrast,w*c,h, false);
    return 0;
}