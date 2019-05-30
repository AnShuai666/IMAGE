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
#include "cuda_include/sharemem.cuh"
#include <cstdio>
template <typename T>
void gpu_cpu2zero1(T *cpu,T *gpu,size_t bytes)
{
    memset(cpu, 0, bytes);
    cudaMemset(gpu,0,bytes);
}


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
 * dim3 grid((kFact_width-1+x)/x,(h-1+y)/y,1);
 * kernel_fill_color<T><<<grid,block>>>(d_out,d_color,kFact_width,h,c);
 */
template <typename T>
__global__ void kernelFillColor(T * p_image, T *p_color,int const kFact_width,int const kHeight,int const kChannels)
{
    int x=threadIdx.x+blockIdx.x*blockDim.x;
    int y=threadIdx.y+blockIdx.y*blockDim.y;
    int idx=y*kFact_width+x;
    //越界判断
    if(x<kFact_width&&y<kHeight)
    {
        int channels=idx%kChannels;
        p_image[idx]=p_color[channels];
    }
}
/* 调用示例
 * dim3 block(x,y,1);
 * dim3 grid((kFact_width-1+x*3)/(x*3),(h-1+y)/y,1);
 * kernel_fill_color3<T><<<grid,block>>>(d_out,d_color,kFact_width,h,c);
 */
template <typename T>
__global__ void kernelFillColor3(T * p_image, T *p_color,int const kFact_width,int const kHeight,int const kChannels)
{
    int x=threadIdx.x+blockIdx.x*blockDim.x*3;
    int y=threadIdx.y+blockIdx.y*blockDim.y;
    int idx=y*kFact_width+x;

    T local_color[4];
    for(int i=0;i<kChannels;i++)
    {
        local_color[i]=p_color[i];
    }
    //越界判断
    if((x+blockDim.x*2)<kFact_width&&y<kHeight)
    {
        int channels=idx%kChannels;
        p_image[idx]=local_color[channels];

        idx+=blockDim.x;
        channels=idx%kChannels;
        p_image[idx]=local_color[channels];

        idx+=blockDim.x;
        channels=idx%kChannels;
        p_image[idx]=local_color[channels];
    }
}
/* 调用示例
 * dim3 block(x,y,1);
 * dim3 grid((kFact_width-1+x*3)/(x*3),(h-1+y)/y,1);
 * kernel_fill_color3_by_share<T><<<grid,block,colorbytes>>>(d_out,d_color,kFact_width,h,c);
 */
template <typename T>
__global__ void kernelFillColorByShare3(T * p_image, T *p_color,int const kFact_width,int const kHeight,int const kChannels)
{
    sharedMemory<T> smem;
    T* data = smem.getPointer();
    int x=threadIdx.x+blockIdx.x*blockDim.x*3;
    int y=threadIdx.y+blockIdx.y*blockDim.y;
    int idx=y*kFact_width+x;
    int sidx=threadIdx.y*blockDim.x+threadIdx.x;
    if(sidx<kChannels)data[sidx]=p_color[sidx];
    __syncthreads();
    //越界判断
    if((x+blockDim.x*2)<kFact_width&&y<kHeight)
    {
        int channels;
        for(int k=0;k<3;k++)
        {
            channels=idx%kChannels;
            p_image[idx]=data[channels];
            idx+=blockDim.x;
        }
    }
}
/* 调用示例
 * dim3 block(x,y,1);
 * dim3 grid((kFact_width-1+x*15)/(x*15),(h-1+y)/y,1);
 * kernel_fill_color15_by_share<T><<<grid,block,colorbytes>>>(d_out,d_color,kFact_width,h,c);
 */
template <typename T>
__global__ void kernelFillColorByShare15(T * p_image, T *p_color,int const kFact_width,int const kHeight,int const kChannels)
{
    sharedMemory<T> smem;
    T* data = smem.p_getPointer();
    int x=threadIdx.x+blockIdx.x*blockDim.x*15;
    int y=threadIdx.y+blockIdx.y*blockDim.y;
    int idx=y*kFact_width+x;
    int sidx=threadIdx.y*blockDim.x+threadIdx.x;
    if(sidx<kChannels)data[sidx]=p_color[sidx];
    __syncthreads();
    //越界判断

    if(x<kFact_width&&y<kHeight)
    {
        int channels;
        for(int k=0;k<15;k++)
        {
            channels=idx%kChannels;
            p_image[idx]=data[channels];
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
 * kernel_add_channels<T><<<grid,block>>>(d_out,d_in,w,h,c,num_channels,value);
 */
template <typename T>
__global__ void kernelAddChannel(T *p_dst,T *p_src, int const kWidth,int const kHeight,int const kChannels,int const kNum_channels,T value)
{
    int x=threadIdx.x+blockIdx.x*blockDim.x;//x坐标索引
    int y=threadIdx.y+blockIdx.y*blockDim.y;//y坐标索引
    int c_add=kChannels+kNum_channels;
    int idx=y*kWidth*c_add+x;//输出索引
    if(x<kWidth*c_add&&y<kHeight)
    {
        int channels=idx%c_add;
        int pixels=idx/c_add;
        if (channels < kChannels) p_dst[idx] = p_src[pixels * kChannels + channels];
        else p_dst[idx] = value;
    }
}
/* 调用示例
 * dim3 block(x,y,1);
 * dim3 grid((w-1+x)/(x),(h-1+y)/y,1);
 * kernel_add_channels_stride<T><<<grid,block>>>(d_out,d_in,w,h,c,num_channels,value);
 */
template <typename T>
__global__ void kernelAddChannelStride(T *p_dst,T *p_src, int const kWidth,int const kHeight,int const kChannels,int const kNum_channels,T value)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;//x坐标索引
    int y = threadIdx.y + blockIdx.y * blockDim.y;//y坐标索引
    int c_add=kChannels+kNum_channels;
    int idx_out = y * kWidth * c_add + x * c_add;//输出索引
    int idx_in = y * kWidth * kChannels + x * kChannels;//输入索引
    if (x < kWidth  && y < kHeight)
    {
        for (int i = 0; i <kChannels ; ++i) p_dst[idx_out+i]=p_src[idx_in+i];
        for (int j = 0; j <kNum_channels ; ++j)  p_dst[idx_out+kChannels+j]=value;
    }
}
/* 调用示例
 * dim3 block(x,y,1);
 * dim3 grid((w-1+x*2)/(x*2),(h-1+y)/y,1);
 * kernel_add_channels_stride2<T><<<grid,block>>>(d_out,d_in,w,h,c,num_channels,value);
 */
template <typename T>
__global__ void kernelAddChannelStride2(T *p_dst,T *p_src, int const kWidth,int const kHeight,int const kChannels,int const kNum_channels,T value)
{
    int x=threadIdx.x+blockIdx.x*blockDim.x*2;//x坐标索引
    int y=threadIdx.y+blockIdx.y*blockDim.y;//y坐标索引
    int c_add=kChannels+kNum_channels;
    int idx_out=y*kWidth*c_add+x*c_add;//输出索引
    int idx_in=y*kWidth*kChannels+x*kChannels;//输入索引
    if (x < kWidth  && y < kHeight)
    {
        for (int i = 0; i <kChannels ; ++i)
        {
            p_dst[idx_out+i]=p_src[idx_in+i];
            p_dst[idx_out+blockDim.x*c_add+i]=p_src[idx_in+blockDim.x*kChannels+i];
        }
        for (int j = 0; j <kNum_channels ; ++j) {
            p_dst[idx_out + kChannels + j] = value;
            p_dst[idx_out + blockDim.x * c_add + kChannels + j] = value;
        }
    }
}

///功能:添加颜色通道(多颜色数据)
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
__global__ void kernelAddChannels(T *p_dst,T *p_src, int const kWidth,int const kHeight,int const kChannels,int const kNum_channels,T * p_value,bool _front_back)
{
    int x=threadIdx.x+blockIdx.x*blockDim.x;//x坐标索引
    int y=threadIdx.y+blockIdx.y*blockDim.y;//y坐标索引
    int c_add=kChannels+kNum_channels;
    int idx=y*kWidth*c_add+x;//输出索引
    if(x<kWidth*c_add&&y<kHeight)
    {
        int channels=idx%c_add;
        int pixels=idx/c_add;
        if(_front_back)
        {
            if (channels < kChannels) p_dst[idx] = p_src[pixels * kChannels + channels];
            else p_dst[idx] = p_value[channels - kChannels];
        }
        else
        {
            if (channels < kNum_channels) p_dst[idx] = p_value[channels];
            else p_dst[idx] = p_src[pixels * kChannels + channels - kNum_channels];
        }
    }
}
/* 调用示例
 * dim3 block(x,y,1);
 * dim3 grid((w-1+x)/(x),(h-1+y)/y,1);
 * kernel_add_channels_stride<T><<<grid,block>>>(d_out,d_in,w,h,c,num_channels,d_value,_front_back);
 */
template <typename T>
__global__ void kernelAddChannelsStride(T *p_dst,T *p_src, int const kWidth,int const kHeight,int const kChannels,int const kNum_channels,T * p_value,bool _front_back)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;//x坐标索引
    int y = threadIdx.y + blockIdx.y * blockDim.y;//y坐标索引
    int c_add=kChannels+kNum_channels;
    int idx_out = y * kWidth * c_add + x * c_add;//输出索引
    int idx_in = y * kWidth * kChannels + x * kChannels;//输入索引
    if (x < kWidth  && y < kHeight)
    {
        if(_front_back)
        {
            for (int i = 0; i <kChannels ; ++i) p_dst[idx_out+i]=p_src[idx_in+i];
            for (int j = 0; j <kNum_channels ; ++j)  p_dst[idx_out+kChannels+j]=p_value[j];
        }
        else
        {
            for (int j = 0; j <kNum_channels ; ++j)  p_dst[idx_out+j]=p_value[j];
            for (int i = 0; i <kChannels ; ++i) p_dst[idx_out+kNum_channels+i]=p_src[idx_in+i];
        }
    }
}
/* 调用示例
 * dim3 block(x,y,1);
 * dim3 grid((w-1+x*2)/(x*2),(h-1+y)/y,1);
 * kernel_add_channels_stride2<T><<<grid,block>>>(d_out,d_in,w,h,c,num_channels,d_value,_front_back);
 */
template <typename T>
__global__ void kernelAddChannelsStride2(T *p_dst,T *p_src, int const kWidth,int const kHeight,int const kChannels,int const kNum_channels,T * p_value,bool _front_back)
{
    int x=threadIdx.x+blockIdx.x*blockDim.x*2;//x坐标索引
    int y=threadIdx.y+blockIdx.y*blockDim.y;//y坐标索引
    int c_add=kChannels+kNum_channels;
    int idx_out=y*kWidth*c_add+x*c_add;//输出索引
    int idx_in=y*kWidth*kChannels+x*kChannels;//输入索引
    if (x < kWidth  && y < kHeight)
    {
        if(_front_back)
        {
            for (int i = 0; i <kChannels ; ++i)
            {
                p_dst[idx_out+i]=p_src[idx_in+i];
                p_dst[idx_out+blockDim.x*c_add+i]=p_src[idx_in+blockDim.x*kChannels+i];
            }
            for (int j = 0; j <kNum_channels ; ++j)
            {
                p_dst[idx_out+kChannels+j]=p_value[j];
                p_dst[idx_out+blockDim.x*c_add+kChannels+j]=p_value[j];
            }
        }
        else
        {
            for (int j = 0; j <kNum_channels ; ++j)
            {
                p_dst[idx_out+j]=p_value[j];
                p_dst[idx_out+blockDim.x*c_add+j]=p_value[j];
            }
            for (int i = 0; i <kChannels ; ++i)
            {
                p_dst[idx_out+kNum_channels+i]=p_src[idx_in+i];
                p_dst[idx_out+blockDim.x*c_add+kNum_channels+i]=p_src[idx_in+blockDim.x*kChannels+i];
            }
        }
    }
}

///功能:交换颜色通道
/*  函数名                           线程块大小       耗费时间
 * kernel_swap_channels	            283.847us	   [32,4,1]**
 * kernel_swap_channels2	        293.352us	   [32,4,1]
 */
///核函数
/* 调用示例
 * dim3 block(x,y,1);
 * dim3 grid((w-1+x)/(x),(h-1+y)/y,1);
 * kernel_swap_channels<T><<<grid,block>>>(d_in,w,h,c,swap_c1,swap_c2);
 */
template <typename T>
__global__ void kernelSwapChannels(T *p_src,int const kWidth,int const kHeight,int const kChannels, int const kSwap_c1,int const kSwap_c2)
{
    int const x=threadIdx.x+blockDim.x*blockIdx.x;
    int const y=threadIdx.y+blockDim.y*blockIdx.y;
    int const idx=y*kWidth+x;
    if(x<kWidth&&y<kHeight)
    {
        T a,b;
        a=p_src[idx*kChannels+kSwap_c1];
        b=p_src[idx*kChannels+kSwap_c2];
        p_src[idx*kChannels+kSwap_c1]=b;
        p_src[idx*kChannels+kSwap_c2]=a;
    }
}
/* 调用示例
 * dim3 block(x,y,1);
 * dim3 grid((w-1+x)/(x),(h-1+y*2)/(y*2),1);
 * kernel_swap_channels2<T><<<grid,block>>>(d_in,w,h,c,swap_c1,swap_c2);
 */
template <typename T>
__global__ void kernelSwapChannels2(T *p_src,int const kWidth,int const kHeight,int const kChannels, int const kSwap_c1,int const kSwap_c2)
{
    int  x=threadIdx.x+blockIdx.x*blockDim.x;
    int  y=threadIdx.y+blockIdx.y*blockDim.y*2;
    for(int i=0;i<2;i++)
    {
        int idx=(y+blockDim.y*i)*kWidth*kChannels+x*kChannels;
        if(x<kWidth&&(y+blockDim.y*i)<kHeight)
        {
            T a,b;
            a=p_src[idx*kChannels+kSwap_c1];
            b=p_src[idx*kChannels+kSwap_c2];
            p_src[idx*kChannels+kSwap_c1]=b;
            p_src[idx*kChannels+kSwap_c2]=a;
        }
    }
}

///功能:复制颜色通道
/*  函数名                           线程块大小       耗费时间
 * kernel_copy_channels	            286.692us	    [32,4,1]**
 */
///核函数
/* 调用示例
 * dim3 block(x,y,1);
 * dim3 grid((w-1+x)/(x),(h-1+y)/y,1);
 * kernel_copy_channels<T><<<grid,block>>>(d_in,w,h,c,copy_c,paste_c);
 */
template <typename  T>
__global__ void kernelCopyChannels(T *p_image,int const kWidth,int const kHeight,int const kChannels,int const kCopy_c,int const kPaste_c)
{
    int x=blockDim.x*blockIdx.x+threadIdx.x;
    int y=blockDim.y*blockIdx.y+threadIdx.y;
    if(x<kWidth&&y<kHeight)
    {
        int idx=y*kWidth*kChannels+x*kChannels;
        T value=p_image[idx+kChannels];
        p_image[idx+kPaste_c]=value;
    }
}

///功能:删除颜色通道
/*  函数名                           线程块大小       耗费时间
 * kernel_delete_channel	        468.206us	   [32,4,1]
 * kernel_delete_channel2	        322.506us	   [32,2,1]**
 * kernel_delete_channel3	        334.987us	   [32,2,1]
 */
///核函数
/* 调用示例
 * dim3 block(x,y,1);
 * dim3 grid((src_w*dst_c-1+x*5)/(x*5),(src_h-1+y)/y,1);
 * kernel_delete_channel<T><<<grid,block>>>(d_out,d_in,src_w,src_h,src_c,dst_c,del_c);
 */
template <typename T>
__global__ void kernelDeleteChannel(T *p_dst,T *p_src,int const kWidth,int const kHeight,int const kChannels,int const kDst_c,int const kDel_c)
{
    int x=threadIdx.x+blockIdx.x*blockDim.x*5;
    int y=threadIdx.y+blockIdx.y*blockDim.y;
    for (int i = 0; i <5 ; ++i) {

        if(x<kWidth*kDst_c&&y<kHeight)
        {
            int idx_out=y*kWidth*kDst_c+x;
            int channel=idx_out%kDst_c;
            int pixel=idx_out/kDst_c;
            int idx_in=pixel*kChannels+channel;
            T value;
            if(channel>=kDel_c)idx_in+=1;
            value=p_src[idx_in];
            p_dst[idx_out]=value;
        }
        x+=blockDim.x;
    }
}
/* 调用示例
 * dim3 block(x,y,1);
 * dim3 grid((src_w-1+x*2)/(x*2),(src_h-1+y)/y,1);
 * kernel_delete_channel2<T><<<grid,block>>>(d_out,d_in,src_w,src_h,src_c,dst_c,del_c);
 */
template <typename T>
__global__ void kernelDeleteChannel2(T *p_dst,T *p_src,int const kWidth,int const kHeight,int const kChannels,int const kDst_c,int const kDel_c)
{
    int x=threadIdx.x+blockIdx.x*blockDim.x*2;
    int y=threadIdx.y+blockIdx.y*blockDim.y;
    if(x<kWidth&&y<kHeight)
    {
        int pixel=y*kWidth+x;
        int pixel1=y*kWidth+x+blockDim.x;
        T value;
        int j=0;
        for (int i = 0; i <kChannels ; ++i)
        {
            if(i!=kDel_c)
            {
                value=p_src[pixel*kChannels+i];
                p_dst[pixel*kDst_c+j]=value;
                value=p_src[pixel1*kChannels+i];
                p_dst[pixel1*kDst_c+j]=value;
                j++;
            }
        }
    }
}
/* 调用示例
 * dim3 block(x,y,1);
 * dim3 grid((src_w-1+x*3)/(x*3),(src_h-1+y)/y,1);
 * kernel_delete_channel3<T><<<grid,block>>>(d_out,d_in,src_w,src_h,src_c,dst_c,del_c);
 */
template <typename T>
__global__ void kernelDeleteChannel3(T *p_dst,T *p_src,int const kWidth,int const kHeight,int const kChannels,int const kDst_c,int const kDel_c)
{
    int x=threadIdx.x+blockIdx.x*blockDim.x*3;
    int y=threadIdx.y+blockIdx.y*blockDim.y;
    if(x<kWidth&&y<kHeight)
    {
        int pixel=y*kWidth+x;
        int pixel2=pixel+blockDim.x;
        int pixel3=pixel2+blockDim.x;
        T value;
        int j=0;
        for (int i = 0; i <kChannels ; ++i)
        {
            if(i!=kDel_c)
            {
                value=p_src[pixel*kChannels+i];
                p_dst[pixel*kDst_c+j]=value;
                value=p_src[pixel2*kChannels+i];
                p_dst[pixel2*kDst_c+j]=value;
                value=p_src[pixel3*kChannels+i];
                p_dst[pixel3*kDst_c+j]=value;
                j++;
            }
        }
    }
}




/******************************************************************************************/
///调用核函数实现加速功能
/******************************************************************************************/

///填充颜色通道函数
template <typename T>
int fillColorCu(T *p_image,T *p_color,int const kWidth,int const kHeight,int const kChannels,int const kColor_size)
{
    //bool flag= false;
    if(kChannels!=kColor_size)
    {
        std::cerr<<"颜色通道不匹配"<<std::endl;
        return 0;
    }
    int wc=kWidth*kChannels;
    //定义显存指针
    T *p_d_out=NULL;
    T *p_d_color=NULL;
    //计算显存所需字节数
    size_t const kImagebytes=kWidth*kHeight*kChannels*sizeof(T);
    int const kColorbytes=kColor_size* sizeof(T);
    //分配显存
    cudaMalloc((void**)&p_d_out  ,kImagebytes);
    cudaMalloc((void**)&p_d_color,kColorbytes);
    //cpu2gpu
    cudaMemcpy(p_d_color,p_color,kColorbytes,cudaMemcpyHostToDevice);

    //线程网格划分
    int x=32;
    int y=4;
    dim3 block(x,y,1);
    dim3 grid((wc-1+x*15)/(x*15),(kHeight-1+y)/y,1);

    kernelFillColorByShare15<T><<<grid,block,kColorbytes>>>(p_d_out,p_d_color,wc,kHeight,kChannels);

    //gpu2cpu
    cudaMemcpy(p_image,p_d_out,kImagebytes,cudaMemcpyDeviceToHost);
    //释放显存
    cudaFree(p_d_out);
    cudaFree(p_d_color);
    return 0;
}
///增加颜色通道函数(单通道多数据)
template <typename T>
int addChannelsCu(T *p_dst_image,T * p_src_image,int const kWidth,int const kHeight,int const kChannels, int const kNum_channels,T  value)
{
    if(kNum_channels<=0)
    {
        std::cerr<<"所添加的颜色通道个数小于1"<<std::endl;
        return 0;
    }
    int const wc =kWidth*kChannels;//输入图像实际宽度
    int const wc_add=kWidth*(kChannels+kNum_channels);//输出图像实际宽度
    //计算存储空间字节数
    size_t const kBytes_src=wc*kHeight* sizeof(T);
    size_t const kBytes_dst=wc_add*kHeight* sizeof(T);
    //声明显存指针
    T *p_d_in=NULL,*p_d_out=NULL;
    //定义显存指针
    cudaMalloc((void**)&p_d_in ,kBytes_src);
    cudaMalloc((void**)&p_d_out,kBytes_dst);
    //cpu2gpu
    cudaMemcpy(p_d_in,p_src_image,kBytes_src,cudaMemcpyHostToDevice);
    //网格划分
    int x=32;
    int y=4;
    dim3 block(x,y,1);
    dim3 grid((kWidth-1+x*2)/(x*2),(kHeight-1+y)/y,1);
    //核函数
    kernelAddChannelStride2<T><<<grid,block>>>(p_d_out,p_d_in,kWidth,kHeight,kChannels,kNum_channels,value);
    //gpu2cpu
    cudaMemcpy(p_dst_image,p_d_out,kBytes_dst,cudaMemcpyDeviceToHost);
    ///释放显存指针
    cudaFree(p_d_in);
    cudaFree(p_d_out);
    return 0;
}
///增加颜色通道函数(多通道多数据)
template <typename T>
int addChannelsCu(T *p_dst_image,T * p_src_image,int const kWidth,int const kHeight,int const kChannels, int const kNum_channels,T * p_value,bool _front_back=true)
{
    if(kNum_channels<=0)
    {
        std::cerr<<"所添加的颜色通道个数小于1"<<std::endl;
        return 0;
    }
    int const wc =kWidth*kChannels;//输入图像实际宽度
    int const wc_add=kWidth*(kChannels+kNum_channels);//输出图像实际宽度
    //计算存储空间字节数
    size_t const kBytes_value=kNum_channels* sizeof(T);
    size_t const kBytes_src  =wc*kHeight* sizeof(T);
    size_t const kBytes_dst  =wc_add*kHeight* sizeof(T);
    //声明显存指针
    T *p_d_in=NULL,*p_d_out=NULL,*p_d_value=NULL;
    //定义显存指针
    cudaMalloc((void**)&p_d_value,kBytes_value);
    cudaMalloc((void**)&p_d_in,kBytes_src);
    cudaMalloc((void**)&p_d_out,kBytes_dst);
    //cpu2gpu
    cudaMemcpy(p_d_value,p_value,kBytes_value,cudaMemcpyHostToDevice);
    cudaMemcpy(p_d_in,p_src_image,kBytes_src,cudaMemcpyHostToDevice);
    //网格划分
    int x=32;
    int y=4;
    dim3 block(x,y,1);
    dim3 grid((kWidth-1+x*2)/(x*2),(kHeight-1+y)/y,1);
    //核函数
    kernelAddChannelsStride2<T><<<grid,block>>>(p_d_out,p_d_in,kWidth,kHeight,kChannels,kNum_channels,p_d_value,_front_back);
    //gpu2cpu
    cudaMemcpy(p_dst_image,p_d_out,kBytes_dst,cudaMemcpyDeviceToHost);
    ///释放显存指针
    cudaFree(p_d_in);
    cudaFree(p_d_out);
    cudaFree(p_d_value);
    return 0;
}
///交换颜色通道函数
template <typename T>
int swapChannelsByCu(T *p_src,int const kWidth,int const kHeight,int const kChannels,int const kSwap_c1,int kSwap_c2)
{
    if(kSwap_c1==kSwap_c2)return 0;
    if(kSwap_c1<0||kSwap_c1>=kChannels||kSwap_c2<0||kSwap_c2>=kChannels)
    {
        std::cerr<<"swapChannelsByCuda函数所要交换的颜色通道不合适!!"<<std::endl;
        return 1;
    }
    //计算字节数
    size_t const kBytes=kWidth*kHeight*kChannels* sizeof(T);
    //声明显存指针
    T *p_d_in=NULL;
    //定义显存指针
    cudaMalloc((void**)&p_d_in,kBytes);
    //cpu2gpu
    cudaMemcpy(p_d_in,p_src,kBytes,cudaMemcpyHostToDevice);
    //网格划分
    int x=32;
    int y=4;
    dim3 block(x,y,1);
    dim3 grid((kWidth-1+x)/(x),(kHeight-1+y)/y,1);
    //核函数
    kernelSwapChannels<T><<<grid,block>>>(p_d_in,kWidth,kHeight,kChannels,kSwap_c1,kSwap_c2);
    //gpu2cpu
    cudaMemcpy(p_src,p_d_in,kBytes,cudaMemcpyDeviceToHost);
    //释放显存指针
    cudaFree(p_d_in);
    return 0;
}
///复制颜色通道
template <typename  T>
int copyChannelsByCu(T *p_image,int const kWidth,int const kHeight,int const kChannels,int const kCopy_c,int const kPaste_c)
{
    if(kCopy_c>=kChannels||kPaste_c>=kChannels)
    {
        std::cerr<<"输入通道数超过图像的最大通道数"<<std::endl;
        return 1;
    }
    if(kCopy_c==kPaste_c)return 0;
    if(kPaste_c<0)
    {
        //TODO:向后添加一个全为零的颜色通道
    }
    //计算字节数
    size_t const kBytes=kWidth*kHeight*kChannels* sizeof(T);
    //声明显存指针
    T *p_d_in=NULL;
    //定义显存指针
    cudaMalloc(&p_d_in,kBytes);
    //cpu2gpu
    cudaMemcpy(p_d_in,p_image,kBytes,cudaMemcpyHostToDevice);
    //网格划分
    int x=32;
    int y=4;
    dim3 block(x,y,1);
    dim3 grid((kWidth-1+x)/(x),(kHeight-1+y)/y,1);
    //核函数
    kernelCopyChannels<T><<<grid,block>>>(p_d_in,kWidth,kHeight,kChannels,kCopy_c,kPaste_c);
    //gpu2cpu
    cudaMemcpy(p_image,p_d_in,kBytes,cudaMemcpyDeviceToHost);
    //释放显存指针
    cudaFree(p_d_in);
    return 0;
}
///删除颜色通道
template <typename T>
int deleteChannelByCu(T *p_dstImage,T *p_srcImage,int const kSrc_width,int const kSrc_height,int const kSrc_channels,int const kDel_channel)
{
    if(kDel_channel<0||kDel_channel>=kSrc_channels)return 0;
    int dst_c=kSrc_channels-1;//输出通道数
    //计算所需存储的字节数
    size_t const kBytes_in=kSrc_width*kSrc_height*kSrc_channels* sizeof(T);
    size_t const kBytes_out=kSrc_width*kSrc_height*dst_c* sizeof(T);
    //声明显存指针
    T *p_d_in=NULL;
    T *p_d_out=NULL;
    //定义显存指针
    cudaMalloc(&p_d_in ,kBytes_in);
    cudaMalloc(&p_d_out,kBytes_out);
    //cpu2gpu
    cudaMemcpy(p_d_in,p_srcImage,kBytes_in,cudaMemcpyHostToDevice);
    //网格划分
    int x=32;
    int y=2;
    dim3 block(x,y,1);
    dim3 grid((kSrc_width-1+x*2)/(x*2),(kSrc_height-1+y)/y,1);
    //核函数
    kernelDeleteChannel2<T><<<grid,block>>>(p_d_out,p_d_in,kSrc_width,kSrc_height,kSrc_channels,dst_c,kDel_channel);
    //gpu2cpu
    cudaMemcpy(p_dstImage,p_d_out,kBytes_out,cudaMemcpyDeviceToHost);
    //释放显存指针
    cudaFree(p_d_in);
    cudaFree(p_d_out);
    return 0;
}




/******************************************************************************************/
///调用函数模板化
/******************************************************************************************/
///填充颜色通道函数
template <typename T>
int fillColorByCuda(T *image,T *color,int const w,int const h,int const c,int const color_size)
{
    fillColorCu<T>(image,color,w,h,c, color_size);
    return 0;
}
template <>
int fillColorByCuda(char *image,char *color,int const w,int const h,int const c,int const color_size)
{
    fillColorCu<char>(image,color,w,h,c, color_size);
    //compare1<char>(image,contrast,w*c,h, false);
    return 0;
}
template <>
int fillColorByCuda(float  *image,float *color,int const w,int const h,int const c,int const color_size)
{
    fillColorCu<float>(image,color,w,h,c, color_size);
    //compare1<float>(image,contrast,w*c,h, true);
    return 0;
}

///增加颜色通道函数(后)
template <typename T>
int addChannelsByCuda(T *dst_image,T  * src_image,int const w,int const h, int const c, int const num_channels,T  value)
{
    addChannelsCu(dst_image,src_image, w, h, c,num_channels,value);
    return  0;
}
template <>
int addChannelsByCuda(char *dst_image,char  * src_image,int const w,int const h, int const c, int const num_channels,char  value)
{
    addChannelsCu<char>(dst_image,src_image, w, h, c,num_channels,value);
    return  0;
}
template <>
int addChannelsByCuda(float *dst_image,float  * src_image,int const w,int const h, int const c, int const num_channels,float  value)
{
    addChannelsCu<float>(dst_image,src_image, w, h, c,num_channels,value);
    //compare1<float>(dst_image,contrast,w*c,h, true);
    return  0;
}

///增加颜色通道函数(前/后)
template <typename T>
int addChannelsFrontByCuda(T *dst_image,T  * src_image,int const w,int const h, int const c, vector<T> value,bool _front_back)
{
    addChannelsCu(dst_image,src_image, w, h, c,(int)value.size(),&value.at(0),_front_back);
    //compare1(dst_image,contrast,w*c,h, false);
    return 0;
}
template <>
int addChannelsFrontByCuda(char *dst_image,char  * src_image,int const w,int const h, int const c, vector<char> value,bool _front_back)
{
    addChannelsCu<char>(dst_image,src_image, w, h, c,(int)value.size(),&value.at(0),_front_back);
    return 0;
}
template <>
int addChannelsFrontByCuda(float *dst_image,float  * src_image,int const w,int const h, int const c, vector<float> value,bool _front_back)
{
    addChannelsCu<float>(dst_image,src_image, w, h, c,(int)value.size(),&value.at(0),_front_back);
    return 0;
}

///交换颜色通道
template <typename T>
int swapChannelsByCuda(T *src,int const w,int const h,int c,int const swap_c1,int swap_c2)
{

    return 0;
}
template <>
int swapChannelsByCuda(char *src,int const w,int const h,int c,int const swap_c1,int swap_c2)
{
    swapChannelsByCu<char>(src,w,h,c,swap_c1,swap_c2);
    //compare1<char>(src,contrast,w*c,h, false);
    return 0;
}
template <>
int swapChannelsByCuda(float *src,int const w,int const h,int c,int const swap_c1,int swap_c2)
{
    swapChannelsByCu<float>(src,w,h,c,swap_c1,swap_c2);
    //compare1<float>(src,contrast,w*c,h, true);
    return 0;
}

///复制颜色通道
template <typename T>
int copyChannelsByCuda(T *image,int const w,int const h,int const c,int const copy_c,int const paste_c)
{
    return 0;
}
template <>
int copyChannelsByCuda(char *image,int const w,int const h,int const c,int const copy_c,int const paste_c)
{
    copyChannelsByCu<char>(image,w,h,c,copy_c,paste_c);
    return 0;
}
template <>
int copyChannelsByCuda(float *image,int const w,int const h,int const c,int const copy_c,int const paste_c)
{
    copyChannelsByCu<float>(image,w,h,c,copy_c,paste_c);
    return 0;
}

///删除颜色通道
template <typename T>
int deleteChannelByCuda(T *dstImage,T *srcImage,int const src_w,int const src_h,int const src_c,int const del_c)
{
    return 0;
}
template <>
int deleteChannelByCuda(char *dstImage,char *srcImage,int const src_w,int const src_h,int const src_c,int const del_c)
{
    deleteChannelByCu<char>(dstImage,srcImage,src_w,src_h,src_c,del_c);
    return 0;
}
template <>
int deleteChannelByCuda(float *dstImage,float *srcImage,int const src_w,int const src_h,int const src_c,int const del_c)
{
    deleteChannelByCu<float>(dstImage,srcImage,src_w,src_h,src_c,del_c);
    return 0;
}