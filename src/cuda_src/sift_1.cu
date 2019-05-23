/**
 * @功能    sift.cpp文件实现
 * @姓名    杨丰拓
 * @日期    2019-05-20
 * @时间    15:35
*/
//#include <cuda_runtime.h>
#include <cstdio>
#include "cuda_include/sift_1.cuh"
#include "../../../../../../../usr/local/cuda/include/cuda_runtime.h"
#include <iostream>

__global__ void kernel_extrema_by_cuda(int *d_compact,int *d_noff,const float *d_in0,const float *d_in1,const float *d_in2,int const w,int const h)
{
   // __shared__ int data[9];
  int x=threadIdx.x+blockDim.x*blockIdx.x;
  int y=threadIdx.y+blockDim.y*blockIdx.y;
  int noff;

  if(x<(w-1)&&y<(h-1))
  {
      x=x+1;
      y=y+1;
      int idx=y*w+x;
      float center_value=d_in1[idx];
      bool largest = true;
      bool smallest = true;

      for (int i = 0; (largest || smallest) && i < 9; ++i)
      {
          noff=d_noff[i]+idx;
          if (d_in0[noff]>= center_value)largest = false;
          if (d_in0[noff]<= center_value)smallest = false;
          if (d_in2[noff]>= center_value)largest = false;
          if (d_in2[noff]<= center_value)smallest = false;
          if(i!=4)
          {
              if (d_in1[noff]>= center_value)largest = false;
              if (d_in1[noff]<= center_value)smallest = false;
          }
      }

      if(idx==251)printf("%d\t%d\n",largest,smallest);
      if (smallest||largest)
      {
          d_compact[idx]=1;
      }
  }
}

void  extrema_detection_cu(int const  w,int const  h,int *noff,int octave_index,int sample_index,
                           const float *s0,const float *s1,const float *s2)
{
    //计算存储空间字节数
    size_t const bytes=w*h* sizeof(float);
    size_t const bytes_off=9* sizeof(int);
    size_t const bytes_compact=w*h*sizeof(int);
    //声明显存指针
    float *d_in0=NULL;
    float *d_in1=NULL;
    float *d_in2=NULL;
    int *d_noff=NULL;
    int *d_compact=NULL;
    //int *h_compact=new int(h*w);
    int *h_compact=(int*)malloc(bytes_compact);
    //分配显存

    cudaMalloc(&d_in0,bytes);
    cudaMalloc(&d_in1,bytes);
    cudaMalloc(&d_in2,bytes);
    cudaMalloc(&d_noff,bytes_off);
    cudaMalloc(&d_compact,bytes_compact);
    //cpu2gpu
    cudaError_t message;
    cudaMemcpy(d_noff,noff,bytes_off,cudaMemcpyHostToDevice);
    cudaMemcpy(d_in0,s0,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_in1,s1,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2,s2,bytes,cudaMemcpyHostToDevice);
    cudaMemset(d_compact,0,bytes_compact);
    //网格划分
    int x=32;
    int y=4;
    dim3 block(x,y,1);
    dim3 grid((w-1+x)/x,(h-1+y)/y,1);

    kernel_extrema_by_cuda<<<grid,block>>>(d_compact,d_noff,d_in0,d_in1,d_in2,w,h);

    cudaMemcpy(h_compact,d_compact,bytes_compact,cudaMemcpyDeviceToHost);
    int sum=0;
    for (int j = 0; j <h ; ++j) {
        for (int i = 0; i <w ; ++i) {
           sum+=h_compact[j*w+i];
        }
    }
    std::cout<<"gpu:"<<sum<<std::endl;

    //释放显存
    cudaFree(d_in0);
    cudaFree(d_in1);
    cudaFree(d_in2);
    cudaFree(d_noff);
    cudaFree(d_compact);
    free(h_compact);
}
