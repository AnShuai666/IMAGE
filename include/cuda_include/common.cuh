
#ifndef _COMMON_CUH_
#define _COMMON_CUH_
#include <sys/time.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}
using namespace std;
template <typename T>
void compare1(T *image,T *contrast,int const wc,int const h,bool has_point);
template <typename T>
void compare1(T *image,T *contrast,int const wc,int const h,bool has_point)
{
    float diff=0.000001f;
    int key=0;
    int message=0;
    for (int j = 0; j <h ; ++j)
    {
        for (int i = 0; i <wc ; ++i)
        {
            int idx=j*wc+i;//像素点索引
            T a=contrast[idx];//对照组数据
            T b=image[idx];//gpu计算后的数据
            if(a!=b)//结果是否一致
            {
                message=1;
                key++;
                cout<<"idx:"<<idx<<"\t"<<endl;
                if(has_point)//输出是否包含小数点
                {
                    cout.setf(ios::fixed);
                    cout << fixed << setprecision(10) << "cpu:" << a << "\tgpu:" << b << "\tdiff:" << (a - b)
                         << endl;
                    cout.unsetf(ios::fixed);
                }
                else cout << "cpu:" << (int)a << "\tgpu:" << (int)b << "\tdiff:" << (a - b)<< endl;

                if(a-b<diff)//误差是否小于设定值
                {
                    message=2;
                }
            }
            if(key==20)break;
        }
        if(key==20)break;
    }

    switch (message)
    {
        case 1:
            cerr<<"gpu实现与cpu实现不一致,gpu实现失败!"<<endl;
            break;
        case 2:
            cout<<"gpu实现与cpu实现不一致,存在误差!(约为"<<diff<<")"<<endl;
            break;
        default:
            cout<<"gpu实现与cpu实现一致"<<endl;
            break;
    }
}

#define CHECK_CUBLAS(call)                                                     \
{                                                                              \
    cublasStatus_t err;                                                        \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CURAND(call)                                                     \
{                                                                              \
    curandStatus_t err;                                                        \
    if ((err = (call)) != CURAND_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUFFT(call)                                                      \
{                                                                              \
    cufftResult err;                                                           \
    if ( (err = (call)) != CUFFT_SUCCESS)                                      \
    {                                                                          \
        fprintf(stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__,        \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUSPARSE(call)                                                   \
{                                                                              \
    cusparseStatus_t err;                                                      \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
    {                                                                          \
        fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);   \
        cudaError_t cuda_err = cudaGetLastError();                             \
        if (cuda_err != cudaSuccess)                                           \
        {                                                                      \
            fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                    cudaGetErrorString(cuda_err));                             \
        }                                                                      \
        exit(1);                                                               \
    }                                                                          \
}
/*
 *
inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
  获取起始时间点:
    iStart = seconds();
    iElaps = (seconds() - iStart)*1000;
 *  获取耗费的时间ms;
 */


#endif // _COMMON_CUH_