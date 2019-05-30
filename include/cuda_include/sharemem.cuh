

#ifndef _SHAREDMEM_H_
#define _SHAREDMEM_H_

/*****************************************************************************
//  template<class T>
//  __global__ void
//  foo( T* g_idata, T* g_odata)
//  {
//      // Shared mem size is determined by the host app at run time
//      extern __shared__  T sdata[];
//      ...
//      doStuff(sdata);
//      ...
//   }
//
//   With this
//  template<class T>
//  __global__ void
//  foo( T* g_idata, T* g_odata)
//  {
//      // Shared mem size is determined by the host app at run time
//      sharedMemory<T> smem;
//      T* sdata = smem.p_getPointer();
//      ...
//      doStuff(sdata);
//      ...
//   }
****************************************************************************/

template <typename T>
struct sharedMemory
{
    __device__ T *p_getPointer()
    {
        extern __device__ void error(void);
        error();
        return NULL;
    }
};


template <>
struct sharedMemory <int>
{
    __device__ int *p_getPointer()
    {
        extern __shared__ int s_int[];
        return s_int;
    }
};

template <>
struct sharedMemory <unsigned int>
{
    __device__ unsigned int *p_getPointer()
    {
        extern __shared__ unsigned int s_uint[];
        return s_uint;
    }
};

template <>
struct sharedMemory <char>
{
    __device__ char *p_getPointer()
    {
        extern __shared__ char s_char[];
        return s_char;
    }
};

template <>
struct sharedMemory <unsigned char>
{
    __device__ unsigned char *p_getPointer()
    {
        extern __shared__ unsigned char s_uchar[];
        return s_uchar;
    }
};

template <>
struct sharedMemory <short>
{
    __device__ short *p_getPointer()
    {
        extern __shared__ short s_short[];
        return s_short;
    }
};

template <>
struct sharedMemory <unsigned short>
{
    __device__ unsigned short *p_getPointer()
    {
        extern __shared__ unsigned short s_ushort[];
        return s_ushort;
    }
};

template <>
struct sharedMemory <long>
{
    __device__ long *p_getPointer()
    {
        extern __shared__ long s_long[];
        return s_long;
    }
};

template <>
struct sharedMemory <unsigned long>
{
    __device__ unsigned long *p_getPointer()
    {
        extern __shared__ unsigned long s_ulong[];
        return s_ulong;
    }
};

template <>
struct sharedMemory <bool>
{
    __device__ bool *p_getPointer()
    {
        extern __shared__ bool s_bool[];
        return s_bool;
    }
};

template <>
struct sharedMemory <float>
{
    __device__ float *p_getPointer()
    {
        extern __shared__ float s_float[];
        return s_float;
    }
};

template <>
struct sharedMemory <double>
{
    __device__ double *p_getPointer()
    {
        extern __shared__ double s_double[];
        return s_double;
    }
};


#endif //_SHAREDMEM_H_
