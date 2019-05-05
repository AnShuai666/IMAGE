/*
 * @功能      image.hpp内TODO函数实现
 * @姓名      杨丰拓
 * @日期      2019-4-29
 * @时间      17:14
 * @邮箱
*/
#ifndef _IMAGE_CUH_
#define _IMAGE_CUH_



/**
 * @func
 * @param_in&out    image
 * @param_in        color
 * @param_in        w
 * @param_in        h
 * @param_in        c
 * @param_in        color_size
 * @param_in        contrast      cpu实现,与gpu实现对比
 * @return
 */
template <typename T>
int fill_color_by_cuda(T *image,T *color,int const w,int const h,int const c,int const color_size,T *contrast);

template <>
int fill_color_by_cuda<char>(char *image,char *color,int const w,int const h,int const c,int const color_size,char *contrast);

template <>
int fill_color_by_cuda<float>(float  *image,float *color,int const w,int const h,int const c,int const color_size,float *contrast);


#endif //_IMAGE_CUH_