/*
 * @desc    基于CUDA加速的图像处理函数
 * @author  杨丰拓
 * @date    2019-04-01
 * @email   yangfengtuo@163.com
*/
#ifndef IMAGE_PROCESS_CUH
#define IMAGE_PROCESS_CUH

#include <cuda_runtime.h>
#include "image.hpp"
#include "function/function.hpp"
enum DesaturateType
{
    DESATURATE_MAXIMUM,     //Maximum = max(R,G,B)
    DESATURATE_LIGHTNESS,   //Lightness = 1/2 * (max(R,G,B) + min(R,G,B))
    DESATURATE_LUMINOSITY,  //Luminosity = 0.21 * R + 0.72 * G + 0.07 * B
    DESATURATE_LUMINANCE,   //Luminince = 0.30 * R + 0.59 * G + 0.11 * B
    DESATURATE_AVERAGE      //Average Brightness = 1/3 * (R + G +B)
};

template <typename T>
__device__ T desaturate_maximum_cu(T const * v)
{
    return max(v,max(v+1,v+2));
}

template <typename T>
__device__ T desaturate_lightness_cu(T const * v)
{
    T const max = max(v,max(v+1,v+2));
    T const min = min(v,min(v+1,v+2));
    return 0.5f*(max+min);
}

template <typename T>
__device__ T desaturate_luminosity_cu(T const* v)
{
    return 0.21f * v[0] + 0.72f * v[1] + 0.07f * v[2];
}

template <typename T>
__device__ T desaturate_luminance_cu(T const* v)
{
    return 0.30 * v[0] + 0.59f * v[1] + 0.11f * v[2];
}


template <typename T>
__device__ T desaturate_average_cu(T const*v)
{
    return ((float)(v[0] + v[1] + v[2])) / 3.0f;
}

template <typename T>
__global__ void desature_cu(T *out,T *in,const int pixel_amount,const bool has_alpha)
{
    for (int i = 0; i < image->get_pixel_amount(); ++i)
    {
        T const* v = &image->at(in_pos);
        out_image->at(out_pos) = func(v);
        //std::cout<<out_image->at(out_pos)<<std::endl;
        if (has_alpha)
        {
            out_image->at(out_pos + 1) = image->at(in_pos + 3);
        }

        out_pos += 1 + has_alpha;
        in_pos += 3 + has_alpha;
    }
}

template <typename T>
typename Image<T>::Ptr desaturate_cuda(typename Image<T>::ConstPtr image, DesaturateType type)
{
    if (image == NULL)
    {
        throw std::invalid_argument("无图像传入！");
    }

    if (image->channels() != 3 && image->channels() != 4)
    {
        throw std::invalid_argument("图像必须是RGB或者RGBA!");
    }

    bool has_alpha = (image->channels() == 4);

    typename Image<T>::Ptr out_image(Image<T>::create());
    out_image->allocate(image->width(),image->height(),1 + has_alpha);

    typedef T (*DesaturateFunc)(T const*);
    DesaturateFunc func;

    switch (type)
    {
        case DESATURATE_MAXIMUM:
            func = desaturate_maximum<T>;
            break;
        case DESATURATE_LIGHTNESS:
            func = desaturate_lightness<T>;
            break;
        case DESATURATE_LUMINOSITY:
            func = desaturate_luminosity;
            break;
        case DESATURATE_LUMINANCE:
            func = desaturate_luminance;
            break;
        case DESATURATE_AVERAGE:
            func = desaturate_average;
            break;
        default:
            throw std::invalid_argument("非法desaturate类型");
    }

    int out_pos = 0;
    int in_pos = 0;
    //TODO:: to be CUDA @Yang
    //opencv 4000*2250*3 图像处理时间: 14.4ms
    /*std::cout<<"*****************开始操作*****************"<<std::endl;
    for (int i = 0; i < image->get_pixel_amount(); ++i)
    {
        T const* v = &image->at(in_pos);
        out_image->at(out_pos) = func(v);
        //std::cout<<out_image->at(out_pos)<<std::endl;
        if (has_alpha)
        {
            out_image->at(out_pos + 1) = image->at(in_pos + 3);
        }

        out_pos += 1 + has_alpha;
        in_pos += 3 + has_alpha;
    }
    std::cout<<"*****************结束操作*****************"<<std::endl;*/

    return out_image;
}
#endif //IMAGE_PROCESS_CUH