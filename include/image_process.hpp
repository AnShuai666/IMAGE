/*
 * @desc    图像处理函数
 * @author  安帅
 * @date    2019-01-22
 * @email   1028792866@qq.com
*/

#ifndef IMAGE_IMAGE_PROCESS_HPP
#define IMAGE_IMAGE_PROCESS_HPP

#include "define.h"
#include "image.hpp"

IMAGE_NAMESPACE_BEGIN
/*******************************************************************
*~~~~~~~~~~~~~~~~~~~~~图像饱和度类型枚举声明~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*******************************************************************/
//http://changingminds.org/explanations/perception/visual/lightness_variants.htm
//三者都是亮度，HSL空间中的L,不同软件的L不一样分别为LIGHTNESS，LUMINOSITY与LUMINANCE
enum DesaturateType
{
    DESATURATE_MAXIMUM,     //Maximum = max(R,G,B)
    DESATURATE_LIGHTNESS,   //Lightness = 1/2 * (max(R,G,B) + min(R,G,B))
    DESATURATE_LUMINOSITY,  //Luminosity = 0.21 * R + 0.72 * G + 0.07 * B
    DESATURATE_LUMINANCE,   //Luminince = 0.30 * R + 0.59 * G + 0.11 * B
    DESATURATE_AVERAGE      //Average Brightness = 1/3 * (R + G +B)
};

/*******************************************************************
*~~~~~~~~~~~~~~~~~~~~~常用图像函数处理声明~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*******************************************************************/

/*
*  @property   图像转换
*  @func       将图像中位图转换为浮点图像，灰度值范围从[0-255]->[0.0,1.0]
*  @param_in   image            待转换图像
*  @return     FloatImage::Ptr  转换后的图像 原图不变，获得新图
*/
FloatImage::Ptr
byte_to_float_image(ByteImage::ConstPtr image);

/*
*  @property   图像转换
*  @func       RGB->HSL          L = max(R,G,B)
*  @param_in   image            待转换图像
*  @return     T
*/
template <typename T>
T desaturate_maximum(T const* v);

/*
*  @property   图像转换
*  @func       RGB->HSL          L = 1/2 * (max(R,G,B)+mai(R,G,B))
*  @param_in   image            待转换图像
*  @return     T
*/
template <typename T>
T desaturate_lightness(T const* v);

/*
*  @property   图像转换
*  @func       RGB->HSL          L = 0.21 * R + 0.72 * G + 0.07 * B
*  @param_in   image            待转换图像
*  @return     T
*/
template <typename T>
T desaturate_luminosity(T const* v);

/*
*  @property   图像转换
*  @func       RGB->HSL          L = 0.30 * R + 0.59 * G + 0.11 * B
*  @param_in   image            待转换图像
*  @return     T
*/
template <typename T>
T desaturate_luminance(T const* v);

/*
*  @property   图像转换
*  @func       RGB->HSL          L = 1/3 * (R + G +B)
*  @param_in   image            待转换图像
*  @return     T
*/
template <typename T>
T desaturate_average(T const* v);

/*
*  @property   图像饱和度降低
*  @func       将图像转换为几种HSL图像
*  @param_in   image            待转换图像
*  @param_in   type             亮度类型
*  @typename   防止歧义，显示声明Image<T>::Ptr是类型而非变量
*  @return     Image<T>::Ptr
*/
template <typename T>
typename Image<T>::Ptr
desaturate(typename Image<T>::ConstPtr image,DesaturateType type);

IMAGE_NAMESPACE_END

/*******************************************************************
*~~~~~~~~~~~~~~~~~~~~~常用图像函数处理实现~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*******************************************************************/

IMAGE_NAMESPACE_BEGIN

template <typename T>
inline T
desaturate_maximum(T const* v)
{
    return *std::max_element(v, v + 3);
}

template <typename T>
inline T
desaturate_lightness(T const* v)
{
    T const max = *std::max_element(v, v + 3);
    T const min = *std::min_element(v, v + 3);
    return 0.5f * (max + min);
}

template <typename T>
inline T
desaturate_luminosity(T const* v)
{
    return 0.21f * v[0] + 0.72f * v[1] + 0.07f * v[2];
}


template <typename T>
inline T
desaturate_luminance(T const* v)
{
    return 0.30 * v[0] + 0.59f * v[1] + 0.11f * v[2];
}


template <typename T>
inline T desaturate_average(T const* v)
{
    return ((float)(v[0] + v[1] + v[2])) / 3.0f;
}


template <typename T>
typename Image<T>::Ptr desaturate(typename Image<T>::ConstPtr image, DesaturateType type)
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
    for (int i = 0; i < image->get_pixel_amount(); ++i)
    {
        T const* v = &image->at(0);
        out_image->at(out_pos) = func(v);

        if (has_alpha)
        {
            out_image->at(out_pos + 1) = image->at(in_pos + 3);
        }

        out_pos += 1 + has_alpha;
        in_pos += 3 + has_alpha;
    }

    return out_image;
}

IMAGE_NAMESPACE_END

#endif //IMAGE_IMAGE_PROCESS_HPP
