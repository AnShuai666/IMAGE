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
//https://ninghao.net/video/2116
//三者都是亮度，HSL空间中的L,不同软件的L不一样分别为LIGHTNESS，LUMINOSITY与LUMINANCE
//HSL色彩模式是工业界的一种颜色标准
//H： Hue 色相             代表的是人眼所能感知的颜色范围，这些颜色分布在一个平面的色相环上
//                        基本参照：360°/0°红、60°黄、120°绿、180°青、240°蓝、300°洋红
//S：Saturation 饱和度     用0%至100%的值描述了相同色相、明度下色彩纯度的变化。数值越大，
//                        颜色中的灰色越少，颜色越鲜艳
//L ：Lightness 明度       作用是控制色彩的明暗变化。它同样使用了0%至100%的取值范围。
//                        数值越小，色彩越暗，越接近于黑色；数值越大，色彩越亮，越接近于白色。
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

/*
*  @property   图像缩放
*  @func       将图像放大为原图两倍　均匀插值，最后一行后最后一列与倒数第二行与倒数第二列相同
*  @param_in   image            待放大图像
*  @typename   防止歧义，显示声明Image<T>::Ptr是类型而非变量
*  @return     Image<T>::Ptr
*/
template <typename T>
typename Image<T>::Ptr
rescale_double_size_supersample(typename Image<T>::ConstPtr img);

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
        T const* v = &image->at(in_pos);
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

template <typename T>
typename Image<T>::Ptr
rescale_double_size_supersample(typename Image<T>::ConstPtr img)
{
    int const iw = img->width();
    int const ih = img->height();
    int const ic = img->channels();
    int const ow = iw << 1;
    int const oh = ih << 1;

    typename Image<T>::Ptr out(Image<T>::create());
    out->allocate(ow,oh,ic);

    int witer = 0;
    for (int y = 0; y < oh; ++y)
    {
        bool nexty = (y + 1) < oh;
        int yoff[2] = {iw * (y >> 1), iw * ((y + nexty) >> 1)};
        for (int x = 0; x < ow; ++x)
        {
            bool nextx = (x + 1) < ow;
            int xoff[2] = {x >> 1,(x + nextx) >> 1};
            T const* val[4] =
            {
                &img->at(yoff[0] + xoff[0],0),
                &img->at(yoff[0] + xoff[1],0),
                &img->at(yoff[1] + xoff[0],0),
                &img->at(yoff[1] + yoff[1],0)
            };

            for (int c = 0; c < ic; ++c)
            {
                out->at(x,y,c) = 0.25f * (val[0][c] + val[1][c] + val[2][c] + val[3][c]);
            }
        }
    }
    return out;
}


IMAGE_NAMESPACE_END

#endif //IMAGE_IMAGE_PROCESS_HPP
