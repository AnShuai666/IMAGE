//
// Created by anshuai on 19-4-1.
//

#ifndef IMAGE_IMAGE_PROCESS_CU_H
#define IMAGE_IMAGE_PROCESS_CU_H
/*
 * @desc    图像处理函数
 * @author  安帅
 * @date    2019-01-22
 * @email   1028792866@qq.com
*/

#ifndef IMAGE_IMAGE_PROCESS_HPP
#define IMAGE_IMAGE_PROCESS_HPP

//#include "define.h"
#include "image.hpp"
#include "function/function.hpp"

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
//TODO:修改注释@anshuai
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
*  @param_in   image          　  待放大图像
*  @typename   防止歧义，显示声明Image<T>::Ptr是类型而非变量
*  @return     Image<T>::Ptr
*/
    template <typename T>
    typename Image<T>::Ptr
    rescale_double_size_supersample(typename Image<T>::ConstPtr img);

/*
*  @property   图像缩放
*  @func       将图像缩小为原图两倍　均匀插值，
*              偶数行与偶数列时：取四个像素的平均值；
*              奇数行列时：
*              奇数列(x)，取相邻上下像素的平均值；
*              奇数行(y)，取相邻左右像素的平均值．
*  @param_in   image            　待缩小图像
*  @typename   防止歧义，显示声明Image<T>::Ptr是类型而非变量
*  @return     Image<T>::Ptr
*/
    template <typename T>
    typename Image<T>::Ptr
    rescale_half_size(typename Image<T>::ConstPtr img);

/*
*  @property   图像缩放
*  @func       将图像缩小为原图两倍　高斯权值插值，　核尺寸为４x4像素
*              边缘需要进行像素拓展．
*              该高斯核为高斯函数f(x)=1/[sqrt(2pi)*sigma] * e^-(x^2/2sigma2)
*              其中，x为中心像素点到核的各像素的像素距离，有三个值：sqrt(2)/2,sqrt(10)/2,3sqrt(2)/2
*              第一个数接近sigma=sqrt(3)/2
*  @param_in   image            　待缩小图像
*  @param_in   sigma2             高斯尺度平方值　　也就是方差　默认3/4
*  @typename   防止歧义，显示声明Image<T>::Ptr是类型而非变量
*  @return     Image<T>::Ptr
*/
//TODO: 这里的sigma2=1/2完全满足３倍sigma，抖动没那么明显，可以用这个尺度
    template <typename T>
    typename Image<T>::Ptr
    rescale_half_size_gaussian(typename Image<T>::ConstPtr image, float sigma2 = 0.75f);


/*
*  @property   图像模糊
*  @func       将对图像进行高斯模糊,运用高斯卷积核,进行可分离卷积,先对x方向进行卷积,再在y方向进行卷积,
*              等同于对图像进行二维卷积
*              该高斯核为高斯函数f(x,y)=1/[(2pi)*sigma^2] * e^-((x^2 + y^2)/2sigma2)
*
*
*  @param_in   in            　  待模糊图像
*  @param_in   sigma             目标高斯尺度值　　也就是标准差　
*  @typename   防止歧义，显示声明Image<T>::Ptr是类型而非变量
*  @return     Image<T>::Ptr
*/
    template <typename T>
    typename Image<T>::Ptr
    blur_gaussian(typename Image<T>::ConstPtr in, float sigma);

/*
*  @property   图像模糊
*  @func       将对图像进行高斯模糊,运用高斯卷积核,进行可分离卷积,先对x方向进行卷积,再在y方向进行卷积,
*              等同于对图像进行二维卷积
*              该高斯核为高斯函数f(x,y)=1/[(2pi)*sigma^2] * e^-((x^2 + y^2)/2sigma2)
*
*
*  @param_in   in            　    待模糊图像
*  @param_in   sigma2             目标高斯尺度平方值　　也就是方差　
*  @typename   防止歧义，显示声明Image<T>::Ptr是类型而非变量
*  @return     Image<T>::Ptr
*/
    template <typename T>
    typename Image<T>::Ptr
    blur_gaussian2(typename Image<T>::ConstPtr in, float sigma2);

/*
*  @property   求图像差
*  @func       求差异图像的有符号图像,image_1 - image<T>
*  @param_in   image_1  image_2  相减的两幅图像
*  @return     Image<T>::Ptr
*/
    template <typename T>
    typename Image<T>::Ptr
    subtract(typename Image<T>::ConstPtr image_1, typename Image<T>::ConstPtr image_2);

/*
*  @property   求图像差
*  @func       求差异图像的无符号图像,image_1 - image<T>
*  @param_in   image_1  image_2  相减的两幅图像
*  @return     Image<T>::Ptr
*/
    template <typename T>
    typename Image<T>::Ptr
    difference(typename Image<T>::ConstPtr image_1, typename Image<T>::ConstPtr image_2);
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
        //TODO:: to be CUDA @Yang
        //opencv 4000*2250*3 图像处理时间: 14.4ms
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


//TODO::to be CUDA@YANG
//opencv 4000*2250*3 图像处理时间: 94.8ms
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
                                &img->at(yoff[1] + xoff[1],0)
                        };

                for (int c = 0; c < ic; ++c)
                {
                    out->at(x,y,c) = 0.25f * (val[0][c] + val[1][c] + val[2][c] + val[3][c]);
                }
            }
        }
        return out;
    }
//TODO::to be CUDA@YANG
//opencv 4000*2250*3 图像处理时间: 16.6ms
    template <typename T>
    typename Image<T>::Ptr
    rescale_half_size(typename Image<T>::ConstPtr img)
    {
        int const iw = img->width();
        int const ih = img->height();
        int const ic = img->channels();
        int ow = (iw + 1) >> 1;//缩小原来两倍，小数向上取整
        int oh = (ih + 1) >> 1;

        if(iw < 2 || ih < 2)
        {
            throw std::invalid_argument("输入图像太小，不可进行降采样！\n");
        }

        typename Image<T>::Ptr out(Image<T>::create());
        out->allocate(ow,oh,ic);

        int out_pos = 0;
        int rowstride = iw * ic;
        for (int y = 0; y < oh; ++y)
        {
            int irow1 = y * 2 * rowstride;
            int irow2 = irow1 + rowstride * (y * 2 + 1 < ih);

            for (int x = 0; x < ow; ++x)
            {
                int ipix1 = irow1 + x * 2 * ic;
                int ipix2 = irow2 + x * 2 * ic;
                int has_next = (x * 2 + 1 < iw);

                for (int c = 0; c < ic; ++c)
                {
                    out->at(out_pos++) = 0.25f * (img->at(ipix1 + c)+img->at(ipix1 + has_next * ic + c)+
                                                  img->at(ipix2 + c)+img->at(ipix2 + has_next * ic + c));
                }
            }
        }
        return out;
    }
//TODO::to be CUDA@YANG
    template <typename T>
    typename Image<T>::Ptr
    rescale_half_size_gaussian(typename Image<T>::ConstPtr image, float sigma2)
    {
        int const iw = image->width();
        int const ih = image->height();
        int const ic = image->channels();
        int const ow = (iw + 1) >> 1;
        int const oh = (ih + 1) >> 1;

        if (iw < 2 || ih < 2)
        {
            throw std::invalid_argument("图像尺寸过小，不可进行降采样!\n");
        }

        typename Image<T>::Ptr out(Image<T>::create());
        out->allocate(ow,oh,ic);

        float const w1 = std::exp(-0.5 / (2.0f * sigma2));//0.5*0.5*2
        float const w2 = std::exp(-2.5f / (2.0 * sigma2));//0.5*0.5+1.5*1.5
        float const w3 = std::exp(-4.5f / (2.0f * sigma2));//1.5*1.5*2

        int out_pos = 0;
        int const rowstride = iw * ic;
        for (int y = 0; y < oh; ++y)
        {
            int y2 = (int)y << 1;
            T const *row[4];
            row[0] = &image->at(std::max(0,(y2 - 1) * rowstride));
            row[1] = &image->at(y2 * rowstride);
            row[2] = &image->at(std::min((int)ih - 1 ,y2 + 1) * rowstride);
            row[3] = &image->at(std::min((int)ih - 2 ,y2 + 2) * rowstride);

            for (int x = 0; x < ow; ++x)
            {
                int x2 = (int)x << 1;
                int xi[4];
                xi[0] =  std::max(0,x2 - 1) * ic;
                xi[1] = x2 * ic;
                xi[2] = std::min((int)iw - 1 ,x2 + 1) * ic;
                xi[3] = std::min((int)iw - 1 ,x2 + 2) * ic;

                for (int c = 0; c < ic; ++c)
                {
                    float sum = 0.0f;//u_char溢出
                    sum += row[0][xi[0] + c] * w3;
                    sum += row[0][xi[1] + c] * w2;
                    sum += row[0][xi[2] + c] * w2;
                    sum += row[0][xi[3] + c] * w3;

                    sum += row[1][xi[0] + c] * w2;
                    sum += row[1][xi[1] + c] * w1;
                    sum += row[1][xi[2] + c] * w1;
                    sum += row[1][xi[3] + c] * w2;

                    sum += row[2][xi[0] + c] * w2;
                    sum += row[2][xi[1] + c] * w1;
                    sum += row[2][xi[2] + c] * w1;
                    sum += row[2][xi[3] + c] * w2;

                    sum += row[3][xi[0] + c] * w3;
                    sum += row[3][xi[1] + c] * w2;
                    sum += row[3][xi[2] + c] * w2;
                    sum += row[3][xi[3] + c] * w3;

                    out->at(out_pos++) = sum / (float)(4 * w3 + 8 * w2 + 4 * w1);
                }
            }
        }
        return out;
    }

//TODO::to be CUDA@YANG
//opencv 4000*2250*3 图像处理时间：60.9ms
    template <typename T>
    typename Image<T>::Ptr
    blur_gaussian(typename Image<T>::ConstPtr in, float sigma)
    {
        if (in == nullptr)
        {
            throw std::invalid_argument("没有输入图像!\n");
        }

        if(MATH_EPSILON_EQ(sigma,0.0f,0.1f))
        {
            return in->duplicate();
        }

        int const w = in->width();
        int const h = in->height();
        int const c = in->channels();
        int const ks = std::ceil(sigma * 2.884f);
        std::vector<float> kernel(ks + 1);
        float weight = 0;

        for (int i = 0; i < ks + 1; ++i)
        {
            kernel[i] = func::gaussian((float)i, sigma);
            weight += kernel[i]*2;
        }
        weight-=kernel[0];
        //可分离高斯核实现
        //x方向对对象进行卷积
        Image<float>::Ptr sep(Image<float>::create(w,h,c));
        int px = 0;
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x,++px)
            {
                for (int cc = 0; cc < c; ++cc)
                {
                    float accum=0;
                    for (int i = -ks; i <=ks; ++i)
                    {
                        int idx = func::clamp(x + i,0,w - 1);
                        accum += in->at(y * w + idx, cc) * kernel[abs(i)];
                        //printf("%f\n",kernel[abs(i)]);
                    }
                    sep->at(px,cc) = accum / weight;
                }
            }
        }
        //y方向对图像进行卷积
        px=0;
        typename Image<T>::Ptr out(Image<T>::create(w,h,c));
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x,++px)
            {
                for (int cc = 0; cc < c; ++cc)
                {
                    float accum =0;
                    for (int i = -ks; i <= ks; ++i)
                    {
                        int idx = func::clamp(y+i,0,(int)h - 1);
                        accum += sep->at(idx * w + x, cc)* kernel[abs(i)];
                    }
                    //printf("%f\n",accum / weight);
                    out->at(px,cc) = (T)(accum / weight);
                }
            }
        }
        return out;
    }

    template <typename T>
    typename Image<T>::Ptr
    blur_gaussian2(typename Image<T>::ConstPtr in, float sigma2)
    {
        if (in == nullptr)
        {
            throw std::invalid_argument("没有输入图像!\n");
        }

        if(sigma2 < 0.01f)
        {
            return in->duplicate();
        }
        float sigma=sqrt(sigma2);
        typename Image<T>::Ptr out = blur_gaussian<T>(in,sigma);
        return out;

//
//    int const w = in->width();
//    int const h = in->height();
//    int const c = in->channels();
//    int const ks = std::ceil(std::sqrt(sigma2) * 2.884f);
//    std::vector<float> kernel(ks + 1);
//    T weight = 0;
//
//    for (int i = 0; i < ks + 1; ++i)
//    {
//        kernel[i] = func::gaussian2((float)i, sigma2);
//        weight += kernel[i];
//    }
//
//    //可分离高斯核实现
//    //x方向对对象进行卷积
//    typename Image<T>::Ptr sep(Image<T>::create(w,h,c));
//    int px = 0;
//    for (int y = 0; y < h; ++y)
//    {
//        for (int x = 0; x < w; ++x,++px)
//        {
//            for (int cc = 0; cc < c; ++cc)
//            {
//                T accum(T(0));
//                for (int i = -ks; i <=ks; ++i)
//                {
//                    int idx = func::clamp(x + i,0,w - 1);
//                    accum += in->at(y * w + idx, cc) * kernel[i];
//                }
//                sep->at(px,cc) = accum / weight;
//            }
//        }
//    }
//    //y方向对图像进行卷积
//    typename Image<T>::Ptr out(Image<T>::create(w,h,c));
//    for (int y = 0; y < h; ++y)
//    {
//        for (int x = 0; x < w; ++x,++px)
//        {
//            for (int cc = 0; cc < c; ++cc)
//            {
//                T accum(T(0));
//                for (int i = -ks; i <= ks; ++i)
//                {
//                    int idx = func::clamp(y+i,0,(int)h - 1);
//                    accum += sep->at(idx * w + x, cc);
//                }
//                out->at(px,cc) = (T)accum / weight;
//            }
//        }
//    }
//    return out;
    }
//TODO::to be CUDA@YANG
//opencv 4000*2250*3 图像处理时间：4.76ms
    template <typename T>
    typename Image<T>::Ptr
    subtract(typename Image<T>::ConstPtr image_1, typename Image<T>::ConstPtr image_2)
    {
        if (image_1 == nullptr || image_2 == nullptr)
        {
            throw std::invalid_argument("至少有一幅图像为空!不满足要求!\n");
        }
        int const w1 = image_1->width();
        int const h1 = image_1->height();
        int const c1 = image_1->channels();

        if(w1 != image_2->width() || h1 != image_2->height() || c1 != image_2->channels())
        {
            throw std::invalid_argument("两图像尺寸不匹配!\n");
        }
        if(typeid(T)== typeid(uint8_t)
           || typeid(T)== typeid(uint16_t)
           || typeid(T)== typeid(uint32_t)
           || typeid(T)== typeid(uint64_t)){
            throw std::invalid_argument("无符号图像不满足要求!\n");
        }

        typename Image<T>::Ptr out(Image<T>::create());
        out->allocate(w1,h1,c1);

        for (int i = 0; i < image_1->get_value_amount(); ++i)
        {
            out->at(i) = image_1->at(i) - image_2->at(i);
        }

        return out;
    }
//TODO::to be CUDA@YANG
//opencv 4000*2250*3 图像处理时间：3.34ms
    template <typename T>
    typename Image<T>::Ptr
    difference(typename Image<T>::ConstPtr image_1, typename Image<T>::ConstPtr image_2)
    {

        if (image_1 == nullptr || image_2 == nullptr)
        {
            throw std::invalid_argument("至少有一幅图像为空!不满足要求!\n");
        }
        int const w1 = image_1->width();
        int const h1 = image_1->height();
        int const c1 = image_1->channels();

        if(w1 != image_2->width() || h1 != image_2->height() || c1 != image_2->channels())
        {
            throw std::invalid_argument("两图像尺寸不匹配!\n");
        }

        typename Image<T>::Ptr out(Image<T>::create());
        out->allocate(w1,h1,c1);

        for (int i = 0; i < image_1->get_value_amount(); ++i)
        {
            if (image_1->at(i) > image_2->at(i) )
            {
                out->at(i) = image_1->at(i) - image_2->at(i);
            } else
            {
                out->at(i) = image_2->at(i) - image_1->at(i);
            }
        }

        return out;
    }

IMAGE_NAMESPACE_END

#endif //IMAGE_IMAGE_PROCESS_HPP

#endif //IMAGE_IMAGE_PROCESS_CU_H
