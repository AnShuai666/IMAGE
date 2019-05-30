

#ifndef _PROCESS_CUDA_00_H_
#define _PROCESS_CUDA_00_H_

//#include "define.h"
#include "MATH/Function/function.hpp"
#include "cuda_include/image_process_1.cuh"
#include "MATH/Util/timer.h"
#include "IMAGE/image_process.hpp"

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

/*******************************************************************
*~~~~~~~~~~~~~~~~~~~~~常用图像函数处理声明~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*******************************************************************/
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
desaturate_cu(typename Image<T>::ConstPtr image,DesaturateType type);

/*
*  @property   图像缩放
*  @func       将图像放大为原图两倍　均匀插值，最后一行后最后一列与倒数第二行与倒数第二列相同
*  @param_in   image          　  待放大图像
*  @typename   防止歧义，显示声明Image<T>::Ptr是类型而非变量
*  @return     Image<T>::Ptr
*/
template <typename T>
typename Image<T>::Ptr
rescale_double_size_supersample_cu(typename Image<T>::ConstPtr img);

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
rescale_half_size_cu(typename Image<T>::ConstPtr img);

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
rescale_half_size_gaussian_cu(typename Image<T>::ConstPtr image, float sigma2 = 0.75f);


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
blur_gaussian_cu(typename Image<T>::ConstPtr in, float sigma);

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
blur_gaussian2_cu(typename Image<T>::ConstPtr in, float sigma2);

/*
*  @property   求图像差
*  @func       求差异图像的有符号图像,image_1 - image<T>
*  @param_in   image_1  image_2  相减的两幅图像
*  @return     Image<T>::Ptr
*/
template <typename T>
typename Image<T>::Ptr
subtract_cu(typename Image<T>::ConstPtr image_1, typename Image<T>::ConstPtr image_2);

/*
*  @property   求图像差
*  @func       求差异图像的无符号图像,image_1 - image<T>
*  @param_in   image_1  image_2  相减的两幅图像
*  @return     Image<T>::Ptr
*/
template <typename T>
typename Image<T>::Ptr
difference_cu(typename Image<T>::ConstPtr image_1, typename Image<T>::ConstPtr image_2);
IMAGE_NAMESPACE_END

/*******************************************************************
*~~~~~~~~~~~~~~~~~~~~~常用图像函数处理实现~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*******************************************************************/

        IMAGE_NAMESPACE_BEGIN
template <typename T>
typename Image<T>::Ptr desaturate_cu(typename Image<T>::ConstPtr image, DesaturateType type)
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

    desaturateByCuda(&out_image->at(0),&image->at(0),image->get_pixel_amount(),type,has_alpha);

    return out_image;
}


//TODO::to be CUDA@YANG
//opencv 4000*2250*3 图像处理时间: 94.8ms
template <typename T>
typename Image<T>::Ptr
rescale_double_size_supersample_cu(typename Image<T>::ConstPtr img)
{


    int const iw = img->width();
    int const ih = img->height();
    int const ic = img->channels();
    int const ow = iw << 1;
    int const oh = ih << 1;
    typename Image<T>::Ptr out(Image<T>::create());
    out->allocate(ow,oh,ic);
    doubleSizeByCuda(&out->at(0),&img->at(0),iw,ih,ic);

    return out;
}

//TODO::to be CUDA@YANG
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
//opencv 4000*2250*3 图像处理时间: 16.6ms
template <typename T>
typename Image<T>::Ptr
rescale_half_size_cu(typename Image<T>::ConstPtr img)
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
    halfsizeByCuda(&out->at(0),&img->at(0),iw,ih,ic,&out->at(0));
    return out;
}
//TODO::to be CUDA@YANG
/*
    *  @property   图像缩放
    *  @func       将图像缩小为原图两倍　高斯权值插值，　核尺寸为４x4像素
            *              边缘需要进行像素拓展．
    *              该高斯核为高斯函数f(x)=1/[sqrt(2pi)*sigma] * e^-(x^2/2sigma2)
    *              其中，x为中心像素点到核的各像素的像素距离，有三个值：sqrt(2)/2,sqrt(10)/2,3sqrt(2)/2
    *              第一个数接近sigma=sqrt(3)/2
                               *  @param_in   image            　待缩小图像
    *  @param_in   sigma2             高斯尺度平方值　　也就是方差　默认3/4
    *  */
template <typename T>
typename Image<T>::Ptr
rescale_half_size_gaussian_cu(typename Image<T>::ConstPtr image, float sigma2)
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
    halfSizeGaussianByCuda(&out->at(0),&image->at(0),iw,ih,ic,sigma2);
    return out;
}

//TODO::to be CUDA@YANG
//opencv 4000*2250*3 图像处理时间：60.9ms
/*
*  @property   图像模糊
*  @func       将对图像进行高斯模糊,运用高斯卷积核,进行可分离卷积,先对x方向进行卷积,再在y方向进行卷积,
*              等同于对图像进行二维卷积
*              该高斯核为高斯函数f(x,y)=1/[(2pi)*sigma^2] * e^-((x^2 + y^2)/2sigma2)
*
*
*  @param_in   in            　  待模糊图像
*  @param_in   sigma             目标高斯尺度值　　也就是标准差
*/
//借鉴深度学习里的卷积优化
template <typename T>
typename Image<T>::Ptr
blur_gaussian_cu(typename Image<T>::ConstPtr in, float sigma)
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
    //可分离高斯核实现
    //调用gpu代码
    typename Image<T>::Ptr out(Image<T>::create());
    out->allocate(w,h,c);
    blurGaussianByCuda(&out->at(0),&in->at(0),w,h,c,sigma);
    return out;///返回gpu的计算结果
}

template <typename T>
typename Image<T>::Ptr
blur_gaussian2_cu(typename Image<T>::ConstPtr in, float sigma2) {
    if (in == nullptr) {
        throw std::invalid_argument("没有输入图像!\n");
    }

    if (sigma2 < 0.01f) {
        return in->duplicate();
    }
    //调用gpu代码
    //blur_gaussian_cu函数中已经调用过blur_gaussian_by_cuda函数
    typename Image<T>::Ptr out(Image<T>::create());
    out->allocate(in->width(),in->height(),in->channels());
    blurGaussian2ByCuda(&out->at(0),&in->at(0),in->width(),in->height(),in->channels(),sigma2);
    return out;///返回gpu的计算结果
}
//TODO::to be CUDA@YANG
//opencv 4000*2250*3 图像处理时间：4.76ms
template <typename T>
typename Image<T>::Ptr
subtract_cu(typename Image<T>::ConstPtr image_1, typename Image<T>::ConstPtr image_2)
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

    ///调用gpu代码
    typename Image<T>::Ptr out(Image<T>::create());
    out->allocate(w1,h1,c1);
    subtractByCuda(&out->at(0),&image_1->at(0),&image_2->at(0),w1,h1,c1);

    return out;
}
//TODO::to be CUDA@YANG
//opencv 4000*2250*3 图像处理时间：3.34ms
template <typename T>
typename Image<T>::Ptr
difference_cu(typename Image<T>::ConstPtr image_1, typename Image<T>::ConstPtr image_2)
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
    ///调用gpu代码
    difference_by_cuda(&out->at(0),&image_1->at(0),&image_2->at(0),w1,h1,c1);
    return out;
}

IMAGE_NAMESPACE_END
#endif //_PROCESS_CUDA_1_H_