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
enum DesaturateType
{
    DESATURATE_MAXIMUM,
    DESATURATE_LIGHTNESS,
    DESATURATE_LUMINOSITY,
    DESATURATE_AVERAGE
};

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
*  @func       将图像中位图转换为浮点图像，灰度值范围从[0-255]->[0.0,1.0]
*  @param_in   image            待转换图像
*  @typename   防止歧义，显示声明Image<T>::Ptr是类型而非变量
*  @return     FloatImage::Ptr  转换后的图像 原图不变，获得新图
*/
template <typename T>
typename Image<T>::Ptr
desaturate(typename Image<T>::ConstPtr image,Des)

IMAGE_NAMESPACE_END



#endif //IMAGE_IMAGE_PROCESS_HPP
