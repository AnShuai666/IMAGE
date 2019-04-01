/*
 * @desc    可视化
 * @author  安帅
 * @date    2019-04-01
 * @email   1028792866@qq.com
*/

#ifndef IMAGE_VISUALLIZER_H
#define IMAGE_VISUALLIZER_H

#include "define.h"
#include <vector>
#include "image.hpp"
IMAGE_NAMESPACE_BEGIN
class Visualizer
{
public:
    struct Keypoint
    {
        float x;
        float y;
        float radius;
        float orientation;
    };

    enum KeypointStyle
    {
        RADIUS_BOX_ORIENTATION,
        RADIUS_CIRCLE_ORIENTATION,
        SMALL_CIRCLE_STATIC,
        SMALL_DOT_STATIC
    };

public:
    /*
    *  @property   画特征点
    *  @func       在灰度图像上画指定形状的特征点
    *  @param_in   image        待绘图像
    *  @param_in   keypoint     关键点
    *  @param_in   style        关键点绘画风格:方框\圆形等等
    *  @param_in   color        绘画颜色
    *  @static     静态成员函数  属于类不属于对象的类成员函数
    *  @return     void
    */
    static void draw_keypoint(image::ByteImage& image, Keypoint const& keypoint,KeypointStyle style,uint8_t const* color);

    /*
    *  @property   画特征点
    *  @func       在灰度图像上画指定形状的特征点
    *  @param_in   image        待绘图像
    *  @param_in   keypoint     关键点序列
    *  @param_in   style        关键点绘画风格:方框\圆形等等
    *  @static     静态成员函数  属于类不属于对象的类成员函数
    *  @return     void
    */
    static image::ByteImage::Ptr draw_keypoints(image::ByteImage::ConstPtr image, std::vector<Keypoint> const& keypoints, KeypointStyle style);

    static image::ByteImage::Ptr draw_keypoints(image::ByteImage::ConstPtr image1,image::ByteImage::ConstPtr image2);

};

IMAGE_NAMESPACE_END


#endif //IMAGE_VISUALLIZER_H
