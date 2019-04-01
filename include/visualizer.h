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
template <typename T>
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

    typedef std::vector<Keypoint> Keypoints;

    enum KeypointStyle
    {
        RADIUS_BOX_ORIENTATION,
        RADIUS_CIRCLE_ORIENTATION,
        SMALL_CIRCLE_STATIC,
        SMALL_DOT_STATIC
    };

    Keypoints keypoints;

public:

    /*
    *  @property   保存关键点
    *  @func       存储图像关键点,供draw_keypoints()函数作为参数使用
    *  @param_in   image        待特征提取图像
    *  @return     Keypoints    返回特征点向量引用
    */
    Keypoints& save_keypoints(typename Image<T>::Ptr image);

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
    static void draw_keypoint(typename Image<T>::Ptr image, Keypoint const& keypoint,KeypointStyle style,uint8_t const* color);

    /*
    *  @property   画特征点
    *  @func       在灰度图像上画指定形状的特征点
    *  @param_in   image        待绘图像
    *  @param_in   keypoint     关键点序列
    *  @param_in   style        关键点绘画风格:方框\圆形等等
    *  @static     静态成员函数  属于类不属于对象的类成员函数
    *  @return     void
    */
    static typename Image<T>::Ptr draw_keypoints(typename Image<T>::Ptr image, Keypoints const& keypoints, KeypointStyle style);

    static typename Image<T>::Ptr draw_matches(Image<T> &image1,Image<T>& image2);

    void draw_line(Image<T>& image, int x1,int y1,int x2,int y2,T const* color);

    void draw_circle(Image<T>image, int x, int y, int radius,T const* color);

    void draw_rectangle(Image<T>& image, int x1,int y1,int x2,int y2,T const* color);

    void draw_box(Image<T>& image, float x, float y,float size, float orientation, uint8_t const* color);
};

template <typename T>
typename Visualizer<T>::Keypoints&
Visualizer<T>::save_keypoints(typename Image<T>::Ptr image)
{

}
template <typename T>
void
Visualizer<T>::draw_keypoint(typename Image<T>::Ptr image, const image::Visualizer<T>::Keypoint &keypoint, image::Visualizer<T>::KeypointStyle style, uint8_t const *color)
{

}
template <typename T>
typename Image<T>::Ptr
Visualizer<T>::draw_keypoints(typename Image<T>::Ptr image,const std::vector<image::Visualizer<T>::Keypoint> &keypoints, image::Visualizer<T>::KeypointStyle style)
{

}

template <typename T>
void
Visualizer<T>::draw_line(image::Image<T> &image, int x1, int y1, int x2, int y2, T const *color)
{

}

template <typename T>
void
image::Visualizer<T>::draw_circle(image::Image<T> image, int x, int y, int radius, T const *color)
{

}

template <typename T>
void
image::Visualizer<T>::draw_rectangle(image::Image<T> &image, int x1, int y1, int x2, int y2, T const *color)
{

}

template <typename T>
void
image::Visualizer<T>::draw_box(image::Image<T> &image, float x, float y, float size, float orientation,uint8_t const *color)
{

}

IMAGE_NAMESPACE_END

#endif //IMAGE_VISUALLIZER_H
