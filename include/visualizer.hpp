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

struct KeypointVis
{
    float x;
    float y;
    float radius;
    float orientation;
};



template <typename T>
class Visualizer
{
public:
    enum KeypointStyle
    {
        RADIUS_BOX_ORIENTATION,
        RADIUS_CIRCLE_ORIENTATION,
        SMALL_CIRCLE_STATIC,
        SMALL_DOT_STATIC
    };

    typedef std::vector<KeypointVis> Keypoints;
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
    void draw_keypoint(typename Image<T>::Ptr image, KeypointVis const& keypoint,KeypointStyle style,uint8_t const* color);

    /*
    *  @property   画特征点
    *  @func       在灰度图像上画指定形状的特征点
    *  @param_in   image        待绘图像
    *  @param_in   keypoint     关键点序列
    *  @param_in   style        关键点绘画风格:方框\圆形等等
    *  @static     静态成员函数  属于类不属于对象的类成员函数
    *  @return     void
    */
    typename Image<T>::Ptr draw_keypoints(typename Image<T>::Ptr image, Keypoints const& keypoints, KeypointStyle style);

    typename Image<T>::Ptr draw_matches(Image<T> &image1,Image<T>& image2);

    void draw_line(Image<T>& image, int x1,int y1,int x2,int y2,T const* color);

    void draw_circle(Image<T>& image, int x, int y, int radius,T const* color);

    void draw_rectangle(Image<T>& image, int x1,int y1,int x2,int y2,T const* color);

    void draw_box(Image<T>& image, float x, float y,float size, float orientation, uint8_t const* color);

private:
    unsigned char color_table[12][3] =
            {
                    { 255, 0,   0 }, { 0, 255,   0 }, { 0, 0, 255   },
                    { 255, 255, 0 }, { 255, 0, 255 }, { 0, 255, 255 },
                    { 127, 255, 0 }, { 255, 127, 0 }, { 127, 0, 255 },
                    { 255, 0, 127 }, { 0, 127, 255 }, { 0, 255, 127 }
            };
};

template <typename T>
typename Visualizer<T>::Keypoints&
Visualizer<T>::save_keypoints(typename Image<T>::Ptr image)
{

}
template <typename T>
void
Visualizer<T>::draw_keypoint(typename Image<T>::Ptr image, const KeypointVis &keypoint, KeypointStyle style, uint8_t const *color)
{
    int const x = static_cast<int>(keypoint.x + 0.5);
    int const y = static_cast<int>(keypoint.y + 0.5);
    int const width = image->width();
    int const height = image->height();
    int const channels = image->channels();

    if (x < 0 || x >= width || y < 0 || y >= height)
        return;

    int required_space = 0;
    bool draw_orientation = false;
    switch (style)
    {
        default:
        case SMALL_DOT_STATIC:
            required_space = 0;
            draw_orientation = false;
            break;
        case SMALL_CIRCLE_STATIC:
            required_space = 3;
            draw_orientation = false;
            break;
        case RADIUS_BOX_ORIENTATION:
            required_space = static_cast<int>(std::sqrt
                    (2.0f * keypoint.radius * keypoint.radius)) + 1;
            draw_orientation = true;
            break;
        case RADIUS_CIRCLE_ORIENTATION:
            required_space = static_cast<int>(keypoint.radius);
            draw_orientation = true;
            break;
    }

    if (x < required_space || x >= width - required_space
        || y < required_space || y >= height - required_space)
    {
        style = SMALL_DOT_STATIC;
        required_space = 0;
        draw_orientation = false;
    }

    switch (style)
    {
        default:
        case SMALL_DOT_STATIC:
            std::copy(color, color + channels, &image->at(x, y, 0));
            break;
        case SMALL_CIRCLE_STATIC:
            draw_circle(*image, x, y, 3, color);
            break;
        case RADIUS_BOX_ORIENTATION:
            draw_box(*image, keypoint.x, keypoint.y,
                     keypoint.radius, keypoint.orientation, color);
            break;
        case RADIUS_CIRCLE_ORIENTATION:
            draw_circle(*image, x, y, required_space, color);
            break;
    }

    if (draw_orientation)
    {
        float const sin_ori = std::sin(keypoint.orientation);
        float const cos_ori = std::cos(keypoint.orientation);
        float const x1 = (cos_ori * keypoint.radius);
        float const y1 = (sin_ori * keypoint.radius);
        draw_line(*image, static_cast<int>(keypoint.x + 0.5f),
                         static_cast<int>(keypoint.y + 0.5f),
                         static_cast<int>(keypoint.x + x1 + 0.5f),
                         static_cast<int>(keypoint.y + y1 + 0.5f), color);
    }
}
template <typename T>
typename Image<T>::Ptr
Visualizer<T>::draw_keypoints(typename Image<T>::Ptr image,const std::vector<KeypointVis> &keypoints, KeypointStyle style)
{
    typename Image<T>::Ptr ret =image;
//    if (image->channels() == 3)
//    {
//        ret = image::desaturate<T>(image, image::DESATURATE_AVERAGE);
//        ret = image::expand_grayscale<T>(ret);
//    }
//    else if (image->channels() == 1)
//    {
//        ret = image::expand_grayscale<T>(image);
//    }

    uint8_t* color = color_table[3];
    //TODO::to be CUDA @YANG
    for (std::size_t i = 0; i < keypoints.size(); ++i)
    {
        Visualizer::draw_keypoint(ret, keypoints[i], style, color);
    }

    return ret;
}
//TODO::to be CUDA @YANG
template <typename T>
void
Visualizer<T>::draw_line(image::Image<T> &image, int x1, int y1, int x2, int y2, T const *color)
{
//    if(x1<0||y1<0||x2<0||y2<0||
//    x1>=image.width()||x2>=image.width()||y1>=image.height()||y2>=image.height()) {
//        throw std::invalid_argument("直线不在图像内\n");
//    }
    int const channels = image.channels();
    int thickness=3;
    T* color_thickness=new T[thickness*channels];
    for(int it=0;it<thickness;it++)
    {
        std::copy(color,color+channels,color_thickness+it*channels);
    }
    int y_origin=y1<y2?y1:y2;
    int y_end=y1<y2?y2:y1;
    int x_origin=x1<x2?x1:x2;
    int x_end=x1>x2?x1:x2;
    int origin;
    if(x2==x1){
        if(x1<(int)(thickness/2))
            origin=0;
        else if(x1>=image.width()-(int)(thickness/2))
            origin=image.width()-thickness;
        else
            origin=x1-(int)(thickness/2);
        y_origin=y_origin>0?y_origin:0;
        y_end=y_end>=image.height()?(image.height()-1):y_end;
        for(int it=y_origin;it<=y_end;it++)
        {
            std::copy(color_thickness,color_thickness+thickness*channels,&image.at(origin,it,0));
        }
    }else if(y1==y2){
        if(y1<(int)(thickness/2))
            origin=0;
        else if(y1>=image.height()-(int)(thickness/2))
            origin=image.height()-thickness;
        else
            origin=y1-(int)(thickness/2);
        x_origin=x_origin>0?x_origin:0;
        x_end=x_end>=image.width()?(image.width()-1):x_end;
        for(int it=x_origin;it<=x_end;it++)
        {
            for(int th=0;th<thickness;th++) {
                std::copy(color, color + channels, &image.at(it, origin + th, 0));
            }
        }
    }else{
        float k=(float)(y2-y1)/(x2-x1);//x=(y-y1)/k+x1; 两点式直线方程
        y_origin=y_origin>0?y_origin:0;
        y_end=y_end>=image.height()?(image.height()-1):y_end;
        for(int y=y_origin+1;y<=y_end;y++)
        {
            if(k>0) {
                x_origin = (y - 1 - y1) / k + x1;//相邻y值下，在直线上的点的x值的范围
                x_end = (y - y1) / k + x1;
            }else{
                x_origin = (y - y1) / k + x1;
                x_end = (y - 1 - y1) / k + x1;
            }
            x_origin=x_origin>0?x_origin:0;
            x_end=x_end>=image.width()?(image.width()-1):x_end;
            for(int x=x_origin;x<=x_end;x++)
            {
                if(x<(int)(thickness/2))
                    origin=0;
                else if(x>=image.width()-(int)(thickness/2))
                    origin=image.width()-thickness;
                else
                    origin=x-(int)(thickness/2);
                std::copy(color_thickness,color_thickness+thickness*channels,&image.at(origin,y,0));
            }
        }
    }

    delete[] color_thickness;

}
//TODO::to be CUDA @YANG
template <typename T>
void
image::Visualizer<T>::draw_circle(image::Image<T>& image, int center_x, int center_y, int radius, T const *color)
{
    if(center_x<0||center_y<0||center_x>=image.width()||center_y>=image.height()) {
        throw std::invalid_argument("圆心不在图像内\n");
    }
    int const channels = image.channels();
    int thickness=3;
    T* color_thickness=new T[thickness*channels];
    for(int it=0;it<thickness;it++)
    {
        std::copy(color,color+channels,color_thickness+it*channels);
    }

    int up=0,down=0;
    int left=0,right=0;
    if(center_x-thickness>radius&&center_x<image.width()-radius-thickness
    &&center_y-thickness>radius&&center_y<image.height()-radius-thickness) {
        for (int y = 0; y <= radius; y++) {
            int x = sqrt(radius * radius - y * y);
            up = center_y - y;
            down = center_y + y;
            left = center_x - x;
            right = center_x + x;
            std::copy(color_thickness, color_thickness + thickness * channels, &image.at(left, up, 0));
            std::copy(color_thickness, color_thickness + thickness * channels, &image.at(right, up, 0));
            std::copy(color_thickness, color_thickness + thickness * channels, &image.at(right, down, 0));
            std::copy(color_thickness, color_thickness + thickness * channels, &image.at(left, down, 0));
        }
        for (int x = 0; x <= radius; x++) {
            int y = sqrt(radius * radius - x * x);
            up = center_y - y;
            down = center_y + y;
            left = center_x - x;
            right = center_x + x;
            for(int th=-thickness/2;th<=thickness/2;th++) {
                std::copy(color, color + channels, &image.at(left, up + th, 0));
                std::copy(color, color + channels, &image.at(right, up + th, 0));
                std::copy(color, color + channels, &image.at(right, down + th, 0));
                std::copy(color, color + channels, &image.at(left, down + th, 0));
            }
        }
    } else{
        for (int y = 0; y <= radius; y++) {
            int x = sqrt(radius * radius - y * y);
            up = (center_y - y)>0?(center_y - y):0;
            down = (center_y + y)<image.width()?(center_y + y):(image.width()-1);
            left = (center_x - x - thickness/2)>0?(center_x-x-thickness/2):0;
            right = (center_x + x + thickness/2)<image.width()?(center_x + x- thickness/2):(image.width()- thickness/2);
            right = right>0?right:0;
            std::copy(color_thickness, color_thickness + thickness * channels, &image.at(left, up, 0));
            std::copy(color_thickness, color_thickness + thickness * channels, &image.at(right, up, 0));
            std::copy(color_thickness, color_thickness + thickness * channels, &image.at(right, down, 0));
            std::copy(color_thickness, color_thickness + thickness * channels, &image.at(left, down, 0));
        }
        for (int x = 0; x <= radius; x++) {
            int y = sqrt(radius * radius - x * x);
            up = (center_y - y)>0?(center_y - y):0;
            down = (center_y + y)<image.width()?(center_y + y):(image.width()-1);
            left = (center_x - x - thickness/2)>0?(center_x-x-thickness/2):0;
            right = (center_x + x + thickness/2)<image.width()?(center_x + x- thickness/2):(image.width()- thickness/2);
            right = right>0?right:0;
            for(int th=-thickness/2;th<=thickness/2;th++) {
                up+=th;
                down+=th;
                up=up>0?up:0;
                down=down>0?down:0;
                down=down>=image.height()?(image.height()-1):down;
                std::copy(color, color + channels, &image.at(left, up, 0));
                std::copy(color, color + channels, &image.at(right, up, 0));
                std::copy(color, color + channels, &image.at(right, down, 0));
                std::copy(color, color + channels, &image.at(left, down, 0));
            }
        }
    }

    delete[] color_thickness;
}
//TODO::to be CUDA @YANG
template <typename T>
void
image::Visualizer<T>::draw_rectangle(image::Image<T> &image, int x1, int y1, int x2, int y2, T const *color)
{
    draw_line(image,x1,y1,x2,y1,color);
    draw_line(image,x2,y1,x2,y2,color);
    draw_line(image,x2,y2,x1,y2,color);
    draw_line(image,x1,y2,x1,x1,color);
}
//TODO::to be CUDA @YANG
template <typename T>
void
image::Visualizer<T>::draw_box(image::Image<T> &image, float x, float y, float size, float orientation,uint8_t const *color)
{
    float const sin_ori = std::sin(orientation);
    float const cos_ori = std::cos(orientation);

    float const x0 = (cos_ori * -size - sin_ori * -size);
    float const y0 = (sin_ori * -size + cos_ori * -size);
    float const x1 = (cos_ori * +size - sin_ori * -size);
    float const y1 = (sin_ori * +size + cos_ori * -size);
    float const x2 = (cos_ori * +size - sin_ori * +size);
    float const y2 = (sin_ori * +size + cos_ori * +size);
    float const x3 = (cos_ori * -size - sin_ori * +size);
    float const y3 = (sin_ori * -size + cos_ori * +size);


    draw_line(image, static_cast<int>(x + x0 + 0.5f),
                     static_cast<int>(y + y0 + 0.5f), static_cast<int>(x + x1 + 0.5f),
                     static_cast<int>(y + y1 + 0.5f), color);
    draw_line(image, static_cast<int>(x + x1 + 0.5f),
                     static_cast<int>(y + y1 + 0.5f), static_cast<int>(x + x2 + 0.5f),
                     static_cast<int>(y + y2 + 0.5f), color);
    draw_line(image, static_cast<int>(x + x2 + 0.5f),
                     static_cast<int>(y + y2 + 0.5f), static_cast<int>(x + x3 + 0.5f),
                     static_cast<int>(y + y3 + 0.5f), color);
    draw_line(image, static_cast<int>(x + x3 + 0.5f),
                     static_cast<int>(y + y3 + 0.5f), static_cast<int>(x + x0 + 0.5f),
                     static_cast<int>(y + y0 + 0.5f), color);
}

IMAGE_NAMESPACE_END

#endif //IMAGE_VISUALLIZER_H
