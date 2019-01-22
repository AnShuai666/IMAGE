/*
 * @desc    SIFT特征
 * @author  安帅
 * @date    2019-1-17
 * @e-mail   1028792866@qq.com
*/


#ifndef IMAGE_SIFT_H
#define IMAGE_SIFT_H

#include <vector>
#include "define.h"
#include "Matrix/vector.hpp"
#include "image.hpp"

IMAGE_NAMESPACE_BEGIN

/********************************************************************
*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Sift类的声明~~~~~~~~~~~~~~~~~~~~~~~~~~~~
********************************************************************/
/*
 * @property    图像Sift类
 * @func        Sift关键点、描述子特征提取等
 * @param
 * @param
 * @param
*/
class Sift
{
/********************************************************************
*~~~~~~~~~~~~~~~~~~~~~~~~~常用结构体声明及定义~~~~~~~~~~~~~~~~~~~~~~~~~~
********************************************************************/
public:
    /*
    * @property    图像Sift类
    * @func        Sift关键点、描述子特征提取等
    * @param
    * @param
    * @param
    */
    struct Options
    {
        //void 避免函数进行检测参数，加速
        Options(void);

        //每阶有效差分个数默认为3
        int num_samples_per_octave;

        //设置最小阶的ID.
        // 默认为0，即原始图像；
        // -1，原图上采样2倍；
        // >0，原图下采样min_octave*2 倍
        int min_octave;

        //设置最大阶ID，默认为4，也就是一共5阶分别是0,1,2,3,4,,下采样4次
        int max_octave;

        //消除边界响应，trace(H)^2/Det(H) < [(r+1)^2]/r 默认为10
        float edge_ratio_threshold;

        //基本图像的尺度，默认1.6
        float base_blur_sigma;

        //是否输出程序运行的状态信息到控制台
        bool verbose_output;

        //输出更多的状态信息到控制台
        bool debug_output;

    };

    struct Keypoint
    {
        //关键点的阶索引，第octave阶
        int octave;

        //有效高斯差分索引，{0,S-1} 高斯差分{-1,S}
        float sample;

        //关键点x坐标
        float x;

        //关键点y坐标
        float y;
    };

    struct Descriptor
    {
        //关键点亚像素x坐标
        float x;

        //关键点亚像素y的坐标
        float y;

        //关Options键点的尺度
        float scale;

        //关键点的方向：[0,2PI]
        float orientation;

        //描述子的数据，[0.0,1.0]
        matrix::Vector<float ,128> data;
    };
/********************************************************************
*~~~~~~~~~~~~~~~~~~~~~~~~~常用数据类型别名定义~~~~~~~~~~~~~~~~~~~~~~~~~~
********************************************************************/
public:
    typedef std::vector<image::Sift::Keypoint> Keypoints;
    typedef std::vector<image::Sift::Descriptor> Descriptors;

public:
    /*
    *  @property   Sift构造函数
    *  @func       为Sift特征分配内存空间，初始化Sift类变量
    *  @param_in   options          Sift特征的初始化数据选项
    *  @explict    显示转换，在构造函数中使用，防止非类对象隐式转换修改成员变量
    */
    explicit Sift(image::Sift::Options const &options);

    /*
    *  @property
    *  @func
    *  @param_in
    *  @explict
    */
    void set_image(image::ByteImage::ConstPtr img);


protected:


private:
    Options options;
    image::FloatImage::ConstPtr srcImg;
    Keypoints keypoints;
    Descriptors descriptors;
};



inline
Sift::Options::Options(void):
num_samples_per_octave(3),
min_octave(0),
max_octave(4),
edge_ratio_threshold(10.0f),
base_blur_sigma(1.6f),
verbose_output(false),
debug_output(false)
{

}

IMAGE_NAMESPACE_END
#endif //IMAGE_SIFT_H
