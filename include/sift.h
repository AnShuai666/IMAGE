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

        //
        int min_octave;

        int max_octave;

        float

    };

    struct Keypoint
    {

    };

    struct Descriptor
    {

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

protected:


private:

};

IMAGE_NAMESPACE_END
#endif //IMAGE_SIFT_H
