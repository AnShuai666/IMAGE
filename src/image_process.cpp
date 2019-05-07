/*
 * @desc    图像处理函数
 * @author  安帅
 * @date    2019-01-22
 * @email   1028792866@qq.com
*/

#include "IMAGE/image_process.hpp"
#include "IMAGE/image.hpp"
IMAGE_NAMESPACE_BEGIN
FloatImage::Ptr
byte_to_float_image(image::Image<unsigned char>::ConstPtr image)
{
    if (image == NULL)
    {
        throw std::invalid_argument("无图像输入！");
    }

    FloatImage::Ptr img = FloatImage::create();
    img->allocate(image->width(),image->height(),image->channels());
    //TODO::to be CUDA @YANG;
    //opencv 4000*2250*3 图像处理时间: 29.4ms
    for (int i = 0; i < image->get_value_amount(); ++i)
    {
        float value = image->at(i) / 255.0f;
        img->at(i) = value;
    }
    return img;
}

IMAGE_NAMESPACE_END