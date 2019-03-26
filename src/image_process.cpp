/*
 * @desc    图像处理函数
 * @author  安帅
 * @date    2019-01-22
 * @email   1028792866@qq.com
*/

#include "include/image_process.hpp"
#include "include/image.hpp"
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
    for (int i = 0; i < image->get_value_amount(); ++i)
    {
        float value = image->at(i) / 255.0f;
        img->at(i) = value;
    }
    return img;
}

IMAGE_NAMESPACE_END