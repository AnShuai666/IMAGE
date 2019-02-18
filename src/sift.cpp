/*
 * @desc    SIFT特征
 * @author  安帅
 * @date    2019-1-17
 * @e-mail   1028792866@qq.com
*/

#include "include/sift.h"
#include "include/image_process.hpp"

IMAGE_NAMESPACE_BEGIN

image::Sift::Sift(image::Sift::Options const &options):
options(options)
{
    if (this->options.min_octave < -1 || this->options.min_octave > options.max_octave)
    {
        throw std::invalid_argument("非法图像阶范围！");
    }

    if (this->options.debug_output)
    {
        this->options.verbose_output = true;
    }
}

void
image::Sift::set_image(image::ByteImage::ConstPtr img)
{
    if (img->channels() != 1 && img->channels() != 3)
    {
        throw std::invalid_argument("需要灰度图或者彩色图");
    }

    this->srcImg = image::byte_to_float_image(img);

    if (img->channels() == 3)
    {
        this->srcImg = image::desaturate<float>(this->srcImg,image::DESATURATE_AVERAGE);
    }
}

void
image::Sift::set_float_image(image::FloatImage::ConstPtr img)
{
    if (img->channels() != 1 && img->channels() != 3)
    {
        throw std::invalid_argument("需要灰度图或者彩色图");
    }

    if(img->channels() == 3)
    {
        this->srcImg = image::desaturate<float>(img,image::DESATURATE_AVERAGE);
    }
    else
    {
        this->srcImg = img->duplicate();
    }
}

void
image::Sift::create_octaves()
{
    this->octaves.clear();

}


IMAGE_NAMESPACE_END