/*
 * @desc    SIFT特征
 * @author  安帅
 * @date    2019-1-17
 * @e-mail   1028792866@qq.com
*/

#include "include/sift.h"

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

IMAGE_NAMESPACE_END