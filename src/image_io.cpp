/*
 * @desc    图像输入输出读取等接口实现
 * @author  安帅
 * @date    2018-12-18
 * @e-mail   1028792866@qq.com
*/
#include "image_io.h"
IMAGE_NAMESPACE_BEGIN

ByteImage::Ptr
load_image(std::string& filename)
{

}

ImageHeaders
load_image_headers(std::string const filename)
{

}

void
save_image(ByteImage::ConstPtr image, std::string const& filename)
{

}

void save_image(ByteImage::Ptr image, std::string const& filename)
{

}

void save_image(FloatImage::ConstPtr image, std::string const& filename)
{

}

void save_image(FloatImage::Ptr image, std::string const& filename)
{

}

IMAGE_NAMESPACE_END