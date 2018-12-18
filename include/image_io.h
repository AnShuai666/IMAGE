/*
 * @desc    图像输入输出读取等接口
 * @author  安帅
 * @date    2018-12-18
 * @e-mail   1028792866@qq.com
*/

#ifndef IMAGE_IMAGE_IO_H
#define IMAGE_IMAGE_IO_H
#include "define.h"
#include "image.hpp"
using namespace image;
IMAGE_NAMESPACE_BEGIN


/*******************************************************************
*~~~~~~~~~~~~~~~~~~~~~图像元数据结构体声明~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*******************************************************************/
struct ImageHeaders
{
    int width;
    int height;
    int channels;
    ImageType imageType;
};
/*******************************************************************
*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~图像加载~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*******************************************************************/

ByteImage::Ptr
load_file(std::string& filename);

IMAGE_NAMESPACE_END
#endif //IMAGE_IMAGE_IO_H
