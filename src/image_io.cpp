/*
 * @desc    图像输入输出读取等接口实现
 * @author  安帅
 * @date    2018-12-18
 * @e-mail   1028792866@qq.com
*/
#include "image_io.h"
#include <fstream>
#include <png.h>
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

ByteImage::Ptr
load_png_image(std::string const& filename)
{
    FILE* fp = std::fopen(filename.c_str(),"rb");
    if (!fp)
    {
        std::exit(0);
        //TODO:写一个文件异常类，此处抛出异常
    }

    ImageHeaders imageHeaders;
    png_structp png = nullptr;
    png_infop png_info = nullptr;
    loa
}

IMAGE_NAMESPACE_END