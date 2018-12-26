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
    if()
    {
        return load_png_image(filename);
    }
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
        checkImageioerror(fp);
        std::exit(0);
        //TODO:写一个文件异常类，此处抛出异常
    }

    ImageHeaders imageHeaders;
    png_structp png = nullptr;
    png_infop png_info = nullptr;

    png_byte signature[PNG_FILE_NAME_NUM];
    if (std::fread(signature,1,PNG_FILE_NAME_NUM,fp) != PNG_FILE_NAME_NUM)
    {

        std::fclose(fp);
        fp = NULL;
    }

    int is_png = !png_sig_cmp(signature,0,8);
    if ( !is_png)
    {

        std::fclose(fp);
        fp = NULL;
    }

    png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if(!png)
    {
        std::fclose(fp);
        fp =NULL;
    }

    png_info = png_create_info_struct(png);

    if(!png_info)
    {
        png_destroy_read_struct(&png, nullptr, nullptr);
    }
    png_init_io(png,fp);
    std::fclose(fp);
    return ;
}

IMAGE_NAMESPACE_END