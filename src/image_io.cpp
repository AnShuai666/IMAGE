/*
 * @desc    图像输入输出读取等接口实现
 * @author  安帅
 * @date    2018-12-18
 * @e-mail   1028792866@qq.com
*/
#include "image_io.h"
#include <fstream>
#include <png.h>
#include <exception.h>

IMAGE_NAMESPACE_BEGIN

ByteImage::Ptr
load_image(std::string& filename)
{
    try
    {
        return load_png_image(filename);
    }
    catch(image::FileException &e)
    {
        e.what();
    }

    try
    {
        return load_jpg_image(filename);
    }
    catch(image::FileException &e)
    {
        e.what();
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
        checkFileerror(fp);
        std::fclose(fp);
        fp = NULL;
        throw FileException(filename);
        //TODO:写一个文件异常类，此处抛出异常
    }

    //89 50 4E 47 0D 0A 1A 0A 是PNG头部署名域，表示这是一个PNG图片
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

    ImageHeaders imageHeaders;
    png_structp png = nullptr;
    png_infop png_info = nullptr;





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