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
#include <jpeglib.h>
#include <string.h>

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
    try
    {
        return load_png_image_headers(filename);
    }
    catch (image::FileException &e)
    {
        e.what();
    }
    catch (image::Exception&)
    {}

    try
    {
        return load_jpg_image_headers(filename);
    }
    catch (image::FileException &e)
    {
        e.what();
    }
    catch (image::Exception&)
    {}


}

void
save_image(ByteImage::ConstPtr image, std::string const& filename)
{
    std::string 
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
        throw image::FileException(filename,"PNG文件打开异常");
        //TODO:写一个文件异常类，此处抛出异常
    }

    //89 50 4E 47 0D 0A 1A 0A 是PNG头部署名域，表示这是一个PNG图片
    png_byte signature[PNG_FILE_NAME_NUM];
    if (std::fread(signature,1,PNG_FILE_NAME_NUM,fp) != PNG_FILE_NAME_NUM)
    {
        std::fclose(fp);
        fp = NULL;
        throw image::FileException(filename,"PNG图的文件头不能读取");
    }

    //判断是否为png图
    int is_png = !png_sig_cmp(signature,0,8);
    if ( !is_png)
    {
        std::fclose(fp);
        fp = NULL;
        throw image::FileException(filename,"并不是PNG图");
    }

    //自定义图像头，包括长宽高、图像数据
    ImageHeaders imageHeaders;
    // png文件读取步骤：
    // 1.初始化png_structp类型指针
    // 2.初始化png_infop变量， 此变量存储了png图像数据的信息，如查阅图像信息，修改图像解码参数
    // 3.设置返回点
    png_structp png_ptr = nullptr;
    png_infop png_info = nullptr;

    // 1.初始化png_structp类型指针,png_create_read_struct()函数返回一个png_struct_p类型的指针，如果结构体指针分配失败，返回NULL
    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if(!png_ptr)
    {
        std::fclose(fp);
        fp =NULL;
        throw image::FileException(filename,"png_structp初始化失败！");
    }

    // 2.初始化png_infop变量
    png_info = png_create_info_struct(png_ptr);
    if(!png_info)
    {
        png_destroy_read_struct(&png_ptr, nullptr, nullptr);
        std::fclose(fp);
        fp = NULL;
        throw image::FileException(filename,"初始化png_infop失败！");
    }
    //复位文件指针
    rewind(fp);
    //开始读取文件，将文件与png_structp相关联
    png_init_io(png_ptr,fp);
    //由于文件头8个字节是png标识符，需要跳过
    png_set_sig_bytes(png_ptr,PNG_FILE_NAME_NUM);
    //查询图像信息 读取png图像信息头
    png_read_info(png_ptr,png_info);

    imageHeaders.width = png_get_image_width(png_ptr,png_info);
    imageHeaders.height = png_get_image_height(png_ptr,png_info);
    imageHeaders.channels = png_get_channels(png_ptr,png_info);

    int const bit_depth = png_get_bit_depth(png_ptr,png_info);
    if (bit_depth <= 8)
        imageHeaders.imageType = IMAGE_TYPE_UINT8;
    else if(bit_depth == 16)
        imageHeaders.imageType = IMAGE_TYPE_UINT16;
    else
    {
        png_destroy_read_struct(&png_ptr,&png_info, nullptr);
        std::fclose(fp);
        fp = NULL;
        throw image::FileException(filename,"PNG图像深度未知！");
    }

    //获取png图像颜色类型
    int const color_type = png_get_color_type(png_ptr,png_info);
    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png_ptr);//转换索引颜色到
    if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png_ptr);//图像位深强制转换为8位
    if(png_get_valid(png_ptr,png_info,PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png_ptr);// 将任何Trns数据块扩展成完整的alpha通道
    //更新转换后的png
    png_read_update_info(png_ptr,png_info);

    //生成自定义图像
    ByteImage::Ptr image = ByteImage::create();
    image->allocate(imageHeaders.width,imageHeaders.height,imageHeaders.channels);
    ByteImage::ImageData &data = image->get_data();

    //创建行指针向量
    std::vector<png_bytep> row_pointers;
    row_pointers.resize(imageHeaders.height);
    for (int i = 0; i < imageHeaders.height; ++i)
    {
        row_pointers[i] = &data[i * imageHeaders.width * imageHeaders.channels];
    }

    //读取图像，将png图数据读入指针向量中去
    png_read_image(png_ptr,&row_pointers[0]);

    png_destroy_read_struct(&png_ptr,&png_info, nullptr);
    std::fclose(fp);
    fp = NULL;
    return image;
}

ImageHeaders
load_png_image_headers(std::string const& filename)
{
    FILE *fp = std::fopen(filename.c_str(),"rb");
    if(fp == NULL)
    {
        std::fclose(fp);
        fp =NULL;
        image::FileException(filename,strerror(errno));
    }

    ImageHeaders imageHeaders;
    png_structp  png_ptr = NULL;
    png_infop png_info = NULL;

    png_byte signature[8];
    if (std::fread(signature,1,8,fp) != PNG_FILE_NAME_NUM)
    {
        std::fclose(fp);
        fp = NULL;
        throw image::FileException(filename,"PNG签名不能读取！");
    }

    //判断是否为png图
    int is_png = !png_sig_cmp(signature,0,8);
    if ( !is_png)
    {
        std::fclose(fp);
        fp = NULL;
        throw image::FileException(filename,"并不是PNG图");
    }

    // 1.初始化png_structp类型指针,png_create_read_struct()函数返回一个png_struct_p类型的指针，如果结构体指针分配失败，返回NULL
    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if(!png_ptr)
    {
        std::fclose(fp);
        fp =NULL;
        throw image::FileException(filename,"png_structp初始化失败！");
    }

    // 2.初始化png_infop变量
    png_info = png_create_info_struct(png_ptr);
    if(!png_info)
    {
        png_destroy_read_struct(&png_ptr, nullptr, nullptr);
        std::fclose(fp);
        fp = NULL;
        throw image::FileException(filename,"初始化png_infop失败！");
    }
    //复位文件指针
    rewind(fp);
    //开始读取文件，将文件与png_structp相关联
    png_init_io(png_ptr,fp);
    //由于文件头8个字节是png标识符，需要跳过
    png_set_sig_bytes(png_ptr,PNG_FILE_NAME_NUM);
    //查询图像信息 读取png图像信息头
    png_read_info(png_ptr,png_info);

    imageHeaders.width = png_get_image_width(png_ptr,png_info);
    imageHeaders.height = png_get_image_height(png_ptr,png_info);
    imageHeaders.channels = png_get_channels(png_ptr,png_info);

    int const bit_depth = png_get_bit_depth(png_ptr,png_info);
    if (bit_depth <= 8)
        imageHeaders.imageType = IMAGE_TYPE_UINT8;
    else if(bit_depth == 16)
        imageHeaders.imageType = IMAGE_TYPE_UINT16;
    else
    {
        png_destroy_read_struct(&png_ptr,&png_info, nullptr);
        std::fclose(fp);
        fp = NULL;
        throw image::FileException(filename,"PNG图像深度未知！");
    }

    png_destroy_read_struct(&png_ptr,&png_info, nullptr);

    std::fclose(fp);
    fp = NULL;
    return imageHeaders;
}

ByteImage::Ptr
load_jpg_image(std::string const& filename,std::string* exif )
{
    FILE *fp = std::fopen(filename.c_str(),"rb");
    if(fp == NULL)
    {
        fclose(fp);
        fp = NULL;
        throw image::FileException(filename,"jpg file open failed!");
    }

    // 分配和初始化一个d
    // ecompression结构体
    // 指定源文件
    // 用jpeg_read_header获得jpg信息
    // 设置解压参数,比如放大、缩小
    // 启动解压：jpeg_start_decompress
    // while (scan lines remain to be read)
    //   循环调用jpeg_read_scanlines
    // jpeg_finish_decompress
    // 释放decompression结构体

    jpeg_decompress_struct cinfo;
    jpeg_error_mgr jerr{};
    ByteImage::Ptr image;

    try
    {
        //分配和初始化一个decompression结构体
        cinfo.err = jpeg_std_error(&jerr);

        jpeg_create_decompress(&cinfo);

        //指定源文件
        jpeg_stdio_src (&cinfo,fp);

        //图像元信息
        if(exif)
        {
            jpeg_save_markers(&cinfo,JPEG_APP0 + 1,0xffff);
        }

        //读取jpeg头 获得jpg信息
        int ret = jpeg_read_header(&cinfo,FALSE);
        if (ret != JPEG_HEADER_OK)
        {
            throw image::FileException(filename,"JPEG header 识别不了！");
        }

        //检查jpeg marker标记
        if (exif)
        {

        }

        if(cinfo.out_color_space != JCS_GRAYSCALE && cinfo.out_color_space != JCS_RGB)
        {
            throw image::FileException(filename,"非法JPEG图像空间！");
        }

        //创建图像
        int const width = cinfo.image_width;
        int const height = cinfo.image_height;
        int const channels = (cinfo.out_color_space == JCS_RGB ? 3 : 1);
        image = ByteImage::create(width,height,channels);
        ByteImage::ImageData& data = image->get_data();
        //开始解压
        jpeg_start_decompress(&cinfo);

        unsigned char* data_ptr = &data[0];
        //循环调用jpeg_read_scanlines来一行一行地获得解压的数据
        while (cinfo.output_scanline < cinfo.output_height)
        {
            jpeg_read_scanlines(&cinfo,&data_ptr,1);
            data_ptr += channels * cinfo.output_width;
        }

        //停止图像解压
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        fclose(fp);
        fp = NULL;
    }

    catch (...)
    {
        jpeg_destroy_decompress(&cinfo);
        std::fclose(fp);
        fp =NULL;
    }

    return image;
}

IMAGE_NAMESPACE_END