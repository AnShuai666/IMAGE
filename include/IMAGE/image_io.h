/*
 * @desc    图像输入输出读取等接口
 * @author  安帅
 * @date    2018-12-18
 * @e-mail   1028792866@qq.com
*/

#ifndef IMAGE_IMAGE_IO_H
#define IMAGE_IMAGE_IO_H
//#include "define.h"
#include "image.hpp"
using namespace image;
IMAGE_NAMESPACE_BEGIN
#define PNG_FILE_NAME_NUM 8

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

/*
*  @property   图像加载
*  @func       将图像filename加载进来,根据图像不同格式调用不同格式的加载函数
*  @param_in   filename         图像路径及图像名
*  @return     ByteImage::ImagePtr   图像类型为uint8的ByteImage，
*/
ByteImage::ImagePtr
loadImage(std::string& filename);

/*
*  @property   图像元数据加载
*  @func       加载图像元数据,根据图像不同格式调用不同格式的加载函数
*  @param_in   filename         图像路径及图像名
*  @return     ImageHeaders     图像元数据结构体
*/
ImageHeaders
loadImageHeaders(std::string const filename);

/*
*  @property   图像保存
*  @func       保存图像,根据图像不同格式调用不同格式的保存函数，原图数据类型：uint8
*  @param_in   image            需要保存的图像      常指针
*  @param_in   filename         图像路径及图像名
*  @return     void
*/

/*******************************************************************
*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~图像保存~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*******************************************************************/

void
saveImage(ByteImage::ConstImagePtr image, std::string const& filename);

/*
*  @property   图像保存
*  @func       保存图像,根据图像不同格式调用不同格式的保存函数，原图数据类型：uint8
*  @param_in   image            需要保存的图像      常指针
*  @param_in   filename         图像路径及图像名
*  @return     void
*/
void saveImage(ByteImage::ImagePtr image, std::string const& filename);

/*
*  @property   图像保存
*  @func       保存图像,根据图像不同格式调用不同格式的保存函数，原图数据类型：float
*  @param_in   image            需要保存的图像      常指针
*  @param_in   filename         图像路径及图像名
*  @return     void
*/
void saveImage(FloatImage::ConstImagePtr image, std::string const& filename);

/*
*  @property   图像保存
*  @func       保存图像,根据图像不同格式调用不同格式的保存函数，原图数据类型：float
*  @param_in   image            需要保存的图像      常指针
*  @param_in   filename         图像路径及图像名
*  @return     void
*/
void saveImage(FloatImage::ImagePtr image, std::string const& filename);

/*******************************************************************
*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~png图像加载~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*******************************************************************/

/*
*  @property   图像加载
*  @func       加载png格式的图像
                通道数           图像类型
                  1              gray
                  2            gray-alpha
                  3               RGB
                  4               RGBA
*  @param_in   filename         图像路径及图像名
*  @return     ByteImage::ImagePtr   图像类型为uint8的ByteImage
*/
ByteImage::ImagePtr
loadPngImage(std::string const& filename);


ImageHeaders
loadPngImageHeaders(std::string const& filename);

/*
*  @property   图像存储
*  @func       存储png格式的图像
*  @param_in   image                待存储图像
*  @param_in   filename             图像路径及图像名        常指针
*  @param_in   compression_level    0-9 0:无压缩 9：最大压缩 常用3-6
*  @return     void
*/
void
savePngImage(ByteImage::ConstImagePtr image, std::string const& filename, int compression_level = 1);

/*******************************************************************
*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~jpg图像加载~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*******************************************************************/

/*
*  @property   图像加载
*  @func       加载jpg格式的图像
*  @param_in   filename         图像路径及图像名        常指针
*  @param_in   exif             图像元信息
*  @return     ByteImage::ImagePtr
*/
ByteImage::ImagePtr
loadJpgImage(std::string const& filename,std::string* exif = nullptr);


ImageHeaders
loadJpgImageHeaders(std::string const& filename);

/*
*  @property   图像存储
*  @func       存储jpg格式的图像
*  @param_in   image             待存储图像
*  @param_in   filename          图像路径及图像名        常指针
*  @param_in   quality           图像压缩质量，一般在80%以上比较清晰
*  @return     void
*/
void
saveJpgImage(ByteImage::ConstImagePtr image, std::string const& filename, int quality);


/*******************************************************************
*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~tiff图像加载~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*******************************************************************/

ByteImage::ImagePtr
loadTiffImage(std::string const& filename);

RawImage::ImagePtr
loadTiff16Image(std::string const& filename);

void
saveTiffImage(ByteImage::ConstImagePtr image, std::string const& filename);


IMAGE_NAMESPACE_END
#endif //IMAGE_IMAGE_IO_H
