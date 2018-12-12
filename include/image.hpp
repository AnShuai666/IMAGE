/*
 * @desc    图像处理类
 * @author  安帅
 * @date    2018-12-12
 * @email   1028792866@qq.com
*/

#include "define.h"
#include <memory>
#include <vector>

#ifndef IMAGE_IMAGE_HPP
#define IMAGE_IMAGE_HPP

IMAGE_NAMESPACE_BEGIN
/********************************************************************
 *~~~~~~~~~~~~~~~~~~~~~常用数据类型别名声明~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *******************************************************************/
typedef unsigned char   uint8_t;
typedef unsigned short  uint16_t;
/********************************************************************
 *~~~~~~~~~~~~~~~~~~~~~常用矩阵类型别名声明~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *******************************************************************/
template <typename T> class Image;
typedef Image<uint8_t >     ByteImage;
typedef Image<uint16_t>     RawImage;
typedef Image<char>         CharImage;
typedef Image<float>        FloatImage;
typedef Image<double>       DoubleImage;
typedef Image<int>          IntImage;


/********************************************************************
 *~~~~~~~~~~~~~~~~~~~~~~~~~Image 类的声明~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *******************************************************************/
 /*
  * RGB图，通道分离
  * */
template <typename T>
class Image
{

/********************************************************************
 *~~~~~~~~~~~~~~~~~~~~~~~~~Image常用容器别名定义~~~~~~~~~~~~~~~~~~~~~~~~
 *******************************************************************/
public:
   typedef std::shared_ptr<Image<T>> Ptr;
   typedef std::shared_ptr<Image<T> const> ConstPtr;
   typedef std::vector<T>  ImageData;
   typedef T ValueType;

/********************************************************************
 *~~~~~~~~~~~~~~~~~~~~~~~Image构造函数与析构函数~~~~~~~~~~~~~~~~~~~~~~~~
 *******************************************************************/
 public:
    Image();

    Image(int width, int height, int channels);

    Image(Image<T> const& image1);

    static Ptr create();

    static Ptr create(int width, int height, int channels);

    static Ptr create(Image<T> const&image1);

    Ptr duplicate() const;

    void fill_color(T )




};

IMAGE_NAMESPACE_END

IMAGE_NAMESPACE_BEGIN

template <typename T>
inline
Image<T>::Image()
{

}

template <typename T>
inline
Image<T>::Image(int width, int height, int channels)
{

}

template <typename T>
inline
Image<T>::Image(Image<T> const& image1)
{

}
template <typename T>
inline Ptr
Image<T>::create()
{

}

template <typename T>
inline static Ptr
Image<T>::create(int width, int height, int channels)
{

}

static Ptr create(Image<T> const&image1);

template <typename T>
inline Ptr
Image<T>::duplicate() const
{

}

IMAGE_NAMESPACE_END

#endif //IMAGE_IMAGE_HPP
