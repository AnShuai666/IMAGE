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
/*******************************************************************
*~~~~~~~~~~~~~~~~~~~~~常用数据类型别名声明~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*******************************************************************/
typedef unsigned char   uint8_t;
typedef unsigned short  uint16_t;
enum ImageType
{
    IMAGE_TYPE_UNKNOW,
    IMAGE_TYPE_UINT8,
    IMAGE_TYPE_UINT16,
    IMAGE_TYPE_UINT32,
    IMAGE_TYPE_UINT64,
    IMAGE_TYPE_SINT8,
    IMAGE_TYPE_SINT16,
    IMAGE_TYPE_SINT32,
    IMAGE_TYPE_SINT64,
    IMAGE_TYPE_FLOAT,
    IMAGE_TYPE_DOUBLE
};

/********************************************************************
*~~~~~~~~~~~~~~~~~~~~~~~~~~ImageBase类的声明~~~~~~~~~~~~~~~~~~~~~~~~~~
********************************************************************/
/*
 * @property    图像基类
 * @func        数据访问等
 * @param       width
 * @param       height
 * @param       channel
*/
class ImageBase
{
/*********************************************************************
*~~~~~~~~~~~~~~~~~~~~~~ImageBase常用容器别名定义~~~~~~~~~~~~~~~~~~~~~~~~
********************************************************************/
public:
    typedef std::shared_ptr<ImageBase> Ptr;
    typedef std::shared_ptr<ImageBase const> ConstPtr;
/********************************************************************
 *~~~~~~~~~~~~~~~~~~~~~~~ImageBase构造函数与析构函数~~~~~~~~~~~~~~~~~~~~
 *******************************************************************/
public:
    /*
    *  @property   默认构造函数
    *  @func       将图像进行初始化，w=h=c=0
    */
    ImageBase();

    /*
    *  @property   默认构造函数
    *  @func        对图像进行析构
    */
    virtual ~ImageBase();

    /*
    *  @property   图像复制
    *  @func       为图像动态分配内存，并以该内存区域对共享指针进行初始化
    *  @return     Ptr
    */
    virtual Ptr duplicate_base() const;

    /*
    *  @property   获取图像宽度
    *  @func       获取图像的宽
    *  @return     int
    */
    int width() const;

    /*
    *  @property   获取图像高度
    *  @func       获取图像的高
    *  @return     int
    */
    int height() const;

    /*
    *  @property   获取图像通道数
    *  @func       获取图像通道数
    *  @return     int
    */
    int channels() const;

protected:
    int w;
    int h;
    int c;
};

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
*~~~~~~~~~~~~~~~~~~~~~~~~~Image 类的声明~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
*******************************************************************/
 /*
  * @func   RGB图，通道分离
 */
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
    /*
    *  @property   默认构造函数
    *  @func       将图像进行初始化，w=h=c=0
    */
    Image();

    Image(int width, int height, int channels);

    Image(Image<T> const& image1);

    static Ptr create();

    static Ptr create(int width, int height, int channels);

    static Ptr create(Image<T> const&image1);

  /*******************************************************************
  *~~~~~~~~~~~~~~~~~~~~~~~~~Image管理函数~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  *******************************************************************/

    Ptr duplicate() const;

    void fill_color(T )




};

IMAGE_NAMESPACE_END

IMAGE_NAMESPACE_BEGIN
/********************************************************************
*~~~~~~~~~~~~~~~~~~~~~ImageBase成员函数实现~~~~~~~~~~~~~~~~~~~~~~~~~~~~
********************************************************************/

    inline
    ImageBase::ImageBase()
            : w(0),h(0),c(0)
    {

    }

    inline
    ImageBase::~ImageBase()
    {

    }

    inline ImageBase::Ptr
    ImageBase::duplicate_base() const
    {
        return ImageBase::Ptr(new ImageBase(*this));
    }

    inline int
    ImageBase::width() const
    {
        return this->w;
    }

    inline int
    ImageBase::height() const
    {
        return this->h;
    }

    inline int
    ImageBase::channels() const
    {
        return this->c;
    }


/********************************************************************
*~~~~~~~~~~~~~~~~~~~~~Image成员函数实现~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
********************************************************************/
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
    inline typename Image<T>::Ptr
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
