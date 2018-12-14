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
    IMAGE_TYPE_UNKNOWN,
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
    virtual ImageBase::Ptr duplicate_base() const;

    /*
    *  @property   获取图像宽度
    *  @func       获取图像的宽
    *  @return     int
    */

/********************************************************************
 *~~~~~~~~~~~~~~~~~~~~~~~ImageBase管理函数~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *******************************************************************/
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

    /*
    *  @property   判断图像是否合法
    *  @func       如果图像w,h,c任意为0，则返回false
    *  @return     bool
    */
    bool valid() const;

    /*
    *  @property   重定义图像尺寸
    *  @func       将图像长宽通道数改变，但是三者乘积需与原来乘积相同，否则失败
    *  @return     bool
    */
    bool reinterpret(int new_w,int new_h,int new_c);

    /*
    *  @property   获取图像字节数
    *  @func       虚函数，具体实现需要在子类中进行
    *  @return     std::size_t  在子类中实现时返回图像字节数，否则返回0
    */
    virtual std::size_t get_byte_size() const;

    /*
    *  @property   获取图像数据指针
    *  @func       虚函数，具体实现需要在子类中进行重载
    *  @const1     指针指向内容不能变，也就是图像数据
    *  @const2     防止改变类成员变量
    *  @return     char const *  在子类中实现时返回图像指针，否则返回nullptr
    */
    virtual char const *get_byte_pointer() const;

    /*
    *  @property   获取图像数据指针
    *  @func       虚函数，具体实现需要在子类中进行重载
    *  @return     char *  在子类中实现时返回图像指针，否则返回nullptr
    */
    virtual char *get_byte_pointer();

    /*
    *  @property   获取图像数据类型
    *  @func       虚函数，具体实现需要在子类中进行重载
    *  @return    ImageType  在子类中实现时返回图像枚举类型，否则返回IMAGE_TYPE_UNKNOW
    */
    virtual ImageType get_type() const;

    /*
    *  @property   获取图像数据类型
    *  @func       虚函数，具体实现需要在子类中进行重载
    *  @return     char const*  在子类中实现时返回图像类型，否则返回IMAGE_TYPE_UNKNOW
    */
    virtual char const* get_type_string() const;

    /*
    *  @property   获取图像数据类型
    *  @func       虚函数，具体实现需要在子类中进行重载
    *  @return     ImageType 在子类中实现时返回图像类型，否则返回IMAGE_TYPE_UNKNOW string是什么就返回什么图像枚举类型
    */
    virtual ImageType get_type_for_string(std::string const& type_string);


protected:
    int w;
    int h;
    int c;

};

/********************************************************************
*~~~~~~~~~~~~~~~~~~~~~~~~~Image 类的声明~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
*******************************************************************/
/*
 * @func   有类型图像基类
*/
template <typename T>
class TypedImageBase : public ImageBase
{

 /********************************************************************
 *~~~~~~~~~~~~~~~~~~~~~~TypedImageBase常用容器别名定义~~~~~~~~~~~~~~~~~~
 *******************************************************************/
public:
    typedef T ValueType;
    typedef std::shared_ptr<TypedImageBase<T>> Ptr;
    typedef std::shared_ptr<TypedImageBase<T> const> ConstPtr;
    typedef std::vector<T> ImageData;

public:
    TypedImageBase();
    TypedImageBase(TypedImageBase<T> const& typedImageBase1);
    virtual ~TypedImageBase();
    virtual ImageBase::Ptr duplicate_base() const;

    virtual void clear();

    void resize(int width, int height, int channels);

    void allocate(int width,int height, int channels);

    void fill(T const& value);

    void swap(TypedImageBase<T>& typedImageBase1);



    ImageData const& get_data() const;

    ImageData& get_data();

    T const* get_data_pointer() const;

    T* get_data_pointer();

    virtual ImageType get_type() const;

    char const* get_type_string() const;

    T* begin();

    T const* begin() const;

    T* end();

    T const* end() const;

    int get_pixel_amount() const;

    int get_value_amount() const;

    std::size_t get_byte_size() const;

    char const* get_byte_pointer() const;

    char* get_byte_pointer();


protected:
    ImageData data;

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

    inline bool
    ImageBase::valid() const
    {
        return this->w && this->h && this->c;
    }

    inline bool
    ImageBase::reinterpret(int new_w, int new_h, int new_c)
    {
        if(new_w * new_h * new_c != this->w * this->h * this->c)
        {
            return false;
        }
        this->w = new_w;
        this->h = new_h;
        this->c = new_c;
        return true;
    }

    inline std::size_t
    ImageBase::get_byte_size() const
    {
        return 0;
    }

    inline char const*
    ImageBase::get_byte_pointer() const
    {
        return nullptr;
    }

    inline char *
    ImageBase::get_byte_pointer()
    {
        return nullptr;
    }

    inline ImageType
    ImageBase::get_type() const
    {
        return IMAGE_TYPE_UNKNOWN;
    }

    inline char const*
    ImageBase::get_type_string() const
    {
        return "unknown";
    }


    inline ImageType
    ImageBase::get_type_for_string(std::string const &type_string)
    {
        if (type_string == "sint8")
            return IMAGE_TYPE_SINT8;
        else if (type_string == "sint16")
            return IMAGE_TYPE_SINT16;
        else if (type_string == "sint32")
            return IMAGE_TYPE_SINT32;
        else if (type_string == "sint64")
            return IMAGE_TYPE_SINT64;
        else if (type_string == "uint8")
            return IMAGE_TYPE_UINT8;
        else if (type_string == "uint16")
            return IMAGE_TYPE_UINT16;
        else if (type_string == "uint32")
            return IMAGE_TYPE_UINT32;
        else if (type_string == "uint64")
            return IMAGE_TYPE_UINT64;
        else if (type_string == "float")
            return IMAGE_TYPE_FLOAT;
        else if (type_string == "double")
            return IMAGE_TYPE_DOUBLE;
        else
            return IMAGE_TYPE_UNKNOWN;
    }



/********************************************************************
*~~~~~~~~~~~~~~~~~~~~~TypedImageBase成员函数实现~~~~~~~~~~~~~~~~~~~~~~~
********************************************************************/

    template <typename T>
    inline
    TypedImageBase<T>::TypedImageBase()
    {

    }

    template <typename T>
    inline
    TypedImageBase<T>::TypedImageBase(TypedImageBase<T> const& typedImageBase1)
        : ImageBase(typedImageBase1),data(typedImageBase1.data)
    {

    }

    template <typename T>
    inline
    TypedImageBase<T>::~TypedImageBase()
    {

    }

    template <typename T>
    inline ImageBase::Ptr
    TypedImageBase<T>::duplicate_base() const
    {
        return ImageBase::Ptr(new TypedImageBase<T>(*this));
    }

    template <typename T>
    inline void
    TypedImageBase<T>::clear()
    {
        this->w = 0;
        this->h = 0;
        this->c = 0;
        this->data.clear();
    }

    template <typename T>
    inline void
    TypedImageBase<T>::resize(int width, int height, int channels)
    {
        this->w = width;
        this->h = height;
        this->c = channels;
        data.resize(width*height*channels);
    }

    template <typename T>
    inline void
    TypedImageBase<T>::allocate(int width, int height, int channels)
    {
        this->clear();
        this->resize(width,height,channels);
    }

    template <typename T>
    inline void
    TypedImageBase<T>::fill(const T &value)
    {
        std::fill(this->data.begin(),this,data.end(),value);
    }

    template <typename T>
    inline void
    TypedImageBase<T>::swap(image::TypedImageBase<T> &typedImageBase1)
    {
        std::swap(this->w,typedImageBase1->w);
        std::swap(this->h,typedImageBase1->h);
        std::swap(this->c,typedImageBase1->c);
        std::swap(this->data,typedImageBase1.data);
    }

    template <typename T>
    inline typename TypedImageBase<T>::ImageData const&
    TypedImageBase<T>::get_data() const
    {
        return this->data;
    }

    template <typename T>
    inline typename TypedImageBase<T>::ImageData&
    TypedImageBase<T>::get_data()
    {
        return this->data;
    }

    template <typename T>
    inline T const*
    TypedImageBase<T>::get_data_pointer() const
    {

    }

    template <typename T>
    inline T *
    TypedImageBase<T>::get_data_pointer()
    {

    }


    

    virtual ImageType get_type() const;

    char const* get_type_string() const;

    T* begin();

    T const* begin() const;

    T* end();

    T const* end() const;

    int get_pixel_amount() const;

    int get_value_amount() const;

    std::size_t get_byte_size() const;

    char const* get_byte_pointer() const;

    char* get_byte_pointer();





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
