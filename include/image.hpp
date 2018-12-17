/*
 * @desc    图像处理类
 * @author  安帅
 * @date    2018-12-12
 * @email   1028792866@qq.com
*/

#include "define.h"
#include <memory>
#include <vector>
#include <algorithm>

#ifndef IMAGE_IMAGE_HPP
#define IMAGE_IMAGE_HPP

IMAGE_NAMESPACE_BEGIN
/*******************************************************************
*~~~~~~~~~~~~~~~~~~~~~图像数据类型枚举声明~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*******************************************************************/
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




    int get_pixel_amount() const;

    int get_value_amount() const;

    std::size_t get_byte_size() const;

    //用reinterpret_cast没有数位丢失
    char const* get_byte_pointer() const;

    char* get_byte_pointer();

    virtual char const* get_type_string() const;

    virtual ImageType get_type() const;



    T* begin();

    T const* begin() const;

    T* end();

    T const* end() const;



protected:
    ImageData data;

};



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
class Image : public TypedImageBase<T>
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

     /*
     *  @property   普通构造函数
     *  @func       将图像进行初始化,设置宽高与通道数
     *  @param_in   width
     *  @param_in   height
     *  @param_in   channels
     */
    Image(int width, int height, int channels);

     /*
     *  @property   拷贝构造函数
     *  @func       将图像进行初始化,
     *  @param_in   image1
     */
    Image(Image<T> const& image1);

     /*
     *  @property   智能指针构造函数
     *  @func       为图像动态分配内存，并赋给智能指针
     *  @static     静态成员函数在类加载时就会分配内存，可以通过类名直接访问，
     *              调用此函数不会访问或修改任何非static数据成员。
     *              用static修饰的函数，限定在本源码文件中，不能被本源码
     *              文件以外的代码文件调用
     *  @return     static Ptr
     */
     static Ptr create();

    /*
    *  @property   智能指针构造函数
    *  @func       为图像动态分配内存，并赋给智能指针
    *  @static     静态成员函数在类加载时就会分配内存，可以通过类名直接访问，
    *              调用此函数不会访问或修改任何非static数据成员。
    *              用static修饰的函数，限定在本源码文件中，不能被本源码
    *              文件以外的代码文件调用
    *  @return     static Ptr
    */
     static Ptr create(int width, int height, int channels);

     /*
     *  @property   智能指针构造函数
     *  @func       为图像动态分配内存，并赋给智能指针
     *  @static     静态成员函数在类加载时就会分配内存，可以通过类名直接访问，
     *              调用此函数不会访问或修改任何非static数据成员。
     *              用static修饰的函数，限定在本源码文件中，不能被本源码
     *              文件以外的代码文件调用
     *  @return     static Ptr
     */
     static Ptr create(Image<T> const&image1);

  /*******************************************************************
  *~~~~~~~~~~~~~~~~~~~~~~~~~Image管理函数~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  *******************************************************************/

    /*
    *  @property   复制图先锋
    *  @func       复制图像
    *  @return     Ptr
    */
    Ptr duplicate() const;

    /*
    *  @property   填充图像
    *  @func       为每一个像素填充color数组的颜色数组
    *  @param_in    T const* color      颜色数组
    *  @return     void
    */
    void fill_color(T const* color);

    /*
    *  @property    给图像添加通道
    *  @func        为图像添加num_channels的通道，值为value
    *  @param_in    num_channels    要添加的通道数
    *  @param_in    value           新通道分量的值
    *  @return      void
    */
    void add_channels(int num_channels, T const& value = T(0));

    /*
    *  @property   复制图先锋
    *  @func       复制图像
    *  @return     Ptr
    */
    void swap_channels(int channel1, int channel2);

    /*
    *  @property   复制图先锋
    *  @func       复制图像
    *  @return     Ptr
    */
    void copy_channel(int src,int dest);

    /*
    *  @property   复制图先锋
    *  @func       复制图像
    *  @return     Ptr
    */
    void delete_channel(int channel);

    /*
    *  @property   访问图像数据
    *  @func       线性访问图像数据
    *  @param_in   index    图像数据线性索引值
    *  @return     T const&
    */
    T const& at(int index) const;

    /*
    *  @property   访问图像数据
    *  @func       线性访问图像数据
    *  @param_in   index    图像像素线性索引值
    *  @param_in   channel  待访问像素通道索引值
    *  @return     T const&
    */
    T const& at(int index, int channel) const;

     /*
     *  @property   访问图像数据
     *  @func       二维索引访问图像数据  更加耗时
     *  @param_in   x    图像像素x方向索引值
     *  @param_in   y    图像像素x方向索引值
     *  @param_in   channel
     *  @return     T const&
     */
    T const& at(int x, int y, int channel) const;

    /*
    *  @property   访问图像数据
    *  @func       线性访问图像数据
    *  @param_in   index    图像数据线性索引值
    *  @return     T &
    */
    T& at(int index);

    /*
    *  @property   访问图像数据
    *  @func       线性访问图像数据
    *  @param_in   index    图像像素线性索引值
    *  @param_in   channel  待访问像素通道索引值
    *  @return     T const&
    */
    T& at(int index, int channel);

    /*
    *  @property   访问图像数据
    *  @func       二维索引访问图像数据  更加耗时
    *  @param_in   x    图像像素x方向索引值
    *  @param_in   y    图像像素x方向索引值
    *  @param_in   channel
    *  @return     T const&
    */
    T& at(int x, int y, int channel);

     /*
 *  @property   复制图先锋
 *  @func       复制图像
 *  @return     Ptr
 */
    T linear_at(float x, float y, int channel) const;

     /*
 *  @property   复制图先锋
 *  @func       复制图像
 *  @return     Ptr
 */
    void linear_at(float x, float y, T* px) const;

    /*
    *  @property   重载运算符[]
    *  @func       访问图像数据
    *  @param_in   index    图像数据线性索引值
    *  @return     T const&
    */
    T const& operator[] (int index) const;

    /*
    *  @property   重载运算符[]
    *  @func       访问图像数据
    *  @param_in   index    图像数据线性索引值
    *  @return     T&
    */
     T& operator[] (int index);

    /*
    *  @property   重载运算符()
    *  @func       访问图像数据
    *  @param_in   index    图像数据线性索引值
    *  @return     T const&
    */
    T const& operator() (int index) const;

    /*
    *  @property   重载运算符()
    *  @func       访问图像数据
    *  @param_in   index    图像像素索引值
    *  @param_in   channel  图像像素通道索引值
    *  @return     T const&
    */
    T const& operator() (int index, int channel) const;

    /*
    *  @property   重载运算符()
    *  @func       访问图像数据
    *  @param_in   x        图像像素x方向索引值
    *  @param_in   y        图像像素y方向索引值
    *  @param_in   channel  图像像素通道索引值
    *  @return     T const&
    */
    T const& operator() (int x, int y, int channel) const;

    /*
    *  @property   重载运算符()
    *  @func       访问图像数据
    *  @param_in   index    图像数据线性索引值
    *  @return     T&
    */
    T& operator()(int index);

    /*
    *  @property   重载运算符()
    *  @func       访问图像数据
    *  @param_in   index    图像像素索引值
    *  @param_in   channel  图像像素通道索引值
    *  @return     T&
    */
    T& operator()(int index, int channel);

    /*
    *  @property   重载运算符()
    *  @func       访问图像数据
    *  @param_in   x        图像像素x方向索引值
    *  @param_in   y        图像像素y方向索引值
    *  @param_in   channel  图像像素通道索引值
    *  @return     T&
    */
    T& operator()(int x, int y, int channel);

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
        std::fill(this->data.begin(),this->data.end(),value);
    }

    template <typename T>
    inline void
    TypedImageBase<T>::swap(image::TypedImageBase<T> &typedImageBase1)
    {
        std::swap(this->w,typedImageBase1.w);
        std::swap(this->h,typedImageBase1.h);
        std::swap(this->c,typedImageBase1.c);
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
        if (this->data.empty())
        {
            return nullptr;
        }
        return &this->data[0];
    }

    template <typename T>
    inline T *
    TypedImageBase<T>::get_data_pointer()
    {
        if (this->data.empty())
        {
            return nullptr;
        }
        return &this->data[0];
    }

    template <typename T>
    inline int
    TypedImageBase<T>::get_pixel_amount() const
    {
        return this->w * this->h;
    }

    template <typename T>
    inline int
    TypedImageBase<T>::get_value_amount() const
    {
        return this->get_pixel_amount() * this->c;
    }

    template <typename T>
    inline size_t
    TypedImageBase<T>::get_byte_size() const
    {
        return this->get_value_amount() * sizeof(T);
    }

    template <typename T>
    inline char const*
    TypedImageBase<T>::get_byte_pointer() const
    {
        return reinterpret_cast<char const*>(this->get_data_pointer());
    }

    template <typename T>
    inline char *
    TypedImageBase<T>::get_byte_pointer()
    {
        return reinterpret_cast<char *>(this->get_data_pointer());
    }

    template <typename T>
    inline char const*
    TypedImageBase<T>::get_type_string() const
    {
        return "unknown";
    }

    template <>
    inline char const*
    TypedImageBase<int8_t>::get_type_string() const
    {
        return "sint8";
    }

    template <>
    inline char const*
    TypedImageBase<char>::get_type_string() const
    {
        return "sint8";
    }

    template <>
    inline char const*
    TypedImageBase<int16_t>::get_type_string() const
    {
        return "sint16";
    }

    template <>
    inline char const*
    TypedImageBase<int32_t>::get_type_string() const
    {
        return "sint32";
    }

    template <>
    inline char const*
    TypedImageBase<int64_t>::get_type_string() const
    {
        return "sint64";
    }

    template <>
    inline char const*
    TypedImageBase<uint8_t>::get_type_string() const
    {
        return "uint8";
    }

    template <>
    inline char const*
    TypedImageBase<uint16_t>::get_type_string() const
    {
        return "uint16";
    }

    template <>
    inline char const*
    TypedImageBase<uint32_t>::get_type_string() const
    {
        return "uint32";
    }

    template <>
    inline char const*
    TypedImageBase<uint64_t>::get_type_string() const
    {
        return "uint64";
    }

    template <>
    inline char const*
    TypedImageBase<float>::get_type_string() const
    {
        return "float";
    }

    template <>
    inline char const*
    TypedImageBase<double>::get_type_string() const
    {
        return "double";
    }

    template <typename T>
    inline ImageType
    TypedImageBase<T>::get_type() const
    {
        return IMAGE_TYPE_UNKNOWN;
    }

    template <>
    inline ImageType
    TypedImageBase<int8_t>::get_type() const
    {
        return IMAGE_TYPE_SINT8;
    }

    template <>
    inline ImageType
    TypedImageBase<int16_t>::get_type() const
    {
        return IMAGE_TYPE_SINT16;
    }

    template <>
    inline ImageType
    TypedImageBase<int32_t>::get_type() const
    {
        return IMAGE_TYPE_SINT32;
    }

    template <>
    inline ImageType
    TypedImageBase<int64_t>::get_type() const
    {
        return IMAGE_TYPE_SINT64;
    }

    template <>
    inline ImageType
    TypedImageBase<uint8_t>::get_type() const
    {
        return IMAGE_TYPE_UINT8;
    }

    template <>
    inline ImageType
    TypedImageBase<uint16_t>::get_type() const
    {
        return IMAGE_TYPE_UINT16;
    }

    template <>
    inline ImageType
    TypedImageBase<uint32_t>::get_type() const
    {
        return IMAGE_TYPE_UINT32;
    }

    template <>
    inline ImageType
    TypedImageBase<uint64_t>::get_type() const
    {
        return IMAGE_TYPE_UINT64;
    }

    template <>
    inline ImageType
    TypedImageBase<float>::get_type() const
    {
        return IMAGE_TYPE_FLOAT;
    }

    template <>
    inline ImageType
    TypedImageBase<double>::get_type() const
    {
        return IMAGE_TYPE_DOUBLE;
    }


    template <typename T>
    inline T *
    TypedImageBase<T>::begin()
    {
        return this->data.empty() ? nullptr : &this->data[0];
    }

    template <typename T>
    inline T const*
    TypedImageBase<T>::begin() const
    {
        return this->data.empty() ? nullptr : &this->data[0];
    }

    template <typename T>
    inline T *
    TypedImageBase<T>::end()
    {
        return this->data.empty() ? nullptr : &this->data[0] + this->data.size();
    }

    template <typename T>
    inline T const*
    TypedImageBase<T>::end() const
    {
        return this->data.empty() ? nullptr : &this->data[0] + this->data.size();
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
        this->allocate(width,height,channels);
    }

    template <typename T>
    inline
    Image<T>::Image(Image<T> const& image1)
        : TypedImageBase<T>(image1)
    {

    }

    template <typename T>
    inline typename Image<T>::Ptr
    Image<T>::create()
    {
        return Ptr(new Image<T>());
    }

    template <typename T>
    inline typename Image<T>::Ptr
    Image<T>::create(int width, int height, int channels)
    {
        return Ptr(new Image<T>(width,height,channels));
    }

    template <typename T>
    inline typename Image<T>::Ptr
    Image<T>::create(Image<T> const&image1)
    {
        return Ptr(new Image<T>(image1));
    }

    template <typename T>
    inline typename Image<T>::Ptr
    Image<T>::duplicate() const
    {
        return Ptr(new Image<T>(*this));
    }

    template <typename T>
    inline void
    Image<T>::fill_color(T const *color)
    {
        for (T* iter = this->begin(); iter != this->end() ; iter += this->c)
        {
            std::copy(iter,iter + this->c, color);
        }
    }

    template <typename T>
    void
    Image<T>::add_channels(int num_channels, const T &value)
    {
        if(!num_channels || !this->valid())
        {
            return;
        }

        std::vector<T> tmp(this->w * this->h * (this->c + num_channels));
        typename std::vector<T>::iterator iter_tmp = tmp.end();
        typename std::vector<T>::iterator iter_this = this->end();

        for (int i = 0; i < this->get_pixel_amount(); ++i)
        {
            for(auto &a : tmp[i])
            {

            }
        }

    }

    template <typename T>
    void
    Image<T>::swap_channels(int channel1, int channel2)
    {

    }

    template <typename T>
    void
    Image<T>::copy_channel(int src, int dest)
    {

    }

    template <typename T>
    void
    Image<T>::delete_channel(int channel)
    {

    }

    template <typename T>
    inline T const&
    Image<T>::at(int index) const
    {
        return this->data[index];
    }

    template <typename T>
    inline T const&
    Image<T>::at(int index, int channel) const
    {
        int offset = index * this->c + channel;
        return this->data[offset];
    }

    template <typename T>
    inline T const&
    Image<T>::at(int x, int y, int channel) const
    {
        int offset = y * this->w * this->c + x * this->c +channel;
        return this->data[offset];
    }

    template <typename T>
    inline T&
    Image<T>::at(int index)
    {
        return this->data[index];
    }

    template <typename T>
    inline T&
    Image<T>::at(int index, int channel)
    {
        int offset = index * this->c + channel;
        return this->data[offset];
    }

    template <typename T>
    inline T&
    Image<T>::at(int x, int y, int channel)
    {
        int offset = y * this->w * this->c + x * this->c +channel;
        return this->data[offset];
    }

    template <typename T>
    T
    Image<T>::linear_at(float x, float y, int channel) const
    {

    }

    template <typename T>
    void
    Image<T>::linear_at(float x, float y, T *px) const
    {

    }

    template <typename T>
    inline T const&
    Image<T>::operator[](int index) const
    {
        return this->data[index];
    }

    template <typename T>
    inline T&
    Image<T>::operator[](int index)
    {
        return this->data[index];
    }

    template <typename T>
    inline T const&
    Image<T>::operator()(int index) const
    {
        return this->at(index);
    }

    template <typename T>
    inline T const&
    Image<T>::operator()(int index, int channel) const
    {
        return this->at(index,channel);
    }

    template <typename T>
    inline T const&
    Image<T>::operator()(int x, int y, int channel) const
    {
        return this->at(x,y,channel);
    }

    template <typename T>
    inline T&
    Image<T>::operator()(int index)
    {
        return this->at(index);
    }

    template <typename T>
    inline T& Image<T>::operator()(int index, int channel)
    {
        return this->at(index,channel);
    }

    template <typename T>
    inline T&
    Image<T>::operator()(int x, int y, int channel)
    {
        return this->at(x,y,channel);
    }


IMAGE_NAMESPACE_END

#endif //IMAGE_IMAGE_HPP
