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
#include <iostream>

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

/*******************************************************************
*~~~~~~~~~~~~~~~~~~~~~图像访问方法枚举声明~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*******************************************************************/
enum SWAP_METHOD
{
    AT,
    ITERATOR
};

/*******************************************************************
*~~~~~~~~~~~~~~~~~~~~~图像数据类型别名声明~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*******************************************************************/
typedef unsigned char uint8_t;
typedef unsigned short int uint16_t;
typedef unsigned int  uint32_t;
typedef unsigned long int uint64_t;
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
    typedef std::shared_ptr<ImageBase> BasePtr;
    typedef std::shared_ptr<ImageBase const> ConstBasePtr;
/********************************************************************
 *~~~~~~~~~~~~~~~~~~~~~~~ImageBase构造函数与析构函数~~~~~~~~~~~~~~~~~~~~
 *******************************************************************/
public:
    /*
    *  @property   默认构造函数
    *  @func       将图像进行初始化，w=h=c=0
    */
    ImageBase();
    ImageBase(const ImageBase& _img);

    /*
    *  @property   默认构造函数
    *  @func        对图像进行析构
    */
    virtual ~ImageBase();




/********************************************************************
 *~~~~~~~~~~~~~~~~~~~~~~~ImageBase管理函数~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *******************************************************************/
    /*
    *  @property   获取图像宽度
    *  @func       获取图像的宽
    *  @return     int
    */
    int width() const;
    int cols() const;

    /*
    *  @property   获取图像高度
    *  @func       获取图像的高
    *  @return     int
    */
    int height() const;
    int rows() const;
    /*
    *  @property   获取图像通道数
    *  @func       获取图像通道数
    *  @return     int
    */
    int channels() const;

    /*
    *  @property   判断是否为空
    *  @func       判断是否为空
    *  @return     bool
    */
    bool empty() const;
//
//    /*
//    *  @property   获取图像字节数
//    *  @func       虚函数，具体实现需要在子类中进行
//    *  @return     std::size_t  在子类中实现时返回图像字节数，否则返回0
//    */
//    virtual std::size_t get_byte_size() const;
//
//    /*
//    *  @property   获取图像数据指针
//    *  @func       虚函数，具体实现需要在子类中进行重载
//    *  @const1     指针指向内容不能变，也就是图像数据
//    *  @const2     防止改变类成员变量
//    *  @return     char const *  在子类中实现时返回图像指针，否则返回nullptr
//    */
//    virtual char const *get_byte_pointer() const;
//
//    /*
//    *  @property   获取图像数据指针
//    *  @func       虚函数，具体实现需要在子类中进行重载
//    *  @return     char *  在子类中实现时返回图像指针，否则返回nullptr
//    */
//    virtual char *get_byte_pointer();
//
//    /*
//    *  @property   获取图像数据类型
//    *  @func       虚函数，具体实现需要在子类中进行重载
//    *  @return    ImageType  在子类中实现时返回图像枚举类型，否则返回IMAGE_TYPE_UNKNOW
//    */
//    virtual ImageType get_type() const;
//
//    /*
//    *  @property   获取图像数据类型
//    *  @func       虚函数，具体实现需要在子类中进行重载
//    *  @return     char const*  在子类中实现时返回图像类型，否则返回IMAGE_TYPE_UNKNOW
//    */
//    virtual char const* get_type_string() const;
//
//    /*
//    *  @property   获取图像数据类型
//    *  @func       虚函数，具体实现需要在子类中进行重载
//    *  @return     ImageType 在子类中实现时返回图像类型，否则返回IMAGE_TYPE_UNKNOW string是什么就返回什么图像枚举类型
//    */
//    virtual ImageType get_type_for_string(std::string const& type_string);


protected:
    int w;
    int h;
    int c;//指的是图像通道数 正整数

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
    typedef std::shared_ptr<TypedImageBase<T>> TypedPtr;
    typedef std::shared_ptr<TypedImageBase<T> const> ConstTypedPtr;
    typedef std::vector<T> ImageData;

/********************************************************************
*~~~~~~~~~~~~~~~~~~TypedImageBase构造函数与析构函数~~~~~~~~~~~~~~~~~~~~~
********************************************************************/
public:
    /*
    *  @property   默认构造函数
    *  @func       将图像进行初始化，产生一幅空图
    */
    TypedImageBase();

    /*
    *  @property   拷贝构造函数
    *  @func       将图像用typedImageBase1进行初始化
    *  @param_in   typedImageBase1  已存在的图
    */
    TypedImageBase(TypedImageBase<T> const& typedImageBase1);

    /*
    *  @property   析构函数
    *  @func       将图像进行析构
    */
    virtual ~TypedImageBase();
        /*
    *  @property   重载运算符=
    *  @func       图像=赋值
    *  @param_in   待复制图像
    *  @return     复制后图像引用
    */
    TypedImageBase<T>& operator= (TypedImageBase<T> const& image);

    /*
    *  @property   图像复制
    *  @func       为图像动态分配内存，并以该内存区域对共享指针进行初始化
    *  @return     Ptr
    */
    virtual TypedPtr duplicate() const;

    /*
    *  @property   图像清理
    *  @func       将图像从内存中清除, 即长、宽、通道及图像数据都清零
    *  @return     void
    */
    virtual void release();

    /*
    *  @property   图像大小重定义
    *  @func       改变图像的尺寸以及数据大小，原来存在的数据继续保留，
                   如果图像缩小，则原多余的图像仍然会占据内存
    *  @param_in   width        新图像宽
    *  @param_in   height       新图像高
    *  @param_in   channels     新图像通道数
    *  @return     void
    */
    void resize(int width, int height, int channels);

    /*
    *  @property   图像大小重定义
    *  @func       重新分配图像空间，原来存在的数据清空
    *  @param_in   width        新图像宽
    *  @param_in   height       新图像高
    *  @param_in   channels     新图像通道数
    *  @return     void
    */
    void allocate(int width,int height, int channels);

    /*
    *  @property   图像数据填充
    *  @func       将图像所有像素填充数据value
    *  @param_in   value    待填充数据值
    *  @return     void
    */
    void fill(T const& value);

    /*
    *  @property   两图像交换
    *  @func       将两幅图像所有值进行交换
    *  @param_in   typedImageBase1    待交换图像
    *  @return     void
    */
    void swap(TypedImageBase<T>& typedImageBase1);

    /*
    *  @property   获取图像数据换
    *  @func       获取图像数据vector
    *  @return     ImageData const&
    */
    ImageData const& get_data() const;

    /*
    *  @property   获取图像数据换
    *  @func       获取图像数据vector
    *  @return     ImageDatat&
    */
    ImageData& get_data();

    /*
    *  @property   获取图像数据换指针
    *  @func       获取图像数据vector的指针，图像为空返回空指针
    *  @return     T const*
    */
    T const* get_data_pointer() const;

    /*
    *  @property   获取图像数据换指针
    *  @func       获取图像数据vector的指针，图像为空返回空指针
    *  @return     T*
    */
    T* get_data_pointer();

    /*
    *  @property   获取图像总像素大小
    *  @func       获取图像总像素个数
    *  @return     int
    */
    int get_pixel_amount() const;

    /*
    *  @property   获取图像总数据大小
    *  @func       获取图像总像数据个数，即数据大小
    *  @return     int
    */
    int get_value_amount() const;

    /*
    *  @property   获取图像数据总字节大小
    *  @func       获取图像总数据字节大小，
    *  @return     std::size_t
    */
    std::size_t get_byte_size() const;

    /*
    *  @property   获取图像数据字符指针
    *  @func       获取字符指针后，访问图像则以8bit长度进行访问
    *  @const1     指针指向数据不能更改
    *  @return     char const*
    */
    //用reinterpret_cast没有数位丢失
    char const* get_byte_pointer() const;

    /*
    *  @property   获取图像数据字符指针
    *  @func       获取字符指针后，访问图像则以8bit长度进行访问
    *  @return     char*
    */
    char* get_byte_pointer();

    /*
    *  @property   获取图像数据类型字符串
    *  @func       根据图像类型，获取对应的字符串数据，多个重载函数实现该虚函数
    *  @return     char const*
    */
    virtual char const* get_type_string() const;

    /*
    *  @property   获取图像数据枚举类型
    *  @func       根据图像类型，获取对应的枚举类型，多个重载函数实现该虚函数
    *  @return     ImageType
    */
    virtual ImageType get_type() const;

    /*
    *  @property   获取图像起始迭代器
    *  @func       获取图像数据初始指针，指向图像第一个数据
    *  @return     T*
    */
    T* begin();

    /*
    *  @property   获取图像起始迭代器
    *  @func       获取图像数据初始指针，指向图像第一个数据
    *  @return     T*
    */
    T const* begin() const;

    /*
    *  @property   获取图像末尾迭代器
    *  @func       获取图像数据末尾指针，指向图像最后一个数据的下一个位置
    *  @return     T*
    */
    T* end();

    /*
    *  @property   获取图像末尾迭代器
    *  @func       获取图像数据初始指针，指向图像最后一个数据的下一个位置
    *  @return     T const*
    */
    T const* end() const;

protected:
    ImageData data; //数组遍历速度比容器快20%左右

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
    Image(int width, int height, int channels=1);

     /*
     *  @property   拷贝构造函数
     *  @func       将图像进行初始化,
     *  @param_in   image1
     */
     Image(Image<T> const& image1)  ;

     virtual ~Image();

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
     static Ptr create(int width, int height, int channels =1 );

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
    //Ptr duplicate() const;

    /*
    *  @property   填充图像
    *  @func       为每一个像素填充color数组的颜色数组
    *  @param_in    T const* color      颜色数组
    *  @return     void
    */
    void fill_color(const T* const _color,int _c);

    void fill_color(const std::vector<T>& _color);

    /*
    *  @property    给图像添加通道
    *  @func        为图像添加num_channels数量的通道，值为value
    *  @param_in    num_channels    要添加的通道数
    *  @param_in    value           新通道分量的值
    *  @param_in    front_back      添加到前或后，0：前，1：后
    *  @return      void
    */
    void add_channels(int num_channels, T const& value = T(0));
    void add_channels(const std::vector<T> _value,int _front_back=1);


    //TODO: 对比下面两种访问时间差 直接访问at（）与迭代器方式
    /*
    *  @property   交换图像两通道的值
    *  @func       将两通道的值互换
    *  @param_in    channel1    要交换的通道
    *  @param_in    channel2    要交换的另一个通道
    *  @return     void
    */
    void swap_channels(int channel1, int channel2, SWAP_METHOD swap_method = AT);


    /*
    *  @property   复制图像通道
    *  @func       复制图像通道src->dest；dest=-1,则会新建一个图像通道
    *  @src        图像源通道
    *  @dest       图像目的通道，dest=-1,则会新建一个图像通道
    *  @return     void
    */
    void copy_channel(int src,int dest);

    /*
    *  @property   删除图像通道
    *  @func       将第channel通道删除
    *  @param_in   channel
    *  @return     void
    */
    void delete_channel(int channel);

     /*
    *  @property   访问图像某行数据
    *  @func       线性访问图像数据
    *  @param_in   index    图像数据行索引值
    *  @return     T* 该行第一个数据的指针
    */
    T* ptr(int row);
    const T* ptr(int row) const;
    /*
    *  @property   访问图像数据
    *  @func       线性访问图像数据
    *  @param_in   index    图像数据线性索引值
    *  @return     T const&
    */
    const T& at(int index) const;
    T& at(int index);
    /*
    *  @property   访问图像数据
    *  @func       线性访问图像数据
    *  @param_in   index    图像像素线性索引值
    *  @param_in   channel  待访问像素通道索引值
    *  @return     T const&
    */
    const T& at(int index, int channel) const;
    T& at(int index, int channel);
     /*
     *  @property   访问图像数据
     *  @func       二维索引访问图像数据  更加耗时
     *  @param_in   x    图像像素x方向索引值
     *  @param_in   y    图像像素x方向索引值
     *  @param_in   channel
     *  @return     T const&
     */
     const T& at(int x, int y, int channel) const;
     T& at(int x, int y, int channel);
    /*
    *  @property    像素插值
    *  @func        计算（x,y）浮点坐标经过双线性插值之后的像素单通道值
    *  @param_in    x           待插值坐标x
    *  @param_in    y           待插值坐标x
    *  @param_in    channel     待插值像素（x,y）的第channel个通道
    *  @return      T           插值后的通道值
    */
    T linear_at(float x, float y, int channel) const;

    /*
    *  @property    像素插值
    *  @func        计算（x,y）浮点坐标经过双线性插值之后的所有像素通道值
    *  @param_in    x           待插值坐标x
    *  @param_in    y           待插值坐标x
    *  @param_in    T* px       待插值像素（x,y）
    *  @return      void
    */
    void linear_at(float x, float y, T* px) const;

     /*
 *  @property   重载运算符=
 *  @func       图像=赋值
 *  @param_in   待复制图像
 *  @return     复制后图像引用
 */

 //    Image<T>& operator= (Image<T> const& image);

     //通过 .at() /  .ptr() 访问数据

//    /*
//    *  @property   重载运算符[]
//    *  @func       访问图像数据
//    *  @param_in   index    图像数据线性索引值
//    *  @return     T const&
//    */
//    T const& operator[] (int index) const;
//
//    /*
//    *  @property   重载运算符[]
//    *  @func       访问图像数据
//    *  @param_in   index    图像数据线性索引值
//    *  @return     T&
//    */
//     T& operator[] (int index);
//
//    /*
//    *  @property   重载运算符()
//    *  @func       访问图像数据
//    *  @param_in   index    图像数据线性索引值
//    *  @return     T const&
//    */
//    T const& operator() (int index) const;
//
//    /*
//    *  @property   重载运算符()
//    *  @func       访问图像数据
//    *  @param_in   index    图像像素索引值
//    *  @param_in   channel  图像像素通道索引值
//    *  @return     T const&
//    */
//    T const& operator() (int index, int channel) const;
//
//    /*
//    *  @property   重载运算符()
//    *  @func       访问图像数据
//    *  @param_in   x        图像像素x方向索引值
//    *  @param_in   y        图像像素y方向索引值
//    *  @param_in   channel  图像像素通道索引值
//    *  @return     T const&
//    */
//    T const& operator() (int x, int y, int channel) const;
//
//    /*
//    *  @property   重载运算符()
//    *  @func       访问图像数据
//    *  @param_in   index    图像数据线性索引值
//    *  @return     T&
//    */
//    T& operator()(int index);
//
//    /*
//    *  @property   重载运算符()
//    *  @func       访问图像数据
//    *  @param_in   index    图像像素索引值
//    *  @param_in   channel  图像像素通道索引值
//    *  @return     T&
//    */
//    T& operator()(int index, int channel);
//
//    /*
//    *  @property   重载运算符()
//    *  @func       访问图像数据
//    *  @param_in   x        图像像素x方向索引值
//    *  @param_in   y        图像像素y方向索引值
//    *  @param_in   channel  图像像素通道索引值
//    *  @return     T&
//    */
//    T& operator()(int x, int y, int channel);

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
    ImageBase:: ImageBase(const ImageBase& _img)//添加拷贝函数
    {
        this->w=_img.w;
        this->h=_img.h;
        this->c=_img.c;
    }
    inline
    ImageBase::~ImageBase()
    {

    }


    inline int
    ImageBase::width() const
    {
        return this->w;
    }
    inline int
    ImageBase::cols() const
    {
        return this->w;
    }
    inline int
    ImageBase::height() const
    {
        return this->h;
    }
    inline int
    ImageBase::rows() const
    {
        return this->h;
    }
    inline int
    ImageBase::channels() const
    {
        return this->c;
    }

    inline bool
    ImageBase::empty() const
    {
        return (w*h*c==0);
    }

//
//    inline std::size_t
//    ImageBase::get_byte_size() const
//    {
//        throw("Error::StsNotImplemented");
//        return 0;
//    }
//
//    inline char const*
//    ImageBase::get_byte_pointer() const
//    {
//        throw("Error::StsNotImplemented");
//        return nullptr;
//    }
//
//    inline char *
//    ImageBase::get_byte_pointer()
//    {
//        throw("Error::StsNotImplemented");
//        return nullptr;
//    }
//
//    inline ImageType
//    ImageBase::get_type() const
//    {
//        return IMAGE_TYPE_UNKNOWN;
//    }
//
//    inline char const*
//    ImageBase::get_type_string() const
//    {
//        return "unknown";
//    }
//
//
//    inline ImageType
//    ImageBase::get_type_for_string(std::string const &type_string)
//    {
//        if (type_string == "sint8")
//            return IMAGE_TYPE_SINT8;
//        else if (type_string == "sint16")
//            return IMAGE_TYPE_SINT16;
//        else if (type_string == "sint32")
//            return IMAGE_TYPE_SINT32;
//        else if (type_string == "sint64")
//            return IMAGE_TYPE_SINT64;
//        else if (type_string == "uint8")
//            return IMAGE_TYPE_UINT8;
//        else if (type_string == "uint16")
//            return IMAGE_TYPE_UINT16;
//        else if (type_string == "uint32")
//            return IMAGE_TYPE_UINT32;
//        else if (type_string == "uint64")
//            return IMAGE_TYPE_UINT64;
//        else if (type_string == "float")
//            return IMAGE_TYPE_FLOAT;
//        else if (type_string == "double")
//            return IMAGE_TYPE_DOUBLE;
//        else
//            return IMAGE_TYPE_UNKNOWN;
//    }
//

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
    TypedImageBase<T>& TypedImageBase<T>::operator = (TypedImageBase<T> const& image){
        if(this==&image)
            return *this;
        this->w=image.w;
        this->h=image.h;
        this->c=image.c;
        this->data.resize(image.data.size());
        this->data.assign(image.data.begin(),image.data.end());
        return *this;
    };
    template <typename T>
    inline typename TypedImageBase<T>::TypedPtr
    TypedImageBase<T>::duplicate() const
    {
        return TypedPtr(new TypedImageBase<T>(*this));
    }

    template <typename T>
    inline void
    TypedImageBase<T>::release()
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
        this->release();
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
    Image<T>::~Image() {

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

//    template <typename T>
//    inline typename Image<T>::Ptr
//    Image<T>::duplicate() const //拷贝函数代替复制函数
//    {
//        return Ptr(new Image<T>(*this));
//    }

    template <typename T>
    inline void
    Image<T>::fill_color(const T* const _color,int _c)
    {
        if(this->c!=_c)
        {
            printf("ArgumentOutOfRangeException\n");
            //TODO：：抛出异常
            return;
        }
        //TODO::TO CUDA @YANG
        for (T* iter = this->begin(); iter != this->end() ; iter += this->c)
        {
            //std::copy(iter,iter + this->c, color);
            std::copy(_color,_color + this->c, iter);
        }
    }
    template <typename T>
    inline void
    Image<T>::fill_color(const std::vector<T>& _color)
    {
        if(this->c!=_color.size())
        {
            printf("ArgumentOutOfRangeException\n");
            //TODO：：抛出异常
            return;
        }
        T* _color_arr=new T[_color.size()];
        for(int _it=0;_it<_color.size();_it++)
            _color_arr[_it]=_color[_it];
        fill_color(_color_arr,_color.size());
        delete[] _color_arr;
        return ;
    }
    template <typename T>
    void
    Image<T>::add_channels(int num_channels, const T &value)
    {
        if(num_channels<=0)
        {
            printf("ArgumentOutOfRangeException\n");
            //TODO::抛出异常
            return;
        }

        ImageData tmp(this->w * this->h * (this->c + num_channels));
        typename ImageData::iterator iter_tmp = tmp.end();
        typename ImageData::iterator iter_this = this->data.end();
        //TODO::TO CUDA @YANG
        for (int i = 0; i < this->get_pixel_amount(); ++i)
        {
            for (int j = 0; j < num_channels; ++j)
            {
                *(--iter_tmp) = value;
            }

            for (int k = 0; k < this->c; ++k)
            {
                *(--iter_tmp) = *(--iter_this);
            }
        }

        this->c += num_channels;

        std::swap(this->data,tmp);
    }
    template <typename T>
    void
    Image<T>::add_channels(const std::vector<T> _value,int _front_back)
    {
        if(_value.empty())
            return;
        int num_channels=_value.size();
        ImageData tmp(this->w * this->h * (this->c + num_channels));
        typename ImageData::iterator iter_tmp = tmp.end();
        typename ImageData::iterator iter_this = this->data.end();
        //TODO::TO CUDA @YANG
        if(_front_back){
            for (int i = 0; i < this->get_pixel_amount(); ++i)
            {
                for (int j = num_channels-1; j >=0; j--)
                {
                    *(--iter_tmp) = _value[j];
                }

                for (int k = 0; k < this->c; ++k)
                {
                    *(--iter_tmp) = *(--iter_this);
                }
            }
        }
        else{
            for (int i = 0; i < this->get_pixel_amount(); ++i)
            {
                for (int k = 0; k < this->c; ++k)
                {
                    *(--iter_tmp) = *(--iter_this);
                }
                for (int j = num_channels-1; j >=0; j--)
                {
                    *(--iter_tmp) = _value[j];
                }
            }
        }
        std::swap(this->data,tmp);
        this->c +=num_channels;
        return;
    }
    template <typename T>
    void
    Image<T>::swap_channels(int channel1, int channel2, SWAP_METHOD swap_method)
    {
        if (!this->valid() || channel1 == channel2)
        {
            return;
        }
        if(channel1<0||channel1>=this->c||channel2<0||channel2>=this->c)
        {
            printf("ArgumentOutOfRangeException\n");
            //TODO::抛出异常
            return;
        }
        if (swap_method != AT && swap_method != ITERATOR)
        {
            std::cout<<"交换方式错误!\n"<<std::endl;
            return;
        }

        //TODO::TO CUDA @YANG
        if (swap_method == AT)
        {
            for (int i = 0; i < this->get_pixel_amount(); ++i)
            {
                std::swap(this->at(i,channel1),this->at(i,channel2));
            }
        } else
        {
            T* iter1 = &this->at(0,channel1);
            T* iter2 = &this->at(0,channel2);
            for (int i = 0; i < this->get_pixel_amount(); iter1 += this->c, iter2 += this->c)
            {
                std::swap(*iter1,*iter2);
            }
        }
    }

    template <typename T>
    void
    Image<T>::copy_channel(int src, int dest)
    {
        if(src>=this->c||dest>=this->c)
        {
            printf("ArgumentOutOfRangeException\n");
            //TODO::抛出异常
        }
        if(!this->valid() || src == dest)
        {
            return;
        }

        if (dest < 0)
        {
            this->add_channels(1);
            dest = this->c;
        }

        T const* src_iter = &this->at(0,src);
        T* dest_iter = &this->at(0,dest);
        //TODO::TO CUDA @YANG
        for (int i = 0; i < this->get_pixel_amount(); src_iter += this->c, dest_iter += this->c,i++)
        {
            *dest_iter = *src_iter;
        }
    }

    template <typename T>
    void
    Image<T>::delete_channel(int channel)
    {
        if (channel < 0 || channel >= this->channels())
        {
            return;
        }

        T* src_iter = this->begin();
        T* dest_iter = this->begin();
        //TODO::TO CUDA @YANG
        for (int i = 0; i < this->data.size(); ++i)
        {
            if(i % this->c == channel)
            {
                src_iter++;
            } else
            {
                *(dest_iter++) = *(src_iter++);
            }
        }
        this->resize(this->w, this->h, this->c - 1);
    }
    template <typename T>
    T* Image<T>::ptr(int row){
        if(row>=this->h) {
            throw("row_index out image range\n");
        }
        T* data_ptr = this->data.data();
        data_ptr+=row*this->w*this->c;
        return data_ptr;
    };
    template <typename T>
    const T* Image<T>::ptr(int row) const{
        if(row>=this->h) {
            throw("row_index out image range\n");
        }
        const T* data_ptr = this->data.data();
        data_ptr+=row*this->w*this->c;
        return data_ptr;
    };

    template <typename T>
    inline const T&
    Image<T>::at(int index) const
    {
        if(index<0||index>=this->data.size())
        {
            printf("ArgumentOutOfRangeException\n");
            //TODO:此处抛出异常
        }
        return this->data[index];
    }
    template <typename T>
    inline T&
    Image<T>::at(int index)
    {
        if(index<0||index>=this->data.size())
        {
            printf("ArgumentOutOfRangeException\n");
            //TODO:此处抛出异常
        }
        return this->data[index];
    }
    template <typename T>
    inline const T&
    Image<T>::at(int index,  int channel) const
    {
        if(index<0||channel<0||index>this->w*this->h||channel>this->c)
        {
            printf("ArgumentOutOfRangeException\n");
            //TODO:此处抛出异常
        }
        int offset = index * this->c + channel;
        return this->data[offset];
    }
    template <typename T>
    inline T&
    Image<T>::at(int index,  int channel)
    {
        if(index<0||channel<0||index>this->w*this->h||channel>this->c)
        {
            printf("ArgumentOutOfRangeException\n");
            //TODO:此处抛出异常
        }
        int offset = index * this->c + channel;
        return this->data[offset];
    }
    template <typename T>
    inline const T&
    Image<T>::at(int x, int y, int channel) const
    {
        if(x<0||y<0||channel<0
        ||x>=this->w||y>=this->h||channel>=this->c)
        {
            printf("ArgumentOutOfRangeException\n");
            //TODO:此处抛出异常
        }
        int offset = y * this->w * this->c + x * this->c + channel;
        return this->data[offset];
    }
    template <typename T>
    inline T&
    Image<T>::at(int x, int y, int channel)
    {
        if(x<0||y<0||channel<0
           ||x>=this->w||y>=this->h||channel>=this->c)
        {
            printf("ArgumentOutOfRangeException\n");
            //TODO:此处抛出异常
        }
        int offset = y * this->w * this->c + x * this->c + channel;
        return this->data[offset];
    }
    template <typename T>
    T
    Image<T>::linear_at(float x, float y, int channel) const
    {
        if(x < 0 || x > this->w - 1 || y < 0 || y > this->h - 1)
        {
            std::cerr<<"插值坐标越界！\n"<<std::endl;
            std::exit(0);
        }

        int const floor_x = static_cast<int>(x);
        int const floor_y = static_cast<int>(y);
        int const ceil_x = static_cast<int>(x+1);
        int const ceil_y = static_cast<int>(y+1);

        float delta_x = x - static_cast<float >(floor_x);
        float delta_y = y - static_cast<float >(floor_y);

        if (x == 0 || x == this->h - 1)
        {
            return this->at(x, floor_y,channel) * (1-delta_y) + this->at(x, ceil_y,channel) * delta_y;
        }

        if(y == 0 || y == this->h -1)
        {
            return this->at(floor_x, y ,channel) * (1-delta_x) + this->at(ceil_x,y,channel) * delta_x;
        }

        T const px_f_x_f_y = this->at(floor_x,floor_y,channel);
        T const px_c_x_f_y = this->at(ceil_x,floor_y,channel);
        T const px_f_x_c_y = this->at(floor_x,ceil_y,channel);
        T const px_c_x_c_y = this->at(ceil_x,ceil_y,channel);

        T const px_x_floor_mid = (1-delta_x) * px_f_x_f_y + delta_x * px_c_x_f_y;
        T const px_x_ceil_mid = (1-delta_x) * px_c_x_f_y + delta_x * px_c_x_c_y;

        T px = (1 - delta_y) * px_x_floor_mid + delta_y * px_x_ceil_mid;
        return px;
    }

    template <typename T>
    void
    Image<T>::linear_at(float x, float y, T *px) const
    {
        for (int i = 0; i < this->c; ++i)
        {
            px[i] = this->linear_at(x,y,1 + i);
        }
    }
//    template <typename T>
//    Image<T>& Image<T>::operator = (Image<T> const& image){
//        if(this==&image)
//            return *this;
//        this->w=image.w;
//        this->h=image.h;
//        this->c=image.c;
//        this->data.resize(image.data.size());
//        this->data.assign(image.data.begin(),image.data.end());
//        return *this;
//    };
//    template <typename T>
//    inline T const&
//    Image<T>::operator[](int index) const
//    {
//        return this->data[index];
//    }
//
//    template <typename T>
//    inline T&
//    Image<T>::operator[](int index)
//    {
//        return this->data[index];
//    }
//
//    template <typename T>
//    inline T const&
//    Image<T>::operator()(int index) const
//    {
//        return this->at(index);
//    }
//
//    template <typename T>
//    inline T const&
//    Image<T>::operator()(int index, int channel) const
//    {
//        return this->at(index,channel);
//    }
//
//    template <typename T>
//    inline T const&
//    Image<T>::operator()(int x, int y, int channel) const
//    {
//        return this->at(x,y,channel);
//    }
//
//    template <typename T>
//    inline T&
//    Image<T>::operator()(int index)
//    {
//        return this->at(index);
//    }
//
//    template <typename T>
//    inline T& Image<T>::operator()(int index, int channel)
//    {
//        return this->at(index,channel);
//    }
//
//    template <typename T>
//    inline T&
//    Image<T>::operator()(int x, int y, int channel)
//    {
//        return this->at(x,y,channel);
//    }


IMAGE_NAMESPACE_END

#endif //IMAGE_IMAGE_HPP
