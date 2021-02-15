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
#include <string.h>
#include "types.hpp"
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
inline
ImageType getType(const char* name)
{
   if(name==typeid(unsigned char).name())
       return IMAGE_TYPE_UINT8;
   else if(name == typeid(char).name())
       return IMAGE_TYPE_SINT8;
   else if(name == typeid(unsigned short).name())
       return IMAGE_TYPE_UINT16;
   else if(name == typeid(short).name())
       return IMAGE_TYPE_SINT16;
   else if(name == typeid(unsigned int).name())
       return IMAGE_TYPE_UINT32;
   else if(name == typeid(int).name())
       return IMAGE_TYPE_SINT32;
   else if(name == typeid(float).name())
       return IMAGE_TYPE_FLOAT;
   else if(name == typeid(double).name())
       return IMAGE_TYPE_DOUBLE;
   else
       return IMAGE_TYPE_UNKNOWN;
}
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
    ImageBase(const ImageBase& img);
    ImageBase(int width,int height,int channels,ImageType type);
    /*
    *  @property   默认构造函数
    *  @func        对图像进行析构
    */
    virtual ~ImageBase();

    ImageBase&operator=(const ImageBase& src);


/********************************************************************
 *~~~~~~~~~~~~~~~~~~~~~~~ImageBase管理函数~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *******************************************************************/
    int getValueAmount() const;
    int getPixelAmount() const;
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

    int depth() const;
    int step() const;
    /*
    *  @property   判断是否为空
    *  @func       判断是否为空
    *  @return     bool
    */
    bool empty() const;

    void resize(int width,int height,int channels);
/********************************************************************
 *~~~~~~~~~~~~~~~~~~~~~~~ImageBase访问函数~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *******************************************************************/

    template <typename T>
    T* ptr(int row=0);
    template <typename T>
    T* at(int x,int y);
    template <typename T>
    T& at(int x,int y,int channel);
    template <typename T>
    const T* ptr(int row=0) const;
    template <typename T>
    const T* at(int x,int y) const;
    template <typename T>
    T at(int x,int y,int channel) const;
    protected:
    int m_w;
    int m_h;
    int m_c;//指的是图像通道数 正整数
    ImageType m_data_type;
    char *m_data;
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
class Image : public ImageBase
{

/********************************************************************
 *~~~~~~~~~~~~~~~~~~~~~~~~~Image常用容器别名定义~~~~~~~~~~~~~~~~~~~~~~~~
 *******************************************************************/
public:
   typedef std::shared_ptr<Image<T>> ImagePtr;
   typedef std::shared_ptr<Image<T> const> ConstImagePtr;
   typedef T*  ImageData;
   typedef const T*  constImageData;
   typedef T ValueType;

/********************************************************************
 *~~~~~~~~~~~~~~~~~~~~~~~Image构造函数与析构函数~~~~~~~~~~~~~~~~~~~~~~~~
 *******************************************************************/
 public:

    /*
    *  @property   默认构造函数
    *  @func       将图像进行初始化，m_w=m_h=m_c=0
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
     Image(ImageBase const& image);

     virtual ~Image();

     /*
     *  @property   智能指针构造函数
     *  @func       为图像动态分配内存，并赋给智能指针
     *  @static     静态成员函数在类加载时就会分配内存，可以通过类名直接访问，
     *              调用此函数不会访问或修改任何非static数据成员。
     *              用static修饰的函数，限定在本源码文件中，不能被本源码
     *              文件以外的代码文件调用
     *  @return     static ImagePtr
     */
     static ImagePtr create();

    /*
    *  @property   智能指针构造函数
    *  @func       为图像动态分配内存，并赋给智能指针
    *  @static     静态成员函数在类加载时就会分配内存，可以通过类名直接访问，
    *              调用此函数不会访问或修改任何非static数据成员。
    *              用static修饰的函数，限定在本源码文件中，不能被本源码
    *              文件以外的代码文件调用
    *  @return     static ImagePtr
    */
     static ImagePtr create(int width, int height, int channels =1 );

     /*
     *  @property   智能指针构造函数
     *  @func       为图像动态分配内存，并赋给智能指针
     *  @static     静态成员函数在类加载时就会分配内存，可以通过类名直接访问，
     *              调用此函数不会访问或修改任何非static数据成员。
     *              用static修饰的函数，限定在本源码文件中，不能被本源码
     *              文件以外的代码文件调用
     *  @return     static ImagePtr
     */
     static ImagePtr create(Image<T> const&image1);

  /*******************************************************************
  *~~~~~~~~~~~~~~~~~~~~~~~~~Image管理函数~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  *******************************************************************/
     /*  @property   重载运算符()
     *  @func       图像ROI提取
     *  @param_in   待提取图像
     *  @return     提取后图像
     */
     Image<T> operator() (Rect roi) const;
    /*
    *  @property   复制图先锋
    *  @func       复制图像
    *  @return     ImagePtr
    */
    //ImagePtr duplicate() const;

    /*
    *  @property   填充图像
    *  @func       为每一个像素填充color数组的颜色数组
    *  @param_in    T const* color      颜色数组
    *  @return     void
    */
    void fillColor(const T* const color,int c);

    void fillColor(const std::vector<T>& color);

    /*
    *  @property    给图像添加通道
    *  @func        为图像添加num_channels数量的通道，值为value
    *  @param_in    num_channels    要添加的通道数
    *  @param_in    value           新通道分量的值
    *  @param_in    front_back      添加到前或后，0：前，1：后
    *  @return      void
    */
    void addChannels(int num_channels, T const& value = T(0));
    void addChannels(const std::vector<T> value,int front_back=1);


    //TODO: 对比下面两种访问时间差 直接访问at（）与迭代器方式
    /*
    *  @property   交换图像两通道的值
    *  @func       将两通道的值互换
    *  @param_in    channel1    要交换的通道
    *  @param_in    channel2    要交换的另一个通道
    *  @return     void
    */
    void swapChannels(int channel1, int channel2, SWAP_METHOD swap_method = AT);


    /*
    *  @property   复制图像通道
    *  @func       复制图像通道src->dest；dest=-1,则会新建一个图像通道
    *  @src        图像源通道
    *  @dest       图像目的通道，dest=-1,则会新建一个图像通道
    *  @return     void
    */
    void copyChannel(int src,int dest);

    /*
    *  @property   删除图像通道
    *  @func       将第channel通道删除
    *  @param_in   channel
    *  @return     void
    */
    void deleteChannel(int channel);

     /*
    *  @property   访问图像某行数据
    *  @func       线性访问图像数据
    *  @param_in   index    图像数据行索引值
    *  @return     T* 该行第一个数据的指针
    */
/*******************************************************************
*~~~~~~~~~~~~~~~~~~~~~~~~~Image访问函数~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*******************************************************************/
    T* ptr(int row=0);
    const T* ptr(int row=0) const;
    /*
    *  @property   访问图像数据
    *  @func       线性访问图像数据
    *  @param_in   index    图像数据线性索引值
    *  @return     T const&
    */
    T at(int index) const;
    T& at(int index);
    /*
    *  @property   访问图像数据
    *  @func       线性访问图像数据
    *  @param_in   index    图像像素线性索引值
    *  @param_in   channel  待访问像素通道索引值
    *  @return     T const&
    */
    const T* at(int x, int y) const;
    T* at(int x, int y);
     /*
     *  @property   访问图像数据
     *  @func       二维索引访问图像数据  更加耗时
     *  @param_in   x    图像像素x方向索引值
     *  @param_in   y    图像像素x方向索引值
     *  @param_in   channel
     *  @return     T const&
     */
     T at(int x, int y, int channel) const;
     T& at(int x, int y, int channel);
    /*
    *  @property    像素插值
    *  @func        计算（x,y）浮点坐标经过双线性插值之后的像素单通道值
    *  @param_in    x           待插值坐标x
    *  @param_in    y           待插值坐标x
    *  @param_in    channel     待插值像素（x,y）的第channel个通道
    *  @return      T           插值后的通道值
    */
    T linearAt(float x, float y, int channel) const;

    /*
    *  @property    像素插值
    *  @func        计算（x,y）浮点坐标经过双线性插值之后的所有像素通道值
    *  @param_in    x           待插值坐标x
    *  @param_in    y           待插值坐标x
    *  @param_in    T* px       待插值像素（x,y）
    *  @return      void
    */
    void linearAt(float x, float y, T* px) const;

     /*
 *  @property   重载运算符=
 *  @func       图像=赋值
 *  @param_in   待复制图像
 *  @return     复制后图像引用
 */

     Image<T>& operator= (Image<T> const& image);

 };

IMAGE_NAMESPACE_END

IMAGE_NAMESPACE_BEGIN
/********************************************************************
*~~~~~~~~~~~~~~~~~~~~~ImageBase成员函数实现~~~~~~~~~~~~~~~~~~~~~~~~~~~~
********************************************************************/

    inline
    ImageBase::ImageBase()
            : m_w(0),
            m_h(0),
            m_c(0),
            m_data_type(ImageType::IMAGE_TYPE_UNKNOWN),
            m_data(NULL){    }

    inline
    ImageBase::ImageBase(int width ,int height,int channels,ImageType type)
    {
        m_w=width;
        m_h=height;
        m_c=channels;
        m_data_type=type;
        m_data=new char[m_w*m_h*m_c*depth()];
    }
    inline
    ImageBase:: ImageBase(const ImageBase& _img)//添加拷贝函数
    {
        this->m_w=_img.m_w;
        this->m_h=_img.m_h;
        this->m_c=_img.m_c;
        this->m_data_type=_img.m_data_type;
        //if(m_data)
        //    delete[] m_data;
        size_t size = m_h*m_w*m_c*depth();
        m_data=new char[size];
        memcpy(this->m_data,_img.m_data,m_h*m_w*m_c*depth());
    }
    inline
    ImageBase::~ImageBase()
    {
        if(m_data!=NULL)
        {
            delete[] m_data;
            m_data=NULL;
        }
    }
    inline
    ImageBase& ImageBase::operator=(const ImageBase& src)
    {
        if(&src==this)
            return *this;
        this->m_w=src.m_w;
        this->m_h=src.m_h;
        this->m_c=src.m_c;
        this->m_data_type=src.m_data_type;
        if(m_data)
            delete[] m_data;
        size_t size = m_h*m_w*m_c*depth();
        m_data=new char[size];
        memcpy(this->m_data,src.m_data,size);

        return *this;
    }

    int inline ImageBase::getValueAmount() const{
        return m_w*m_h*m_c;
    }
    int inline ImageBase::getPixelAmount() const {
        return m_w*m_h;
    }
    inline int
    ImageBase::width() const
    {
        return this->m_w;
    }
    inline int
    ImageBase::cols() const
    {
        return this->m_w;
    }
    inline int
    ImageBase::height() const
    {
        return this->m_h;
    }
    inline int
    ImageBase::rows() const
    {
        return this->m_h;
    }
    inline int
    ImageBase::channels() const
    {
        return this->m_c;
    }

    inline int
    ImageBase::step() const{
        return m_c*m_w*depth();
    }

    inline int
    ImageBase::depth() const{
        switch(m_data_type){
            case IMAGE_TYPE_UNKNOWN:
                return 0;
            case IMAGE_TYPE_SINT8:
            case IMAGE_TYPE_UINT8:
                return 1;
            case IMAGE_TYPE_SINT16:
            case IMAGE_TYPE_UINT16:
                return 2;
            case IMAGE_TYPE_SINT32:
            case IMAGE_TYPE_UINT32:
            case IMAGE_TYPE_FLOAT:
                return 4;
            case IMAGE_TYPE_DOUBLE:
                return 8;
        }
    }

    inline bool
    ImageBase::empty() const
    {
        return (m_w*m_h*m_c==0);
    }
    inline void
    ImageBase::resize(int width, int height, int channels) {
        if(width==m_w&&height==m_h&&channels==m_c)
            return;
        m_w=width;
        m_h=height;
        m_c=channels;
        delete[] m_data;
        m_data=new char[m_w*m_h*m_c*depth()];
    }



    template <typename T>
    T* ImageBase::ptr(int row) {
        if(row>=m_h||row<0)
            throw("row index out of range\n");
        T* data_ptr=(T*)m_data;
        data_ptr+=row*m_w*m_c;
        return data_ptr;
    }
    template <typename T>
    T* ImageBase::at(int x,int y){
        if(y>=m_h||y<0||x>=m_w||x<0)
            throw("index out of range\n");
        T* data_ptr=(T*)m_data;
        data_ptr+=y*m_w*m_c+x*m_c;
        return data_ptr;
    }
    template <typename T>
    T& ImageBase::at(int x,int y,int channel){
        if(y>=m_h||y<0||x>=m_w||x<0)
            throw("index out of range\n");
        T* data_ptr=(T*)m_data;
        data_ptr+=y*m_w*m_c+x*m_c+channel;
        return *data_ptr;
    }
    template <typename T>
    const T* ImageBase::ptr(int row) const{
        if(row>=m_h||row<0)
            throw("row index out of range\n");
        T* data_ptr=(T*)m_data;
        data_ptr+=row*m_w*m_c;
        return data_ptr;
    }
    template <typename T>
    const T* ImageBase::at(int x,int y) const{
        if(y>=m_h||y<0||y>=m_w||y<0)
            throw("index out of range\n");
        T* data_ptr=(T*)m_data;
        data_ptr+=y*m_w*m_c+x*m_c;
        return data_ptr;
    }
    template <typename T>
    T ImageBase::at(int x,int y,int channel) const{
        if(y>=m_h||y<0||x>=m_w||x<0)
            throw("index out of range\n");
        T* data_ptr=(T*)m_data;
        data_ptr+=y*m_w*m_c+x*m_c+channel;
        return *data_ptr;
    }

/********************************************************************
*~~~~~~~~~~~~~~~~~~~~~Image成员函数实现~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
********************************************************************/
    template <typename T>
    inline
    Image<T>::Image():ImageBase()
    {
        m_data_type = getType(typeid(T).name());
    }

    template <typename T>
    inline
    Image<T>::Image(int width, int height, int channels)
    :ImageBase(width,height,channels,getType(typeid(T).name()))
    {
    }
    template <typename T>
    inline
    Image<T>::Image(ImageBase const& image1)
        : ImageBase(image1){
        }

    template <typename T>
    Image<T>::~Image() {
    }

    template <typename T>
    inline typename Image<T>::ImagePtr
    Image<T>::create()
    {
        return std::make_shared<Image<T>>();
    }

    template <typename T>
    inline typename Image<T>::ImagePtr
    Image<T>::create(int width, int height, int channels)
    {
        return std::make_shared<Image<T>>(width,height,channels);
    }

    template <typename T>
    inline typename Image<T>::ImagePtr
    Image<T>::create(Image<T> const&image1)
    {
        return std::make_shared<Image<T>>(image1);
    }

//    template <typename T>
//    inline typename Image<T>::ImagePtr
//    Image<T>::duplicate() const //拷贝函数代替复制函数
//    {
//        return ImagePtr(new Image<T>(*this));
//    }
    template <typename T>
    Image<T> Image<T>::operator()(Rect roi) const{

        if(roi.height<=0||roi.width<=0||
        roi.x+roi.width>=this->m_w||roi.y+roi.height>=this->m_h)
        {
            throw("error roi\n");
        }
        Image<T> img_roi(roi.width,roi.height,this->m_c);
        for(int i=roi.y;i<roi.y+roi.height;i++){
           const T* src_data=this->ptr(i);
           T* roi_data=img_roi.ptr(i);
           memcpy(roi_data,src_data,roi.width*this->m_c*sizeof(T));
        }

    }
    template <typename T>
    inline void
    Image<T>::fillColor(const T* const color,int c)
    {
        if(this->m_c!=c)
        {
            printf("ArgumentOutOfRangeException\n");
            //TODO：：抛出异常
            return;
        }
        //TODO::TO CUDA @YANG
        for (T* iter = this->begin(); iter != this->end() ; iter += this->m_c)
        {
            //std::copy(iter,iter + this->m_c, color);
            std::copy(color,color + this->m_c, iter);
        }
    }
    template <typename T>
    inline void
    Image<T>::fillColor(const std::vector<T>& color)
    {
        if(this->m_c!=color.size())
        {
            printf("ArgumentOutOfRangeException\n");
            //TODO：：抛出异常
            return;
        }
        T* color_arr=new T[color.size()];
        for(int _it=0;_it<color.size();_it++)
            color_arr[_it]=color[_it];
        fillColor(color_arr,color.size());
        delete[] color_arr;
        return ;
    }
//    template <typename T>
//    void
//    Image<T>::addChannels(int num_channels, const T &value)
//    {
//        if(num_channels<=0)
//        {
//            printf("ArgumentOutOfRangeException\n");
//            //TODO::抛出异常
//            return;
//        }
//
//        ImageData tmp(this->m_w * this->m_h * (this->m_c + num_channels));
//        typename ImageData::iterator iter_tmp = tmp.end();
//        typename ImageData::iterator iter_this = this->m_data.end();
//        //TODO::TO CUDA @YANG
//        for (int i = 0; i < this->getPixelAmount(); ++i)
//        {
//            for (int j = 0; j < num_channels; ++j)
//            {
//                *(--iter_tmp) = value;
//            }
//
//            for (int k = 0; k < this->m_c; ++k)
//            {
//                *(--iter_tmp) = *(--iter_this);
//            }
//        }
//
//        this->m_c += num_channels;
//
//        std::swap(this->m_data,tmp);
//    }
//    template <typename T>
//    void
//    Image<T>::addChannels(const std::vector<T> value,int front_back)
//    {
//        if(value.empty())
//            return;
//        int num_channels=value.size();
//        ImageData tmp(this->m_w * this->m_h * (this->m_c + num_channels));
//        typename ImageData::iterator iter_tmp = tmp.end();
//        typename ImageData::iterator iter_this = this->m_data.end();
//        //TODO::TO CUDA @YANG
//        if(front_back){
//            for (int i = 0; i < this->getPixelAmount(); ++i)
//            {
//                for (int j = num_channels-1; j >=0; j--)
//                {
//                    *(--iter_tmp) = value[j];
//                }
//
//                for (int k = 0; k < this->m_c; ++k)
//                {
//                    *(--iter_tmp) = *(--iter_this);
//                }
//            }
//        }
//        else{
//            for (int i = 0; i < this->getPixelAmount(); ++i)
//            {
//                for (int k = 0; k < this->m_c; ++k)
//                {
//                    *(--iter_tmp) = *(--iter_this);
//                }
//                for (int j = num_channels-1; j >=0; j--)
//                {
//                    *(--iter_tmp) = value[j];
//                }
//            }
//        }
//        std::swap(this->m_data,tmp);
//        this->m_c +=num_channels;
//    }
//    template <typename T>
//    void
//    Image<T>::swapChannels(int channel1, int channel2, SWAP_METHOD swap_method)
//    {
//        if (this->empty() || channel1 == channel2)
//        {
//            return;
//        }
//        if(channel1<0||channel1>=this->m_c||channel2<0||channel2>=this->m_c)
//        {
//            printf("ArgumentOutOfRangeException\n");
//            //TODO::抛出异常
//            return;
//        }
//        if (swap_method != AT && swap_method != ITERATOR)
//        {
//            std::cout<<"交换方式错误!\n"<<std::endl;
//            return;
//        }
//
//        //TODO::TO CUDA @YANG
//        if (swap_method == AT)
//        {
//            for (int i = 0; i < this->getPixelAmount(); ++i)
//            {
//                std::swap(this->at(i,channel1),this->at(i,channel2));
//            }
//        } else
//        {
//            T* iter1 = &this->at(0,channel1);
//            T* iter2 = &this->at(0,channel2);
//            for (int i = 0; i < this->getPixelAmount(); iter1 += this->m_c, iter2 += this->m_c)
//            {
//                std::swap(*iter1,*iter2);
//            }
//        }
//    }
//    template <typename T>
//    void
//    Image<T>::copyChannel(int src, int dest)
//    {
//        if(src>=this->m_c||dest>=this->m_c)
//        {
//            printf("ArgumentOutOfRangeException\n");
//            //TODO::抛出异常
//        }
//        if(this->empty() || src == dest)
//        {
//            return;
//        }
//
//        if (dest < 0)
//        {
//            this->addChannels(1);
//            dest = this->m_c;
//        }
//
//        T const* src_iter = &this->at(0,src);
//        T* dest_iter = &this->at(0,dest);
//        //TODO::TO CUDA @YANG
//        for (int i = 0; i < this->getPixelAmount(); src_iter += this->m_c, dest_iter += this->m_c,i++)
//        {
//            *dest_iter = *src_iter;
//        }
//    }
//    template <typename T>
//    void
//    Image<T>::deleteChannel(int channel)
//    {
//        if (channel < 0 || channel >= this->channels())
//        {
//            return;
//        }
//
//        T* src_iter = this->begin();
//        T* dest_iter = this->begin();
//        //TODO::TO CUDA @YANG
//        for (int i = 0; i < this->m_data.size(); ++i)
//        {
//            if(i % this->m_c == channel)
//            {
//                src_iter++;
//            } else
//            {
//                *(dest_iter++) = *(src_iter++);
//            }
//        }
//        this->resize(this->m_w, this->m_h, this->m_c - 1);
//    }
    template <typename T>
    T* Image<T>::ptr(int row){
        T* data_ptr=ImageBase::ptr<T>(row);
        return data_ptr;
    };
    template <typename T>
    const T* Image<T>::ptr(int row) const{
        const T* data_ptr=ImageBase::ptr<T>(row);
        return data_ptr;
    };

    template <typename T>
    inline T
    Image<T>::at(int index) const
    {
        return *(((T*)m_data)+index);
    }
    template <typename T>
    inline T&
    Image<T>::at(int index)
    {
        return *(((T*)m_data)+index);
    }
    template <typename T>
    inline const T*
    Image<T>::at(int x,  int y) const
    {
        return ImageBase::at<T>(x,y);
    }
    template <typename T>
    inline T*
    Image<T>::at(int x,  int y)
    {
        return ImageBase::at<T>(x,y);
    }
    template <typename T>
    inline T
    Image<T>::at(int x,  int y, int channel) const
    {
        return ImageBase::at<T>(x,y,channel);
    }
    template <typename T>
    inline T&
    Image<T>::at(int x,  int y, int channel)
    {
        return ImageBase::at<T>(x,y,channel);
    }
    template <typename T>
    T
    Image<T>::linearAt(float x, float y, int channel) const
    {
        if(x < 0 || x > this->m_w - 1 || y < 0 || y > this->m_h - 1)
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

        if (x == 0 || x == this->m_h - 1)
        {
            return this->at(x, floor_y,channel) * (1-delta_y) + this->at(x, ceil_y,channel) * delta_y;
        }

        if(y == 0 || y == this->m_h -1)
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
    Image<T>::linearAt(float x, float y, T *px) const
    {
        for (int i = 0; i < this->m_c; ++i)
        {
            px[i] = this->linearAt(x,y,1 + i);
        }
    }
    template <typename T>
    Image<T>& Image<T>::operator = (Image<T> const& image){
        if(this==&image)
            return *this;
        ImageBase::operator=(image);
        return *this;
    };
//    template <typename T>
//    inline T const&
//    Image<T>::operator[](int index) const
//    {
//        return this->m_data[index];
//    }
//
//    template <typename T>
//    inline T&
//    Image<T>::operator[](int index)
//    {
//        return this->m_data[index];
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
