/*
 * @desc    图像处理函数
 * @author  安帅
 * @date    2019-01-22
 * @email   1028792866@qq.com
*/

#ifndef IMAGE_IMAGE_PROCESS_HPP
#define IMAGE_IMAGE_PROCESS_HPP

//#include "define.h"
#include "IMAGE/image.hpp"
#include "MATH/Function/function.hpp"

IMAGE_NAMESPACE_BEGIN
/*******************************************************************
*~~~~~~~~~~~~~~~~~~~~~图像饱和度类型枚举声明~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*******************************************************************/
//http://changingminds.org/explanations/perception/visual/lightness_variants.htm
//https://ninghao.net/video/2116
//三者都是亮度，HSL空间中的L,不同软件的L不一样分别为LIGHTNESS，LUMINOSITY与LUMINANCE
//HSL色彩模式是工业界的一种颜色标准
//H： Hue 色相             代表的是人眼所能感知的颜色范围，这些颜色分布在一个平面的色相环上
//                        基本参照：360°/0°红、60°黄、120°绿、180°青、240°蓝、300°洋红
//S：Saturation 饱和度     用0%至100%的值描述了相同色相、明度下色彩纯度的变化。数值越大，
//                        颜色中的灰色越少，颜色越鲜艳
//L ：Lightness 明度       作用是控制色彩的明暗变化。它同样使用了0%至100%的取值范围。
//                        数值越小，色彩越暗，越接近于黑色；数值越大，色彩越亮，越接近于白色。
enum KDesaturateType
{
    DESATURATE_MAXIMUM,     //Maximum = max(R,G,B)
    DESATURATE_LIGHTNESS,   //Lightness = 1/2 * (max(R,G,B) + min(R,G,B))
    DESATURATE_LUMINOSITY,  //Luminosity = 0.21 * R + 0.72 * G + 0.07 * B
    DESATURATE_LUMINANCE,   //Luminince = 0.30 * R + 0.59 * G + 0.11 * B
    DESATURATE_AVERAGE      //Average Brightness = 1/3 * (R + G +B)
};

/*******************************************************************
*~~~~~~~~~~~~~~~~~~~~~常用图像函数处理声明~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*******************************************************************/
enum kResizeMode{
    INTER_NEAREST=0,
    INTER_LINEAR=1
};
template <typename T>
void imageResizeNearest(const Image<T>& src,Image<T>& dst,int width, int height) {
    //最邻近插值法
    if(width==src.width()&&height==src.height()) {
        dst=Image<T>(src);
        return;
    }
    int src_h=src.height();
    int src_w=src.width();
    int channels=src.channels();
    dst.resize(width,height,channels);
    float scale_x=(float)src_w/width;
    float scale_y=(float)src_h/height;

    for(int i=0;i<height;i++){
        //int dst_offst=i*width*channels;
        int src_row=(int)(i*scale_y);
        src_row=src_row<src_h?src_row:src_h-1;
        //int src_offst=src_row*src_w*channels;
        T* dst_row_ptr=dst.ptr(i);
        const T* src_row_ptr=src.ptr(src_row);
        for(int j=0;j<width;j++){
            int src_col=(int)(j*scale_x);
            src_col=src_col<src_w?src_col:src_w-1;
            for(int ch=0;ch<channels;ch++) {
                //dst[dst_offst+j*channels+ch]=src[src_offst+src_col*channels+ch];
                dst_row_ptr[j*channels+ch]=src_row_ptr[src_col*channels+ch];
            }
        }
    }
}
template <typename T>
void imageResizeLinear(const Image<T>& src,Image<T>& dst,int width, int height) {
        //最邻近插值法
        if(width==src.width()&&height==src.height()) {
            dst=Image<T>(src);
            return;
        }
        int src_h=src.height();
        int src_w=src.width();
        int channels=src.channels();
        dst.resize(width,height,channels);
        float scale_x=(float)src_w/width;
        float scale_y=(float)src_h/height;

        T* dataDst=dst.getData().data();
        int stepDst=width*channels;
        const T* dataSrc=src.getData().data();
        int stepSrc=src_w*channels;
        for(int i=0;i<height;i++){
            int dst_offst=i*width*channels;
            float fy=(float)((i+0.5)*scale_y-0.5);
            int sy=std::floor(fy);
            fy-=sy;
            sy=std::min(sy,src_h-2);
            sy=std::max(sy,0);
            short cbufy[2];
            cbufy[0]=(short)((1.f-fy)*2048);
            cbufy[1]=2048-cbufy[0];
            for(int j=0;j<width;j++){
                float fx=(float)((j+0.5)*scale_x-0.5);
                int sx=std::floor(fx);
                fx-=sx;
                if(sx<0){
                    fx=0,sx=0;
                }
                if(sx>=src_w-1){
                    fx=0,sx=src_w-2;
                }
                short cbufx[2];
                cbufx[0]=(short)((1.f-fx)*2048);
                cbufx[1]=2048-cbufx[0];
                for(int k=0;k<channels;k++)
                {
                    *(dataDst+ i*stepDst + channels*j + k) = (int)(*(dataSrc + sy*stepSrc + channels*sx + k) * cbufx[0] * cbufy[0]
                            + *(dataSrc + (sy+1)*stepSrc + channels*sx + k) * cbufx[0] * cbufy[1]
                            + *(dataSrc + sy*stepSrc + channels*(sx+1) + k) * cbufx[1] * cbufy[0]
                            + *(dataSrc + (sy+1)*stepSrc + channels*(sx+1) + k) * cbufx[1] * cbufy[1])>> 22;
                }
            }
        }
    }
template <typename T>
void imageResize(const Image<T>& src,Image<T>& dst,int width, int height,kResizeMode mode=INTER_LINEAR){
    if(mode==INTER_NEAREST)
    {
        imageResizeNearest<T>(src,dst,width,height);
    } else if(mode ==INTER_LINEAR )
    {
        imageResizeLinear<T>(src,dst,width,height);
    }
}
enum kThresholdType{
            THRESH_BINARY=0,
            THRESH_BINARY_INV,
            THRESH_TRUNC,
            THRESH_TOZERO,
            THRESH_TOZERO_INV
        };
template <typename T>
void imageThreshold(const Image<T>& src,Image<T>&dst,double thresh, double maxval, int type ){
    if(src.channels()!=1){
        throw("threshold channels num must be 1\n");
    }
    dst.resize(src.width(),src.height(),src.channels());

    switch (type){
        case THRESH_BINARY:
            for(int y=0;y<src.rows();y++) {
                T *dst_ptr = dst.ptr(y);
                const T *src_ptr = src.ptr(y);
                for (int x = 0; x < src.cols(); x++) {
                    if (src_ptr[x] > thresh)
                        dst_ptr[x] = maxval;
                    else
                        dst_ptr[x] = 0;
                }
            }
            break;
        case THRESH_BINARY_INV:
            for(int y=0;y<src.rows();y++) {
                T *dst_ptr = dst.ptr(y);
                const T *src_ptr = src.ptr(y);
                for (int x = 0; x < src.cols(); x++) {
                    if (src_ptr[x] > thresh)
                        dst_ptr[x] = 0;
                    else
                        dst_ptr[x] = maxval;
                }
            }
            break;
        case THRESH_TRUNC:
            for(int y=0;y<src.rows();y++) {
                T *dst_ptr = dst.ptr(y);
                const T *src_ptr = src.ptr(y);
                for (int x = 0; x < src.cols(); x++) {
                    if (src_ptr[x] > thresh)
                        dst_ptr[x] = thresh;
                    else
                        dst_ptr[x] = src_ptr[x];
                }
            }
            break;
        case THRESH_TOZERO:
            for(int y=0;y<src.rows();y++) {
                T *dst_ptr = dst.ptr(y);
                const T *src_ptr = src.ptr(y);
                for (int x = 0; x < src.cols(); x++) {
                    if (src_ptr[x] > thresh)
                        dst_ptr[x] = src_ptr[x];
                    else
                        dst_ptr[x] = 0;
                }
            }
            break;
        case THRESH_TOZERO_INV:
            for(int y=0;y<src.rows();y++) {
                T *dst_ptr = dst.ptr(y);
                const T *src_ptr = src.ptr(y);
                for (int x = 0; x < src.cols(); x++) {
                    if (src_ptr[x] > thresh)
                        dst_ptr[x] = 0;
                    else
                        dst_ptr[x] = src_ptr[x];
                }
            }
            break;
    }
}
template <typename T>
void imageThreshold(Image<T>& src,Image<T>&dst,double thresh, double maxval, int type ){
    if(src.channels()!=1){
        throw("threshold channels num must be 1\n");
    }
    if(&src!=&dst)
        dst.resize(src.width(),src.height(),src.channels());

    switch (type){
        case THRESH_BINARY:
            for(int y=0;y<src.rows();y++) {
                T *dst_ptr = dst.ptr(y);
                const T *src_ptr = src.ptr(y);
                for (int x = 0; x < src.cols(); x++) {
                    if (src_ptr[x] > thresh)
                        dst_ptr[x] = maxval;
                    else
                        dst_ptr[x] = 0;
                }
            }
            break;
        case THRESH_BINARY_INV:
            for(int y=0;y<src.rows();y++) {
                T *dst_ptr = dst.ptr(y);
                const T *src_ptr = src.ptr(y);
                for (int x = 0; x < src.cols(); x++) {
                    if (src_ptr[x] > thresh)
                        dst_ptr[x] = 0;
                    else
                        dst_ptr[x] = maxval;
                }
            }
            break;
        case THRESH_TRUNC:
            for(int y=0;y<src.rows();y++) {
                T *dst_ptr = dst.ptr(y);
                const T *src_ptr = src.ptr(y);
                for (int x = 0; x < src.cols(); x++) {
                    if (src_ptr[x] > thresh)
                        dst_ptr[x] = thresh;
                    else
                        dst_ptr[x] = src_ptr[x];
                }
            }
            break;
        case THRESH_TOZERO:
            for(int y=0;y<src.rows();y++) {
                T *dst_ptr = dst.ptr(y);
                const T *src_ptr = src.ptr(y);
                for (int x = 0; x < src.cols(); x++) {
                    if (src_ptr[x] > thresh)
                        dst_ptr[x] = src_ptr[x];
                    else
                        dst_ptr[x] = 0;
                }
            }
            break;
        case THRESH_TOZERO_INV:
            for(int y=0;y<src.rows();y++) {
                T *dst_ptr = dst.ptr(y);
                const T *src_ptr = src.ptr(y);
                for (int x = 0; x < src.cols(); x++) {
                    if (src_ptr[x] > thresh)
                        dst_ptr[x] = 0;
                    else
                        dst_ptr[x] = src_ptr[x];
                }
            }
            break;
    }
}
enum kMakeborderType{
     BORDER_REPLICATE ,     //重复：对边界像素进行复制
     BORDER_REFLECT   ,     //反射：对感兴趣的图像中的像素在两边进行复制例如：fedcba|abcdefgh|hgfedcb
     BORDER_REFLECT_101 ,   //反射101：例子：gfedcb|abcdefgh|gfedcba
     BORDER_CONSTANT        //常量复制：例子：iiiiii|abcdefgh|iiiiiii
                    };
template <typename T>
void copyMakeBorder(const Image<T>& src,Image<T>& dst,int top,int bottom,int left,int right,int type,const T* _val= nullptr){
    dst.resize(src.width()+left+right,src.height()+top+bottom,src.channels());
    int width=src.width();
    int height=src.height();
    int channels=src.channels();
    switch (type){
        case BORDER_CONSTANT:
            T* val;
            if(_val== nullptr)
                val=new T[channels]{0};
            else
                val=_val;
            dst.fill_color(val,channels);
            for(int y=0;y<height;y++){
                const T* src_ptr=src.ptr(y);
                T* dst_ptr=dst.ptr(y+top);
                for(int x=0;x<width*channels;x++){
                    dst_ptr[left*channels+x]=src_ptr[x];
                }
            }
            break;
        case BORDER_REPLICATE:
            for(int y=0;y<top+bottom+height;y++){
                int y2=std::min(height-1,std::max(0,y-top));
                const T* src_ptr=src.ptr(y2);
                T* dst_ptr=dst.ptr(y);
                for(int x=0;x<left*channels;x++){
                    dst_ptr[x]=src[x%channels];
                }
                for(int x=left*channels,x2=0;x<(left+width)*channels;x++,x2++){
                    dst_ptr[x]=src_ptr[x2];
                }
                for(int x=(left+width)*channels;x<(left+right+width)*channels;x++){
                    dst_ptr[x]=src_ptr[(width-1)*channels+x%channels];
                }
            }
            break;
        case BORDER_REFLECT_101:
        case BORDER_REFLECT:
            if(top>=height||bottom>=height||left>=width||right>=width)
                throw("border width too big\n");
            for(int y=top,y2=0;y<top+height;y++,y2++){
                const T* src_ptr=src.ptr(y2);
                T* dst_ptr=dst.ptr(y);
                for(int x=0,x2=left;x<left;x++,x2--){
                    for(int c=0;c<channels;c++){
                     dst_ptr[x*channels+c]=src_ptr[x2*channels+c];
                    }
                }
                for(int x=left*channels,x2=0;x<(left+width)*channels;x++,x2++){
                    dst_ptr[x]=src_ptr[x2];
                }
                for(int x=left+width,x2=width-2;x<left+width+right;x++,x2--){
                    for(int c=0;c<channels;c++){
                        dst_ptr[x*channels+c]=src_ptr[x2*channels+c];
                    }
                }
            }
            for(int y=0,y2=top*2;y<top;y++,y2--){
                T* src_ptr=dst.ptr(y2);
                T* dst_ptr=dst.ptr(y);
                for(int x=0;x<(left+width+right)*channels;x++)
                    dst_ptr[x]=src_ptr[x];
            }
            for(int y=top+height,y2=top+height-2;y<top+height+bottom;y++,y2--){
                T* src_ptr=dst.ptr(y2);
                T* dst_ptr=dst.ptr(y);
                for(int x=0;x<(left+width+right)*channels;x++)
                    dst_ptr[x]=src_ptr[x];
            }
            break;
    }
}
/*
*  @property   图像转换
*  @func       RGB->HSL          L = max(R,G,B)
*  @param_in   image            待转换图像
*  @return     T
*/
//TODO:修改注释@anshuai
template <typename T>
T desaturateMaximum(T const* v);

/*
*  @property   图像转换
*  @func       RGB->HSL          L = 1/2 * (max(R,G,B)+mai(R,G,B))
*  @param_in   image            待转换图像
*  @return     T
*/
template <typename T>
T desaturateLightness(T const* v);

/*
*  @property   图像转换
*  @func       RGB->HSL          L = 0.21 * R + 0.72 * G + 0.07 * B
*  @param_in   image            待转换图像
*  @return     T
*/
template <typename T>
T desaturateLuminosity(T const* v);

/*
*  @property   图像转换
*  @func       RGB->HSL          L = 0.30 * R + 0.59 * G + 0.11 * B
*  @param_in   image            待转换图像
*  @return     T
*/
template <typename T>
T desaturateLuminance(T const* v);

/*
*  @property   图像转换
*  @func       RGB->HSL          L = 1/3 * (R + G +B)
*  @param_in   image            待转换图像
*  @return     T
*/
template <typename T>
T desaturateLverage(T const* v);

/*
*  @property   图像饱和度降低
*  @func       将图像转换为几种HSL图像
*  @param_in   image            待转换图像
*  @param_in   type             亮度类型
*  @typename   防止歧义，显示声明Image<T>::Ptr是类型而非变量
*  @return     Image<T>::ImagePtr
*/
template <typename T>
typename Image<T>::ImagePtr
desaturate(typename Image<T>::ConstImagePtr image,KDesaturateType type);

//TODO: 这里的sigma2=1/2完全满足３倍sigma，抖动没那么明显，可以用这个尺度
template <typename T>
typename Image<T>::ImagePtr
rescaleHalfSizeGaussian(typename Image<T>::ConstImagePtr image, float sigma2 = 0.75f);


/*
*  @property   图像模糊
*  @func       将对图像进行高斯模糊,运用高斯卷积核,进行可分离卷积,先对x方向进行卷积,再在y方向进行卷积,
*              等同于对图像进行二维卷积
*              该高斯核为高斯函数f(x,y)=1/[(2pi)*sigma^2] * e^-((x^2 + y^2)/2sigma2)
*
*
*  @param_in   in            　  待模糊图像
*  @param_in   sigma             目标高斯尺度值　　也就是标准差　
*  @typename   防止歧义，显示声明Image<T>::Ptr是类型而非变量
*  @return     Image<T>::ImagePtr
*/
template <typename T>
typename Image<T>::ImagePtr
blurGaussian(typename Image<T>::ConstImagePtr in, float sigma);
template <typename T>
void blurGaussian(const Image<T>* in,Image<T>* out,  float sigma);
/*
*  @property   图像模糊
*  @func       将对图像进行高斯模糊,运用高斯卷积核,进行可分离卷积,先对x方向进行卷积,再在y方向进行卷积,
*              等同于对图像进行二维卷积
*              该高斯核为高斯函数f(x,y)=1/[(2pi)*sigma^2] * e^-((x^2 + y^2)/2sigma2)
*

/*
*  @property   求图像差
*  @func       求差异图像的有符号图像,image_1 - image<T>
*  @param_in   image_1  image_2  相减的两幅图像
*  @return     Image<T>::ImagePtr
*/
template <typename T>
typename Image<T>::ImagePtr
subtract(typename Image<T>::ConstImagePtr image_1, typename Image<T>::ConstImagePtr image_2);

template <typename T>
void subtract(const Image<T>& image_1, const Image<T>& image_2,Image<T>& dst);

/*
*  @property   求图像差
*  @func       求差异图像的无符号图像,image_1 - image<T>
*  @param_in   image_1  image_2  相减的两幅图像
*  @return     Image<T>::ImagePtr
*/
template <typename T>
typename Image<T>::ImagePtr
subtractAbs(typename Image<T>::ConstImagePtr image_1, typename Image<T>::ConstImagePtr image_2);

/*
*  @property   扩展图像通道
*  @func       将灰度图拓展为RGB图或者RGBA图
*  @param_in   image         待拓展图像
*  @return     Image<T>::ImagePtr
*/
template <typename T>
typename Image<T>::ImagePtr
expandGrayscale(typename Image<T>::ConstImagePtr image);

template <typename _Tp1,typename _Tp2>
void converTo(const TypedImageBase<_Tp1>& src,TypedImageBase<_Tp2>& dst,float alpha=1.f,float offset=0.f);
IMAGE_NAMESPACE_END

/*******************************************************************
*~~~~~~~~~~~~~~~~~~~~~常用图像函数处理实现~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*******************************************************************/

IMAGE_NAMESPACE_BEGIN

template <typename T>
inline T
desaturateMaximum(T const* v)
{
    return *std::max_element(v, v + 3);
}

template <typename T>
inline T
desaturateLightness(T const* v)
{
    T const max = *std::max_element(v, v + 3);
    T const min = *std::min_element(v, v + 3);
    return 0.5f * (max + min);
}

template <typename T>
inline T
desaturateLuminosity(T const* v)
{
    return 0.21f * v[0] + 0.72f * v[1] + 0.07f * v[2];
}


template <typename T>
inline T
desaturateLuminance(T const* v)
{
    return 0.30 * v[0] + 0.59f * v[1] + 0.11f * v[2];
}


template <typename T>
inline T desaturateLverage(T const* v)
{
    return ((float)(v[0] + v[1] + v[2])) / 3.0f;
}


template <typename T>
typename Image<T>::ImagePtr desaturate(typename Image<T>::ConstImagePtr image, KDesaturateType type)
{
    if (image == NULL)
    {
        throw std::invalid_argument("无图像传入！");
    }

    if (image->channels() != 3 && image->channels() != 4)
    {
        throw std::invalid_argument("图像必须是RGB或者RGBA!");
    }

    bool has_alpha = (image->channels() == 4);

    typename Image<T>::ImagePtr out_image(Image<T>::create());
    out_image->allocate(image->width(),image->height(),1 + has_alpha);

    typedef T (*DesaturateFunc)(T const*);
    DesaturateFunc func;

    switch (type)
    {
        case DESATURATE_MAXIMUM:
            func = desaturateMaximum<T>;
            break;
        case DESATURATE_LIGHTNESS:
            func = desaturateLightness<T>;
            break;
        case DESATURATE_LUMINOSITY:
            func = desaturateLuminosity;
            break;
        case DESATURATE_LUMINANCE:
            func = desaturateLuminance;
            break;
        case DESATURATE_AVERAGE:
            func = desaturateLverage;
            break;
        default:
            throw std::invalid_argument("非法desaturate类型");
    }

    int out_pos = 0;
    int in_pos = 0;
    //TODO:: to be CUDA @Yang
    //opencv 4000*2250*3 图像处理时间: 14.4ms
    T* dst_ptr=out_image->getDataPointer();
    T const* v = image->getDataPointer();
    for (int i = 0; i < image->getPixelAmount(); ++i)
    {

        //out_image->at(out_pos) = func(v);
        dst_ptr[out_pos]=func(v+in_pos);

        if (has_alpha)
        {
            //out_image->at(out_pos + 1) = image->at(in_pos + 3);
            dst_ptr[out_pos+1]=v[in_pos+3];
        }

        out_pos += 1 + has_alpha;
        in_pos += 3 + has_alpha;
    }

    return out_image;
}


//TODO::to be CUDA@YANG
template <typename T>
typename Image<T>::ImagePtr
rescaleHalfSizeGaussian(typename Image<T>::ConstImagePtr image, float sigma2)
{
    int const iw = image->width();
    int const ih = image->height();
    int const ic = image->channels();
    int const ow = (iw + 1) >> 1;
    int const oh = (ih + 1) >> 1;

    if (iw < 2 || ih < 2)
    {
        throw std::invalid_argument("图像尺寸过小，不可进行降采样!\n");
    }

    typename Image<T>::ImagePtr out(Image<T>::create());
    out->allocate(ow,oh,ic);

    float const w1 = std::exp(-0.5 / (2.0f * sigma2));//0.5*0.5*2
    float const w2 = std::exp(-2.5f / (2.0 * sigma2));//0.5*0.5+1.5*1.5
    float const w3 = std::exp(-4.5f / (2.0f * sigma2));//1.5*1.5*2

    int out_pos = 0;
    int const rowstride = iw * ic;
    for (int y = 0; y < oh; ++y)
    {
        int y2 = (int)y << 1;
        T const *row[4];
        row[0] = &image->at(std::max(0,(y2 - 1) * rowstride));
        row[1] = &image->at(y2 * rowstride);
        row[2] = &image->at(std::min((int)ih - 1 ,y2 + 1) * rowstride);
        row[3] = &image->at(std::min((int)ih - 2 ,y2 + 2) * rowstride);

        for (int x = 0; x < ow; ++x)
        {
            int x2 = (int)x << 1;
            int xi[4];
            xi[0] =  std::max(0,x2 - 1) * ic;
            xi[1] = x2 * ic;
            xi[2] = std::min((int)iw - 1 ,x2 + 1) * ic;
            xi[3] = std::min((int)iw - 1 ,x2 + 2) * ic;

            for (int c = 0; c < ic; ++c)
            {
                float sum = 0.0f;//u_char溢出
                sum += row[0][xi[0] + c] * w3;
                sum += row[0][xi[1] + c] * w2;
                sum += row[0][xi[2] + c] * w2;
                sum += row[0][xi[3] + c] * w3;

                sum += row[1][xi[0] + c] * w2;
                sum += row[1][xi[1] + c] * w1;
                sum += row[1][xi[2] + c] * w1;
                sum += row[1][xi[3] + c] * w2;

                sum += row[2][xi[0] + c] * w2;
                sum += row[2][xi[1] + c] * w1;
                sum += row[2][xi[2] + c] * w1;
                sum += row[2][xi[3] + c] * w2;

                sum += row[3][xi[0] + c] * w3;
                sum += row[3][xi[1] + c] * w2;
                sum += row[3][xi[2] + c] * w2;
                sum += row[3][xi[3] + c] * w3;

                out->at(out_pos++) = sum / (float)(4 * w3 + 8 * w2 + 4 * w1);
            }
        }
    }
    return out;
}

//TODO::to be CUDA@YANG
//opencv 4000*2250*3 图像处理时间：60.9ms
template <typename T>
typename Image<T>::ImagePtr
blurGaussian(typename Image<T>::ConstImagePtr in, float sigma)
{
    if (in == nullptr||in->empty())
    {
        throw std::invalid_argument("没有输入图像!\n");
    }
    typename Image<T>::ImagePtr out=make_shared(in->width(),in->height(),in->channels());
    blurGaussian(in.get(),out.get(),sigma);
    return out;
}

template <typename T>
void blurGaussian(const Image<T>* in,Image<T>* out, float sigma)
{
    if (in == nullptr||in->empty())
    {
        throw std::invalid_argument("没有输入图像!\n");
    }
    if(out->empty())
    {
        out->resize(in->width(),in->height(),in->channels());
    }
    else if(out->width()!=in->width()||out->height()!=in->height()||out->channels()!=in->channels())
    {
        throw std::invalid_argument("输出图像尺寸应等于输入图像!\n");
    }


    if(MATH_EPSILON_EQ(sigma,0.0f,0.1f))
    {
        //out->getData().assign(in->getData().begain(),in->getData().end());
        out->resize(in->width(),in->height(),in->channels());
        int amount=in->getValueAmount();
        for(int i=0;i<amount;i++)
            out->getData()[i]=in->getData()[i];
        return;
    }

    int const w = in->width();
    int const h = in->height();
    int const c = in->channels();
    int const ks = std::ceil(sigma * 2.884f);
    std::vector<float> kernel(ks + 1);
    float weight = 0;

    for (int i = 0; i < ks + 1; ++i)
    {
        kernel[i] = math::func::gaussian((float)i, sigma);
        weight += kernel[i]*2;
    }
    weight-=kernel[0];
    //可分离高斯核实现
    //x方向对对象进行卷积
    Image<float>::ImagePtr sep(Image<float>::create(w,h,c));
    int px = 0;
    for (int y = 0; y < h; ++y)
    {
        const T* src_ptr=in->ptr(y);
        float* dst_ptr=sep->ptr(y);
        for (int x = ks; x < w-ks; ++x,++px)
        {
            for (int cc = 0; cc < c; ++cc)
            {
                float accum=0;
                for (int i = -ks; i <=ks; ++i)
                {
                    //int idx = math::func::clamp(x + i,0,w - 1);
                    int idx=x+i;
                    //accum += in->at(y * w + idx, cc) * kernel[abs(i)];
                    accum+=src_ptr[idx*c+cc]*kernel[abs(i)];
                }
                //sep->at(px,cc) = accum / weight;
                dst_ptr[x*c+cc]=accum/weight;
            }
        }
    }
    //y方向对图像进行卷积
    px=0;
    for (int y = ks; y < h-ks; ++y)
    {
        T* dst_ptr=out->ptr(y);
        for (int x = 0; x < w; ++x,++px)
        {
            for (int cc = 0; cc < c; ++cc)
            {
                float accum =0;
                for (int i = -ks; i <= ks; ++i)
                {
                    //int idx = math::func::clamp(y+i,0,(int)h - 1);
                    //accum += sep->at(idx * w + x, cc)* kernel[abs(i)];
                    int idx=y+i;
                    accum+=sep->ptr(idx)[x*c+cc]*kernel[abs(i)];
                }
                //out->at(px,cc) = (T)(accum / weight);
                dst_ptr[x*c+cc]=(T)(accum/weight);
            }
        }
    }
    return;
}


//TODO::to be CUDA@YANG
//opencv 4000*2250*3 图像处理时间：4.76ms
template <typename T>
typename Image<T>::ImagePtr
subtract(typename Image<T>::ConstImagePtr image_1, typename Image<T>::ConstImagePtr image_2)
{
    if (image_1 == nullptr || image_2 == nullptr)
    {
        throw std::invalid_argument("至少有一幅图像为空!不满足要求!\n");
    }
    int const w1 = image_1->width();
    int const h1 = image_1->height();
    int const c1 = image_1->channels();

    if(w1 != image_2->width() || h1 != image_2->height() || c1 != image_2->channels())
    {
        throw std::invalid_argument("两图像尺寸不匹配!\n");
    }
    if(typeid(T)== typeid(uint8_t)
    || typeid(T)== typeid(uint16_t)
    || typeid(T)== typeid(uint32_t)
    || typeid(T)== typeid(uint64_t)){
        throw std::invalid_argument("无符号图像不满足要求!\n");
    }

    typename Image<T>::ImagePtr out(Image<T>::create());
    out->allocate(w1,h1,c1);
    const T* image_1_ptr=image_1->getDataPointer();
    const T* image_2_ptr=image_2->getDataPointer();
    T* out_ptr=out->getDataPointer();
    for (int i = 0; i < image_1->getValueAmount(); ++i)
    {
        out_ptr[i]=image_1_ptr[i]-image_2_ptr[2];
    }

    return out;
}

template <typename T>
void subtract(const Image<T>& image_1,const Image<T>& image_2,Image<T>& dst)
{
    if (image_1.empty() || image_2.empty())
    {
        throw std::invalid_argument("至少有一幅图像为空!不满足要求!\n");
    }
    int const w1 = image_1.width();
    int const h1 = image_1.height();
    int const c1 = image_1.channels();

    if(w1 != image_2.width() || h1 != image_2.height() || c1 != image_2.channels())
    {
        throw std::invalid_argument("两图像尺寸不匹配!\n");
    }
    if(typeid(T)== typeid(uint8_t)
       || typeid(T)== typeid(uint16_t)
       || typeid(T)== typeid(uint32_t)
       || typeid(T)== typeid(uint64_t)){
        throw std::invalid_argument("无符号图像不满足要求!\n");
    }


    dst.resize(w1,h1,c1);
    const T* image_1_ptr=image_1.getDataPointer();
    const T* image_2_ptr=image_2.getDataPointer();
    T* out_ptr=dst.getDataPointer();
    for (int i = 0; i < image_1.getValueAmount(); ++i)
    {
        //dst.at(i) = image_1.at(i) - image_2.at(i);
        out_ptr[i]=image_1_ptr[i]-image_2_ptr[i];
    }
}

//TODO::to be CUDA@YANG
//opencv 4000*2250*3 图像处理时间：3.34ms
template <typename T>
typename Image<T>::ImagePtr
subtractAbs(typename Image<T>::ConstImagePtr image_1, typename Image<T>::ConstImagePtr image_2)
{

    if (image_1 == nullptr || image_2 == nullptr)
    {
        throw std::invalid_argument("至少有一幅图像为空!不满足要求!\n");
    }
    int const w1 = image_1->width();
    int const h1 = image_1->height();
    int const c1 = image_1->channels();

    if(w1 != image_2->width() || h1 != image_2->height() || c1 != image_2->channels())
    {
        throw std::invalid_argument("两图像尺寸不匹配!\n");
    }

    typename Image<T>::ImagePtr out(Image<T>::create());
    out->allocate(w1,h1,c1);
    const T* image_1_ptr=image_1->getDataPointer();
    const T* image_2_ptr=image_2->getDataPointer();
    T* out_ptr=out->getDataPointer();
    for (int i = 0; i < image_1->getValueAmount(); ++i)
    {
       out_ptr[i]=abs(image_1_ptr[i]-image_2_ptr[i]);
    }

    return out;
}

template <typename T>
typename Image<T>::ImagePtr
expandGrayscale(typename Image<T>::ConstImagePtr image)
{
    //typename Image<T>::ImagePtr image_ptr = Image<T>::create(*image);
    return image;
}

template <typename _Tp1,typename _Tp2>
void converTo(const TypedImageBase<_Tp1>& src,TypedImageBase<_Tp2>& dst,float alpha,float offset)
{
    if(dst.empty())
        dst.resize(src.width(),src.height(),src.channels());
    else {
        if (dst.height() != src.height()
            || dst.width() != src.width()
            || dst.channels() != src.channels()) {
            throw ("convert func need same size\n");
        }
    }
    float EPSILON =1.192093e-007;

    const _Tp1* src_ptr=src.getDataPointer();
    _Tp2* dst_ptr=dst.getDataPointer();
    if(alpha-1.f>EPSILON){
        for(int i=0;i<src.getValueAmount();i++)
            dst_ptr[i]=(_Tp2)(src_ptr[i]+offset);
    }else{
        for(int i=0;i<src.getValueAmount();i++)
            dst_ptr[i]=(_Tp2)(src_ptr[i]*alpha+offset);
    }


}

IMAGE_NAMESPACE_END

#endif //IMAGE_IMAGE_PROCESS_HPP
