/*
 * @desc    基于CUDA加速的图像处理函数
 * @author  杨丰拓
 * @date    2019-04-01
 * @email   yangfengtuo@163.com
*/
#ifndef IMAGE_PROCESS_1_CUH
#define IMAGE_PROCESS_1_CUH


//opencv 4000*2250*3 图像处理时间: 14.4ms
//调用desaturate_by_cuda函数处理时间：36ms,其中H2D 27ms, D2H 8ms,kernel 0.7ms
//串行实现时间：约200ms
/**
 * @property    图像饱和度降低
 * @func        将图像转换为几种HSL图像
 * @param_out   p_out_image          转换后的图像
 * @param_in    in_image           待转换图像
 * @param_in    pixel_amount       像素点个数
 * @param_in    type               亮度类型
 * @param_in    alpha              是否有alpha通道
 */
void desaturateByCuda(float * const p_out_image,float const  *p_in_image,const int kPixel_amount, const int kType,const bool kAlpha);


/**
 * @property    图像缩放
 * @func        将图像放大为原图两倍  均匀插值，最后一行后最后一列与倒数第二行与倒数第二列相同
 *                                 偶数行列时：取(y/2,x/2)d的像素点
 *                                 奇数行列时：
 *                                      奇数行：(,x/2)与(,x/2+1)的平均值
 *                                      偶数行：(y/2,)与(y/2+1,)的平均值
 * @param_out   p_out_image          放大后的图像首地址
 * @param_in    p_in_image           待放大图像首地址
 * @param_in    kWidth              输入图像的宽度
 * @param_in    kHeight             输入图像的高度
 * @param_in    kChannels           输入图像的颜色通道数
 * @param_in    out                cpu计算结果，用于对比数据
 * 调用示例：
 * doubleSizeByCuda(&p_out_image->at(0),&img->at(0),img->width(),img->height(),img->channels());
 */
void doubleSizeByCuda(float * const p_out_image,float const  * const p_in_image,int const kWidth,int const kHeight,int const kChannels);
/**
 * @property    图像缩放
 * @func        将图像缩小为原图1/2倍　 像素点为(2*y,2*x)(2*y,2*x+1)(2*y+1,2*x)(2*y+1,2*x+1)的平均值
 *                                 若最后一行或最后一列为奇数列．则越界部分再取最后一行或最后一列
 * @param_out   p_out_image          放大后的图像首地址
 * @param_in    p_in_image           待放大图像首地址
 * @param_in    kWidth              输入图像的宽度
 * @param_in    kHeight             输入图像的高度
 * @param_in    kChannels           输入图像的颜色通道数
 * @param_in    out                cpu计算结果，用于对比数据
 * 调用示例：
 * halfSizeByCuda(&p_out_image->at(0),&img->at(0),img->width(),img->height(),img->channels());
 */
void halfSizeByCuda(float * const p_out_image,float const  * const p_in_image,int const kWidth,int const kHeight,int const kChannels);

/**
 * @property    图像缩放
 * @func        将图像缩小为原图1/2倍　 输出图像的每一个像素点都由输入图像上的4*4个像素值与4*4的高斯模板卷积
 *                                  高斯模板:
 *                                      w[2] w[1] w[1] w[2]
 *                                      w[1] w[0] w[0] w[1]
 *                                      w[1] w[0] w[0] w[1]
 *                                      w[2] w[1] w[1] w[2]
 *                                  输入图像上选定模板大小的y坐标：
 *                                      y[0] = max(0, 2*y - 1);
 *                                      y[1] = 2*y;
 *                                      y[2] = min(2*y + 1, (int)height - 1);
 *                                      y[3] = min(2*y + 2, (int)height - 2);
 *                                  输入图像上选定模板大小的x坐标：
 *                                      x[0] = max(0, 2*x - 1);
 *                                      x[1] = 2*x;
 *                                      x[2] = min(2*x + 1, (int)weight - 1);
 *                                      x[3] = min(2*x + 2, (int)weight - 1);
 * @param_out   p_out_image          放大后的图像首地址
 * @param_in    p_in_image           待放大图像首地址
 * @param_in    kWidth              输入图像的宽度
 * @param_in    kHeight             输入图像的高度
 * @param_in    kChannels           输入图像的颜色通道数
 * @param_in    sigma2             目标高斯尺度平方值　　也就是方差
 * @param_in    out                cpu计算结果，用于对比数据
 * 调用示例：
 * halfsizeGuassianByCuda(&p_out_image->at(0),&img->at(0),img->width(),img->height(),img->channels(),sigma2);
 */
void halfSizeGaussianByCuda(float * const p_out_image,float const  * const p_in_image, int const kWidth,int const kHeight,int const kChannels,float sigma2);
/**
 * @property    分离式高斯模糊函数
 * @func        对图像进行高斯模糊    高斯核为高斯函数f(x,y)=1/[(2pi)*sigma^2] * e^-((x^2 + y^2)/2sigma2)
 *                                 先沿x维方向对输入图像进行高斯模糊，再对生成的图像进行y维高斯模糊操作
 *                                 通过输入的sigma计算出高斯核中心点的左右两侧的元素个数ks，计算高斯权值及高斯权值的和weight
 *                                 x维：
 *                                 x维模糊图片的每个像素点为输入图片上的(x-ks)到(x+ks)的值乘以高斯权值和再除以weight
 *                                      边界考虑：x-ks>0,x+ks<w-1
 *                                 y维：
 *                                 输出图片的每个像素点为x维模糊图片上的(y-ks)到(y+ks)的值乘以高斯权值和再除以weight
 *                                      边界考虑：y-ks>0,y+ks<w-1
 * @param_out   p_out_image        放大后的图像首地址
 * @param_in    p_in_image         待放大图像首地址
 * @param_in    kWidth             输入图像的宽度
 * @param_in    kHeight            输入图像的高度
 * @param_in    kChannels          输入图像的颜色通道数
 * @param_in    sigma              目标高斯尺度值　　也就是标准差　
 * @param_in    out                cpu计算结果，用于对比数据
 * @return
 * 调用示例：
 * blurGaussianByCuda(&p_out_image->at(0),&img->at(0),img->width(),img->height(),img->channels(),sigma);
 */
int blurGaussianByCuda(float * const p_out_image,float const  * const p_in_image, int const kWidth,int const kHeight,int const kChannels,float sigma);

/**
 * @property    分离式高斯模糊函数
 * @func        对图像进行高斯模糊    sigma = sqrt(sigma2);
 *                                 将对图像进行高斯模糊,运用高斯卷积核,进行可分离卷积,先对x方向进行卷积,再在y方向进行卷积,
 *                                 等同于对图像进行二维卷积
 *                                 该高斯核为高斯函数f(x,y)=1/[(2pi)*sigma^2] * e^-((x^2 + y^2)/2sigma2)
 * @param_out   p_out_image        放大后的图像首地址
 * @param_in    p_in_image         待放大图像首地址
 * @param_in    kWidth             输入图像的宽度
 * @param_in    kHeight            输入图像的高度
 * @param_in    kChannels          输入图像的颜色通道数
 * @param_in    sigma2             目标高斯尺度平方值　　也就是方差
 * @param_in    out                cpu计算结果，用于对比数据
 * @return
 * 调用示例：
 * blurGaussian2ByCuda(&p_out_image->at(0),&img->at(0),img->width(),img->height(),img->channels(),sigma2);
 */
int blurGaussian2ByCuda(float * const p_out_image,float const  * const p_in_image, int const kWidth,int const kHeight,int const kChannels,float sigma2);

/***
 * @property    求图像差函数
 * @func        求差异图像的有符号图像,in_image1-in_image2
 * @param_out   p_out_image     图像差
 * @param_in    p_in_image1     输入图像1
 * @param_in    p_in_image2     输入图像2
 * @param_in    kWidth          输入图像的宽度
 * @param_in    kHeight         输入图像的高度
 * @param_in    kChannels       输入图像的颜色通道数
 * @return
 * 调用示例：
 * subtractByCuda(&p_out_image->at(0),&img1->at(0),&img2->at(0),img->width(),img->height(),img->channels());
 */
int subtractByCuda(float * const p_out_image,float const  * const p_in_image1,float const  * const p_in_image2, int const kWidth,int const kHeight,int const kChannels);
/***
 * @property    求图像差函数
 * @func        求差异图像的无符号图像,|in_image1-in_image2|
 * @paramout     p_out_image     图像差
 * @param_in     p_in_image1     输入图像1
 * @param_in     p_in_image2     输入图像2
 * @param_in     w             输入图像的宽度
 * @param_in     h             输入图像的高度
 * @param_in     c             输入图像的颜色通道数
 * @return
 * 调用示例：
 * differenceByCuda(&p_out_image->at(0),&img1->at(0),&img2->at(0),img->width(),img->height(),img->channels());
 */
template <typename T>
int differenceByCuda(T * const p_out_image,T const  * const p_in_image1,T const  * const p_in_image2, int const kWidth,int const kHeight,int const kChannels);
template <>
int differenceByCuda<float>(float * const p_out_image,float const  * const p_in_image1,float const  * const p_in_image2, int const kWidth,int const kHeight,int const kChannels);
template <>
int differenceByCuda<char>(char * const p_out_image,char const  * const p_in_image1,char const  * const p_in_image2, int const kWidth,int const kHeight,int const kChannels);

/**
 * @property    图像变换
 * @func        将图像中位图转换为浮点图像，灰度值范围从[0-255]->[0.0,1.0]
 * @param_out   dstImage    输出图像
 * @param_in    srcImage    输入图像
 * @param_in    kWidth      输入图像的宽度
 * @param_in    kHeight     输入图像的高度
 * @param_in    kChannels   输入图像的颜色通道数
 * @param_in    contrast    cpu计算结果，用于对比数据
 * @return
 */
int byteToFloatImageByCuda(float * p_dstImage,unsigned char *p_srcImage,int const kWidth,int const kHeight,int const kChannels);
#endif //IMAGE_PROCESS_1_CUH