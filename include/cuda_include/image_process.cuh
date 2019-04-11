/*
 * @desc    基于CUDA加速的图像处理函数
 * @author  杨丰拓
 * @date    2019-04-01
 * @email   yangfengtuo@163.com
*/
#ifndef IMAGE_PROCESS_CUH
#define IMAGE_PROCESS_CUH
#include <cstdio>
#include <iostream>



//opencv 4000*2250*3 图像处理时间: 14.4ms
//调用desaturate_by_cuda函数处理时间：36ms,其中H2D 27ms, D2H 8ms,kernel 0.7ms
//串行实现时间：约200ms
/*
 * @property    图像饱和度降低
 * @func        将图像转换为几种HSL图像
 * @param_out   out_image          转换后的图像
 * @param_in    in_image           待转换图像
 * @param_in    pixel_amount       像素点个数
 * @param_in    type               亮度类型
 * @param_in    alpha              是否有alpha通道
 */
//void desaturate_by_cuda(float * const out_image,float const  *in_image,const int pixel_amount, const int type,const bool alpha);
void warm(void);

/**
 * @property    图像缩放
 * @func        将图像放大为原图两倍  均匀插值，最后一行后最后一列与倒数第二行与倒数第二列相同
 *                                 偶数行列时：取(y/2,x/2)d的像素点
 *                                 奇数行列时：
 *                                      奇数行：(,x/2)与(,x/2+1)的平均值
 *                                      偶数行：(y/2,)与(y/2+1,)的平均值
 * @param_out   out_image          放大后的图像首地址
 * @param_in    in_image           待放大图像首地址
 * @param_in    weight             输入图像的宽度
 * @param_in    height             输入图像的高度
 * @param_in    channels           输入图像的颜色通道数
 * 调用示例：
 * double_size_by_cuda(&out_image->at(0),&img->at(0),img->width(),img->height(),img->channels());
 */
void double_size_by_cuda(float * const out_image,float const  * const in_image, int const weight,int const height,int const channels);

/**
 * @property    图像缩放
 * @func        将图像缩小为原图两倍　 像素点为(2*y,2*x)(2*y,2*x+1)(2*y+1,2*x)(2*y+1,2*x+1)的平均值
 *                                 若最后一行或最后一列为奇数列．则越界部分再取最后一行或最后一列
 * @param_out   out_image          放大后的图像首地址
 * @param_in    in_image           待放大图像首地址
 * @param_in    weight             输入图像的宽度
 * @param_in    height             输入图像的高度
 * @param_in    channels           输入图像的颜色通道数
 * 调用示例：
 * halfsize_by_cuda(&out_image->at(0),&img->at(0),img->width(),img->height(),img->channels());
 */
void halfsize_by_cuda(float * const out_image,float const  * const in_image, int const weight,int const height,int const channels);
#endif //IMAGE_PROCESS_CUH