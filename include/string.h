/*
 * @desc    常用字符串处理函数
 * @author  安帅
 * @date    2019-01-07
 * @email   1028792866@qq.com
*/

#ifndef IMAGE_STRING_H
#define IMAGE_STRING_H

#include "define.h"
#include <string>
IMAGE_NAMESPACE_BEGIN

/*
*  @property   截取字符串
*  @func       截取字符串左边size个字符
*  @param_in   str         待截取字符串
*  @param_in   size        待截取字符个数
*  @return     std::string
*/
std::string left(std::string const &str,std::size_t size);

/*
*  @property   截取字符串
*  @func       截取字符串右边size个字符
*  @param_in   str         待截取字符串
*  @param_in   size        待截取字符个数
*  @return     std::string
*/
std::string right(std::string const &str,std::size_t size);

/*
*  @property   字符小写转换
*  @func       将字符串转换为小写
*  @param_in   str         待转换字符串
*  @return     std::string
*/
std::string lowercase(std::string const &str);

/*
*  @property   字符大写转换
*  @func       将字符串转换为大写
*  @param_in   str         待转换字符串
*  @return     std::string
*/
std::string uppercase(std::string const &str);

IMAGE_NAMESPACE_END
#endif //IMAGE_STRING_H
