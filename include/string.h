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

/*******************************************************************
*~~~~~~~~~~~~~~~~~~~~~常用字符串处理函数声明~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*******************************************************************/
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

IMAGE_NAMESPACE_BEGIN

/*******************************************************************
*~~~~~~~~~~~~~~~~~~~~~常用字符串处理函数实现~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*******************************************************************/
std::string
left(std::string const &str, std::size_t size)
{
    return str.substr(0,size);
}

std::string
right(std::string const &str, std::size_t size)
{
    unsigned long sub_size=str.size() > size?(str.size() - size):0;
    return str.substr(sub_size);
}

std::string
lowercase(std::string const &str)
{
    std::string string(str);
    for (int i = 0; i < str.size(); ++i)
    {
        if(string[i] >= 0x41 && string[i] <= 0x5a)
        {
            string[i] += 0x20;
        }
    }
    return string;
}

std::string
uppercase(std::string const &str)
{
    std::string string(str);
    for (int i = 0; i < str.size(); ++i)
    {
        if(string[i] >= 0x61 && string[i] <= 0x7a)
        {
            string[i] -= 0x20;
        }
    }
    return string;
}
IMAGE_NAMESPACE_END

#endif //IMAGE_STRING_H
