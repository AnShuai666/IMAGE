/*
 * @desc    图像文件读取异常类
 * @author  安帅
 * @date    2018-12-12
 * @email   1028792866@qq.com
*/

#ifndef IMAGE_EXCEPTION_H
#define IMAGE_EXCEPTION_H

#include <exception>
#include <string>
#include "define.h"

IMAGE_NAMESPACE_BEGIN

class Exception : public std::exception, public std::string
{
public:

};

IMAGE_NAMESPACE_END

#endif //IMAGE_EXCEPTION_H
