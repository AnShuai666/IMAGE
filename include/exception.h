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

class Exception : public std::exception
{
public:
    Exception();
    virtual ~Exception(void);
    virtual const char* what() const noexcept
    {
        std::cout<<"Got an Exception!"<<std::endl;
    }
};

class FileException : public std::exception
{
public:
    FileException(const std::string errMsg = "")
        :_errMsg(errMsg){}
    virtual ~FileException(void);
    virtual const char* what() const noexcept
    {
        std::cout<<"File Exception at: "<<_errMsg<<std::endl;
    }
    std::string _errMsg;
};
IMAGE_NAMESPACE_END

#endif //IMAGE_EXCEPTION_H
