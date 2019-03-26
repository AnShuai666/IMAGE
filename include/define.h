/*
 * @desc    宏定义
 * @author  安帅
 * @date    2018-12-12
 * @email   1028792866@qq.com
*/

#ifndef IMAGE_DEFINE_H
#define IMAGE_DEFINE_H

#include <iostream>
#define IMAGE_NAMESPACE_BEGIN namespace image {
#define IMAGE_NAMESPACE_END }

#define checkImageioerror(val) checkimage((val),#val,__FILE__,__LINE__);

#define checkFileerror(fp) checkfile((fp),__FILE__,__LINE__)

template <typename T>
void checkimage(T val, char const* const funcname,char const* const filename,int const linenum);

void
checkfile(FILE *fp, char const* const filename, int const linenum);

template <typename T>
void checkimage(T val, char const* const funcname,char const* const filename,int const linenum)
{
    if(val != 1)
    {
        std::cerr<<"IMAGE IO ERROR AT: "<<filename<<": "<<linenum<<std::endl;
        std::cerr<<"ERROR FUNCNAME IS: "<<funcname<<std::endl;
    }
    return;
}


#endif //IMAGE_DEFINE_H
