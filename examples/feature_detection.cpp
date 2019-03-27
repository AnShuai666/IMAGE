/*
 * @desc    sift特征提取检测
 * @author  安帅
 * @date    2019-03-27
 * @email   1028792866@qq.com
*/
#include "sift.hpp"
int main(int argc, char ** argv)
{
    if (argc < 2)
    {
        std::cerr<<"使用方法: "<<argv[0]<<" <图像名>"<<std::endl;
        return 1;
    }

    image::ByteImage::Ptr image;
    return 0;
}