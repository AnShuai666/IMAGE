/*
 * @desc    sift特征提取检测
 * @author  安帅
 * @date    2019-03-27
 * @email   1028792866@qq.com
*/
#include "sift.hpp"
#include "image_io.h"
#include <iostream>
int main(int argc, char ** argv)
{
    if (argc < 2)
    {
        std::cerr<<"使用方法: "<<argv[0]<<" <图像名>"<<std::endl;
        return 1;
    }

    image::ByteImage::Ptr image;
    std::string image_filename = argv[1];

    try
    {
        std::cout<<"加载 "<<image_filename<<"中..."<<std::endl;
        image = image::load_image(image_filename);
    }
    catch (std::exception& e)
    {
        std::cerr<<"错误: "<<e.what()<<std::endl;
        return 1;
    }

    image::Sift::Keypoints sift_keypoints;
    image::Sift::Descriptors sift_descriptors;

    image::Sift::Options sift_options;
    sift_options.verbose_output = true;
    sift_options.debug_output = true;



    return 0;
}

