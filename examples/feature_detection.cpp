/*
 * @desc    sift特征提取检测
 * @author  安帅
 * @date    2019-03-27
 * @email   1028792866@qq.com
*/
#include "sift.hpp"
#include "image_io.h"
#include "timer.h"
#include <iostream>

//自定义排序函数 描述子尺度从大到小排序
bool scale_compare(image::Sift::Descriptor const& d1, image::Sift::Descriptor const& d2)
{
    return d1.scale > d2.scale;
}


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

    std::cout<<"================================="<<std::endl;
    std::cout<<"           Debug专用              "<<std::endl;
    std::cout<<"================================="<<std::endl;

    image::Sift::Keypoints sift_keypoints;
    image::Sift::Descriptors sift_descriptors;

    image::Sift::Options sift_options;
    sift_options.verbose_output = true;
    sift_options.debug_output = true;
    image::Sift sift(sift_options);
    sift.set_image(image);

    image::TimerHigh timer;
    sift.process();
    std::cout<<"计算Sift特征用时 "<<timer.get_elapsed()<<" 毫秒"<<std::endl;

    sift_keypoints = sift.get_keypoints();
    sift_descriptors = sift.get_descriptors();

    //TODO:sort函数 CUDA@杨丰拓
    std::sort(sift_descriptors.begin(),sift_descriptors.end(),scale_compare);

    std::cout<<"================================="<<std::endl;
    std::cout<<"           Debug专用              "<<std::endl;
    std::cout<<"================================="<<std::endl;

    return 0;
}

