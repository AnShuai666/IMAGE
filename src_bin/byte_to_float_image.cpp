/*
 * @功能      删除图像颜色通道函数cpu实现与gpu实现对比
 * @姓名      杨丰拓
 * @日期      2019-5-13
 * @时间      16:33
 * @邮箱
*/
#include "IMAGE/image_io.h"
#include "MATH/Util/timer.h"
#include "IMAGE/image_process.hpp"
#include <iostream>
#include <cuda_include/image_process_1.cuh>
#include "cuda_include/common.cuh"
using namespace std;
int main(int argc, char ** argv) {
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
    //CPU实现
    FloatImage::Ptr dst_cpu;

    util::TimerHigh time;
    dst_cpu=image::byte_to_float_image(image);//执行cpu功能
    cout<<"cpu:"<<time.get_elapsed()<<"ms"<<endl;

    //GPU实现
    warmUp();//无NVVP检测核函数时间时需调用此函数,防止检测显卡启动的时间
    
    FloatImage dst_gpu(image->width(),image->height(),image->channels());
    util::TimerHigh time1;
    byteToFloatImageByCuda(&dst_gpu.at(0),&image->at(0),image->width(),image->height(),image->channels());
    cout<<"gpu:"<<time1.get_elapsed()<<"ms"<<endl;
    
    int wc=(image->width())*(image->channels());
    compare1(&dst_gpu.at(0),&dst_cpu->at(0),wc,image->height(),true);
}