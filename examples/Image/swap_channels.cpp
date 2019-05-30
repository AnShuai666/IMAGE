/*
 * @功能      图像交换通道cpu实现与gpu实现对比
 * @姓名      杨丰拓
 * @日期      2019-5-8
 * @时间      13:40
 * @邮箱
*/
#include "IMAGE/image.hpp"
#include "MATH/Util/timer.h"
#include "IMAGE/image_process.hpp"
#include <iostream>
#include <vector>
#include "cuda_include/image_1.cuh"
#include "cuda_include/common.cuh"
using namespace std;
int main(int argc, char ** argv)
{
    cout<<"*************调试开始*************"<<endl;
    int width=4000;
    int height=2250;
    int channels=3;
    char color[3]={12,10,25};
    image::Image<char> src_cpu(width,height,channels);//创建初始图像(用于cpu)
    src_cpu.fillColor(color,channels);//填充图像

    image::Image<char> src_gpu(width,height,channels);//创建初始图像(用于gpu)
    src_gpu.fillColor(color,channels);//填充图像

    ///CPU实现及时间检测
    util::TimerHigh time;
    src_cpu.swapChannels(0,1,image::AT);//交换颜色通道(AT模式)
    cout<<time.get_elapsed()<<"ms"<<endl;

    cout<<"///////////gpu实现////////////"<<endl;

    warmUp();

    util::TimerHigh time1;
    swapChannelsByCuda(&src_gpu.at(0),src_gpu.width(),src_gpu.height(),src_gpu.channels(),0,1);
    cout<<"gpu:"<<time1.get_elapsed()<<"ms"<<endl;

    int wc=(src_cpu.width())*(src_cpu.channels());
    compare1(&src_gpu.at(0),&src_cpu.at(0),wc,src_cpu.height(),false);
    cout<<"*************调试结束*************"<<endl;
    return  0;
}

