/*
 * @功能      图像添加通道cpu实现与gpu实现对比
 * @姓名      杨丰拓
 * @日期      2019-5-5
 * @时间      15:52
 * @邮箱
*/

#include "image_io.h"
#include "Util/timer.h"
#include "image_process.hpp"
#include <iostream>

#include "cuda_include/image.cuh"

using namespace std;
int main(int argc, char ** argv)
{
    int num_channels=1;
    char value=2;
    cout<<"*************调试开始*************"<<endl;
    char color[3]={12,10,18};
    Image<char> src(4000,2250,3);
    src.fill_color(color,3);

    util::TimerHigh time;
    src.add_channels(num_channels, value);
    cout<<time.get_elapsed()<<"ms"<<endl;

    cout<<"///////////gpu实现////////////"<<endl;
    Image<char> src1(4000,2250,3);
    fill_color_by_cuda(&src1.at(0),color,src1.width(),src1.height(),src1.channels(),3,&src.at(0));
    Image<char> dst(4000,2250,3+num_channels);

    add_channels_by_cuda(&dst.at(0),&src1.at(0),src1.width(),src1.height(),src1.channels(),num_channels,value,&src.at(0));
    cout<<"*************调试结束*************"<<endl;
    return  0;
}