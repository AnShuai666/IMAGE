/*
 * @功能      图像拷贝颜色通道cpu实现与gpu实现对比
 * @姓名      杨丰拓
 * @日期      2019-5-9
 * @时间      15:17
 * @邮箱
*/

#include "IMAGE/image_io.h"
#include "MATH/Util/timer.h"
#include "IMAGE/image_process.hpp"
#include <iostream>

#include "cuda_include/image_1.cuh"
#include "cuda_include/common.cuh"

using namespace std;
int main(int argc, char ** argv)
{
    int const w=4000;
    int const h=2250;
    int const c=3;

    Image<char> src_cpu(w,h,c);
    Image<char> src_gpu(w,h,c);
    int copy_c=0;
    int paste_c=1;
    //int paste_c=-1;
    char color[3]={12,25,36};
    src_cpu.fillColor(color,c);
    util::TimerHigh time;
    src_cpu.copyChannel(copy_c,paste_c);
    cout<<time.get_elapsed()<<"ms"<<endl;

    cout<<"*********************gpu实现*********************"<<endl;
    warmUp();

    src_gpu.fillColor(color,c);
    util::TimerHigh time1;
    copyChannelsByCuda(&src_gpu.at(0),src_gpu.width(),src_gpu.height(),src_gpu.channels(),copy_c,paste_c);
    cout<<time1.get_elapsed()<<"ms"<<endl;

    int wc=(src_gpu.width())*(src_cpu.channels());
    compare1(&src_gpu.at(0),&src_cpu.at(0),wc,src_cpu.height(),false);
}