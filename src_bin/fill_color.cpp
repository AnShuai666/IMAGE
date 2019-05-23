/*
 * @功能      图像填充cpu实现与gpu实现对比
 * @姓名      杨丰拓
 * @日期      2019-4-29
 * @时间      17:04
 * @邮箱
*/

#include "IMAGE/image_io.h"
#include "MATH/Util/timer.h"
#include "IMAGE/image_process.hpp"
#include <iostream>

#include "cuda_include/image_1.cuh"

using namespace std;
int main(int argc, char ** argv)
{
    cout<<"*************调试开始*************"<<endl;
    char color[3]={12,10,18};
    Image<char> src(4000,2250,3);
    //printf("%d\n",src.at(0));
    util::TimerHigh time;
    src.fill_color(color,3);
    cout<<time.get_elapsed()<<"ms"<<endl;


    cout<<"///////////gpu实现////////////"<<endl;
    Image<char> src1(4000,2250,3);
    fill_color_by_cuda(&src1.at(0),color,src1.width(),src1.height(),src1.channels(),3,&src.at(0));
    cout<<"*************调试结束*************"<<endl;

    return  0;
}