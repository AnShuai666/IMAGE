/*
 * @功能      删除图像颜色通道函数cpu实现与gpu实现对比
 * @姓名      杨丰拓
 * @日期      2019-5-10
 * @时间      16:56
 * @邮箱
*/

#include "IMAGE/image_io.h"
#include "MATH/Util/timer.h"
#include "IMAGE/image_process.hpp"
#include <iostream>
#include <vector>
#include "cuda_include/image_1.cuh"

using namespace std;
int main(int argc, char ** argv)
{
    int const w=4000;
    int const h=2250;
    int const c=3;
    char color[3]={12,25,36};

    Image<char> src_cpu(w,h,c);//创建cpu处理图像
    Image<char> src_gpu(w,h,c);//创建gpu处理图像
    int del_c=1;//所要删除的通道

    src_cpu.fill_color(color,c);//填充cpu处理图像
    src_gpu.fill_color(color,c);//填充gpu处理图像
    /*for (int i = 0; i <10 ; ++i) {
        for (int k = 0; k <src_cpu.channels(); ++k) {
            cout<<(int)src_cpu.at(i,0,k)<<"\t";
        }
        cout<<endl;
    }*/
    util::TimerHigh time;
    src_cpu.delete_channel(del_c);//执行cpu删除功能
    cout<<time.get_elapsed()<<"ms"<<endl;

    cout<<"*********************gpu实现*********************"<<endl;
    Image<char> dst_gpu(w,h,c-1);//创建gpu处理图像
    delete_channel_by_cuda(&dst_gpu.at(0),&src_gpu.at(0),src_gpu.width(),src_gpu.height(),src_gpu.channels(),del_c,&src_cpu.at(0));

}