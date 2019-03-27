//
// Created by AnShuai on 18-12-18.
//
#include "image.hpp"
#include "Matrix/matrix.hpp"
#include "Matrix/vector.hpp"
#include <iostream>
using namespace matrix;
using namespace image;
using namespace std;

//HUB TEST


int main()
{
    cout<<"debug"<<endl;

    Matrix3d mat_33(3);
    TypedImageBase<float> img;
    img.resize(3,3,3);
    img.fill(3);
    TypedImageBase<float> img1(img),img2;
    img1.allocate(4,4,1);

    float *data = img.end();

    Image<float> img3(4,4,2);
    Image<float> img4(img3);
    float color[3] = {0,225,115};
    //img3.fill_color(color);//填充颜色数组报错
    //img3.add_channels(1,2);//添加通道数报错
    //img3.copy_channel(1,2);//复制通道报错
    float linear[2] = {1,1};
    img3.fill(2);
    int value = img3.reinterpret(2,8,2);


    cout<<value<<endl;
    return 0;
}