//
// Created by AnShuai on 18-12-18.
//
#include "IMAGE/image.hpp"
#include "include/MATH/Matrix/matrix.hpp"
#include <iostream>
using namespace math::matrix;
using namespace image;
using namespace std;

//HUB TEST

class base{
public:
    base(int row,int col)
    {
        date=row+col;
    }

    int  date;
};
class son:public base
{
public:
    son():base(0,0)
    {row=0;
    col=0;};
    son(int _row,int _col):base(_row,_col)
    {
        row=_row;
        col=_col;
    }

    son&operator=(son& src)
    {

        row=src.row;
        col=src.col;
    }
    int row;
    int col;
};



int main()
{

    Image<float> s(10,10);
    for(int i=0;i<100;i++)
        s.at(i)=i;
    Image<float> s2;
    s2=s;
    for(int i=0;i<100;i++)
        cout<<s2.at(i)<<endl;
    Matrix3d mat_33(3);
    TypedImageBase<float> img;
    img.resize(3,3,3);
    img.fill(3);
    TypedImageBase<float> test;
    test=TypedImageBase<float>();
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
    cout<<img3.ptr(0)[0]<<endl;
    return 0;
}