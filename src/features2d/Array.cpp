//
// Created by doing on 19-4-26.
//

#include "features2d/Array.h"

Mat::Mat():Image<float>(0,0,0) {
}
Mat::Mat(int rows, int cols) : Image<float>(rows,cols,1){
}
Mat::Mat(const Mat& _in):Image<float>(_in){
}
Mat::~Mat(){

}
int Mat::cols() const {
    return w;
}
int Mat::rows() const {
    return h;
}

void Mat::release(){
    w=0;
    h=0;
    data.resize(0);
}
