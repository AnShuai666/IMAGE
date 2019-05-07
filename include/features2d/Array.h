//
// Created by doing on 19-4-26.
//

#ifndef IMAGE_ARRAY_H
#define IMAGE_ARRAY_H
#include "image.hpp"
class Mat:public image::Image<float>
{
public:
    typedef std::shared_ptr<Mat> Ptr;
    typedef std::shared_ptr<Mat const> ConstPtr;
    typedef std::vector<float>  ImageData;
    typedef float ValueType;
public:
    Mat();
    Mat(int rows,int cols);
    Mat(const Mat& _in);
    virtual ~Mat();
    int rows() const;
    int cols() const;
    void release();
};

#endif //IMAGE_ARRAY_H
