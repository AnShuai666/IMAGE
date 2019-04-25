
#include "image_io.h"
#include "cuda_include/process_cuda.h"
#include "sift.hpp"
#include "timer.h"
#include "image_process.hpp"

#include <iostream>
using namespace std;
int main(int argc, char ** argv)
{
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

    image::byte_to_float_image(image);
    image::FloatImage::ConstPtr srcImage1,srcImage2,srcImage3,srcImage4,srcImage5;
    cout<<"*************调试开始*************"<<endl;
    //image::rescale_half_size_gaussian_cu<float>(image::byte_to_float_image(image));
    srcImage1=image::blur_gaussian_cu<float>(image::byte_to_float_image(image), 0.75f);
    srcImage2=image::blur_gaussian2_cu<float>(image::byte_to_float_image(image), 0.75f);

    srcImage3=image::blur_gaussian<float>(image::byte_to_float_image(image), 0.75f);
    srcImage4=image::blur_gaussian2<float>(image::byte_to_float_image(image), 0.75f);

    image::subtract_cu<float>(srcImage1, srcImage5);
    cout<<"*************调试结束*************"<<endl;


    return  0;
}
