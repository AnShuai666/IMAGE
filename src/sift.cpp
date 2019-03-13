/*
 * @desc    SIFT特征
 * @author  安帅
 * @date    2019-1-17
 * @e-mail   1028792866@qq.com
*/

#include <include/timer.h>
#include "include/sift.hpp"
#include "include/image_process.hpp"
#include "include/timer.h"

IMAGE_NAMESPACE_BEGIN

image::Sift::Sift(image::Sift::Options const &options):
options(options)
{
    if (this->options.min_octave < -1 || this->options.min_octave > options.max_octave)
    {
        throw std::invalid_argument("非法图像阶范围！");
    }

    if (this->options.debug_output)
    {
        this->options.verbose_output = true;
    }
}

void
image::Sift::set_image(image::ByteImage::ConstPtr img)
{
    if (img->channels() != 1 && img->channels() != 3)
    {
        throw std::invalid_argument("需要灰度图或者彩色图");
    }

    this->srcImg = image::byte_to_float_image(img);

    if (img->channels() == 3)
    {
        this->srcImg = image::desaturate<float>(this->srcImg,image::DESATURATE_AVERAGE);
    }
}

void
image::Sift::process()
{
    image::TimerLess timer, total_timer;

    if (this->options.verbose_output)
    {
        std::cout << " SIFT: 创建"
                  << (this->options.max_octave - this->options.min_octave)
                  << "个八阶 (从"
                  << this->options.min_octave
                  << " 到 "
                  << this->options.max_octave
                  << ")..."
                  <<std::endl;
    }

    timer.reset();
    this->create_octaves();

    if (this->options.debug_output)
    {
        std::cout << "SIFT: 创建八阶用时 "
                  << timer.get_elapsed()
                  << "毫秒。"
                  <<std::endl;
    }

    timer.reset();
    this->extrema_detection();

    if (this->options.debug_output)
    {
        std::cout << "SIFT: 检测到 "
                  << this->keypoints.size()
                  << " 个关键点, 用时 "
                  << timer.get_elapsed()
                  << "毫秒。"
                  << std::endl;
    }


    if (this->options.debug_output)
    {
        std::cout << "SIFT: 正在定位与过滤关键点中......"<<std::endl;
    }

    timer.reset();
    this->keypoint_localization();

    if (this->options.debug_output)
    {
        std::cout << "SIFT: 保留了 "
                  << this->keypoints.size()
                  << " 个稳定关键点， 滤波用时 "
                  << timer.get_elapsed()
                  << "毫秒。"
                  << std::endl;
    }

    //清楚高斯查分图像
    for (auto& octave : this->octaves)
    {
        octave.img_dog.clear();
    }


    if (this->options.verbose_output)
    {
        std::cout << "SIFT: 正在生成描述子列表......"<<std::endl;
    }

    timer.reset();
    this->descriptor_generation();

    if (this->options.debug_output)
    {
        std::cout << "SIFT: 生成了 "
                  << this->descriptors.size()
                  << " 个描述子, 用时 "
                  << timer.get_elapsed()
                  << " 毫秒。"
                  << std::endl;
    }

    if (this->options.verbose_output)
    {
        std::cout << "SIFT: 从 "
                  << this->keypoints.size()
                  << " 个关键点, 共生成了 "
                  << this->descriptors.size()
                  << " 个描述子。 用时 "
                  << total_timer.get_elapsed()
                  << " 毫秒。"
                  << std::endl;
    }

    //清空八阶以清除内存
    this->octaves.clear();
}


void
image::Sift::set_float_image(image::FloatImage::ConstPtr img)
{
    if (img->channels() != 1 && img->channels() != 3)
    {
        throw std::invalid_argument("需要灰度图或者彩色图");
    }

    if(img->channels() == 3)
    {
        this->srcImg = image::desaturate<float>(img,image::DESATURATE_AVERAGE);
    }
    else
    {
        this->srcImg = img->duplicate();
    }
}



void
image::Sift::create_octaves(void)
{
    this->octaves.clear();

    // 若min_octave < 0, 则创建－１八阶，
    // 原图假设模糊尺度为0.5,则上采样后的图像模糊尺度为2*0.5即可达到原图模糊效果． 模糊半径为像素单位
    if (this->options.min_octave < 0)
    {
        image::FloatImage::Ptr img = image::rescale_double_size_supersample<float >(this->srcImg);
        this->add_octave(img,this->options.inherent_blur_sigma * 2.0f,this->options.base_blur_sigma);
    }

    // 若min_octave > 0, 则创建正八阶，
    image::FloatImage::ConstPtr img = this->srcImg;
    for (int i = 0; i < this->options.min_octave; ++i)
    {
        img = image::rescale_half_size_gaussian<float>(img);
    }

    float img_sigma = this->options.inherent_blur_sigma;
    for (int i = std::max(0,this->options.min_octave); i <= this->options.max_octave; ++i)
    {
        //鄙人认为在进行高斯模糊降采样的时候就已经进行了尺度变换，不需要再进行尺度变换了，且高斯模糊尺度是根据两个八阶
        //第一幅图像的尺度来进行计算的。若不明白，联系我即可。
        this->add_octave(img,img_sigma,std::pow(2,i) * this->options.base_blur_sigma);
        image::FloatImage::ConstPtr pre_base = octaves[octaves.size()-1].img_src[0];
        img = image::rescale_half_size_gaussian<float >(pre_base,std::pow(2.0f,i + 1) * MATH_POW2(this->options.base_blur_sigma));

        img_sigma = std::pow(2.0f,i+1) * this->options.base_blur_sigma;
    }
}

void
image::Sift::add_octave(image::FloatImage::ConstPtr image,float has_sigma, float target_sigma)
{
    //图像固有尺度为has_sigma,要模糊到尺度为target_sigma,连续模糊有如下性质:
    //L * G(sigma1) * G(sigma2) = L * G (sart(sigma1^2 + sigma^2))
    //即 L * G(has_sigma) * G(sigma) = L * G(target_sigma),
    //则has_sigma^2 + sigma^2 = target_sigma^2
    //得sigma = sart(target_sigma^2 - has_sigma^2)
    //现有图像为 L1 = L * G(has_sigma)
    float sigma = std::sqrt(MATH_POW2(target_sigma) - MATH_POW2(has_sigma));
    image::FloatImage::Ptr base = (target_sigma > has_sigma
            ? image::blur_gaussian<float>(image,sigma)
            : image->duplicate());
    this->octaves.push_back(Octave());
    Octave& octave = this->octaves.back();
    octave.img_src.push_back(base); //高斯空间的图像,第一个图像的模糊尺度为1.6

    //TODO: 建立表格，减少计算，加快速度
    //k是同一个八阶内,相邻阶的尺度比例,分别,sigama,k*sigma,k*k*sigma,...k^(s+2)*sigma
    //s 有效差分数默认 s = 3, k = 2 ^ (1/3) = 1.25992105 k*sigma,....k^5*sigma
    // sigma = 1.6,
    // k * sigma = 2.016,
    // k * k * sigma = 2.539841683,
    // k * k * k * sigma = 3.20000000,
    // k * k * k * k * sigma = 4.031747361,
    // k * k * k * k * k * sigma = 5.079683368
    float const k = std::pow(2.0f,1.0f / this->options.num_samples_per_octave);
    sigma = target_sigma;

    for (int i = 1; i < this->options.num_samples_per_octave + 3; ++i)
    {
        float sigmak = k * sigma;
        float blur_sigma = std::sqrt(MATH_POW2(sigmak) - MATH_POW2(sigma));

        image::FloatImage::Ptr img = image::blur_gaussian<float>(base,blur_sigma);
        octave.img_src.push_back(img);

        // 创建高斯差分图像(DOG Difference of Gaussian)
        image::FloatImage::Ptr dog = image::subtract<float>(img,base);
        octave.img_dog.push_back(dog);

        base = img;
        sigma = sigmak;
    }
}

void
image::Sift::add_octave2(image::FloatImage::ConstPtr image,float has_sigma2, float target_sigma2)
{
    //图像固有尺度为has_sigma,要模糊到尺度为target_sigma,连续模糊有如下性质:
    //L * G(sigma1) * G(sigma2) = L * G (sart(sigma1^2 + sigma^2))
    //即 L * G(has_sigma) * G(sigma) = L * G(target_sigma),
    //则has_sigma^2 + sigma^2 = target_sigma^2
    //得sigma = sart(target_sigma^2 - has_sigma^2)
    //现有图像为 L1 = L * G(has_sigma)
    float sigma2 = target_sigma2 - has_sigma2;
    image::FloatImage::Ptr base = (target_sigma2 > has_sigma2
                                   ? image::blur_gaussian2<float>(image,sigma2)
                                   : image->duplicate());
    this->octaves.push_back(Octave());
    Octave& octave = this->octaves.back();
    octave.img_src.push_back(base); //高斯空间的图像,第一个图像的模糊尺度为1.6

    //k是同一个八阶内,相邻阶的尺度比例,分别,sigama,k*sigma,k*k*sigma,...k^(s+2)*sigma
    //s 有效差分数默认 s = 3, k = 2 ^ (1/3) = 1.25992105 k*sigma,....k^5*sigma
    // sigma = 1.6,
    // k * sigma = 2.016,
    // k * k * sigma = 2.539841683,
    // k * k * k * sigma = 3.20000000,
    // k * k * k * k * sigma = 4.031747361,
    // k * k * k * k * k * sigma = 5.079683368
    float const k = std::pow(2.0f,1.0f / this->options.num_samples_per_octave);
    float const k2 = MATH_POW2(k);
    sigma2 = target_sigma2;

    for (int i = 1; i < this->options.num_samples_per_octave + 3; ++i)
    {
        float sigma2k2 = k2 * sigma2;
        float blur_sigma2 = sigma2k2 - sigma2;

        image::FloatImage::Ptr img = image::blur_gaussian2<float>(base,blur_sigma2);
        octave.img_src.push_back(img);

        // 创建高斯差分图像(DOG Difference of Gaussian)
        image::FloatImage::Ptr dog = image::subtract<float>(img,base);
        octave.img_dog.push_back(dog);

        base = img;
        sigma2 = sigma2k2;
    }
}

void
image::Sift::extrema_detection()
{
    this->keypoints.clear();
    
    // 在每个八阶检测图像的关键点
    for (auto& octave : this->octaves)
    {



    }
}

std::size_t image::Sift::extrama_detection(image::Image<float>::ConstPtr *s, int oi, int si)
{
    int const w = s[1]->width();
    int const h = s[1]->height();

    int noff[9] = {};

    int detected = 0;
    int off = w;
    for (int y = 0; y < h - 1; ++y)
    {
        for (int x = 0; x < w - 1; ++x)
        {
            int idx = off + x;

            bool largest = true;
            bool smallest = true;
            float center_value = s[1]->at(idx);

        }
    }

    int
}
IMAGE_NAMESPACE_END
