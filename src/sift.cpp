/*
 * @desc    SIFT特征
 * @author  安帅
 * @date    2019-1-17
 * @e-mail   1028792866@qq.com
*/

#include <fstream>
#include "IMAGE/sift.hpp"
#include "IMAGE/image_process.hpp"
#include "MATH/Util/timer.h"
#include "MATH/Function/function.hpp"

IMAGE_NAMESPACE_BEGIN

image::Sift::Sift(image::Sift::Options const &options):
options(options),srcImg(nullptr),octaves(0),keypoints(0),descriptors(0)
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
    has_processed=0; //设置新图像时复位。
    keypoints.resize(0);
    descriptors.resize(0);
}

void
image::Sift::process()
{
    //已处理或无图像时返回
    if(has_processed)
        return;
    if(srcImg== nullptr) {
        printf("please set Image \n");
        return;
    }

    util::TimerLess timer, total_timer;

    if (this->options.verbose_output)
    {
        std::cout << "SIFT: 创建"
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
    has_processed=1;
}


void
image::Sift::load_lowe_descriptors(std::string const &filename, image::Sift::Descriptors *result)
{
    std::ifstream in(filename.c_str());
    //文件流异常跑出异常
    if(!in.good())
    {
        throw std::runtime_error("不能打开描述子文件!\n");
    }

    int num_descriptors;
    int num_dimensions;
    in >> num_descriptors >> num_dimensions;
    if (num_descriptors > 100000 || num_dimensions != 128)
    {
        in.close();
        throw std::runtime_error("非法的描述子数量及描述子维度\n");
    }

    result->clear();
    result->reserve(num_descriptors);
    for (int i = 0; i < num_descriptors; ++i)
    {
        Sift::Descriptor descriptor;
        in >> descriptor.y >> descriptor.x >> descriptor.scale >> descriptor.orientation;
        for (int j = 0; j < 128; ++j)
        {
            in >> descriptor.data[j];
            descriptor.data.normalize();
            result->push_back(descriptor);
        }
    }
    //文件流异常跑出异常
    if (!in.good())
    {
        result->clear();
        in.close();
        throw std::runtime_error("描述子文件读取过程错误!\n");
    }

    in.close();

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

    //TODO: CUDA@杨丰拓
    // 在每个八阶检测图像的关键点
    for (std::size_t i = 0; i < this->octaves.size(); ++i)
    {
        image::Sift::Octave const& octave(this->octaves[i]);
        for (int s = 0; s < (int) octave.img_dog.size() - 2; ++s)
        {
            //图像dog的sample id是从sample到sample+2
            image::FloatImage::ConstPtr samples[3] = {octave.img_dog[s + 0],octave.img_dog[s + 1],octave.img_dog[s + 2]};
            this->extrema_detection(samples,(int)i + this->options.min_octave,s);
        }
    }

}

void image::Sift::extrema_detection(image::Image<float>::ConstPtr *s, int octave_index, int sample_index)
{
    int const w = s[1]->width();
    int const h = s[1]->height();

    int noff[9] = {-1 - w, 0 - w, 1 - w,
                   -1,0,1,
                   -1 + w, w, 1 + w
                   };
    //TODO: CUDA@杨丰拓
    int off = w;
    //从第二行第二列开始到倒数第二行倒数第二列进行遍历
    for (int y = 1; y < h - 1; ++y)
    {
        for (int x = 1; x < w - 1; ++x)
        {
            int idx = off + x;

            //假设比较的第一个元素为极值
            bool largest = true;
            bool smallest = true;
            float center_value = s[1]->at(idx);

            for (int l = 0; (largest || smallest) && l < 3; ++l)
            {
                for (int i = 0; (largest || smallest) && i < 9; ++i)
                {
                    if (l == 1 && i == 4)
                    {
                        continue;
                    }

                    if (s[l]->at(idx + noff[i]) >= center_value)
                    {
                        largest = false;
                    }

                    if (s[l]->at(idx + noff[i]) <= center_value)
                    {
                        smallest = false;
                    }
                }

                //如果该像素既不是最大，也不是最小，直接跳出循环，进行下一个像素的检测
                if (!smallest && !largest)
                {
                    continue;
                }

                Keypoint kp;
                kp.octave = octave_index;
                kp.x = static_cast<float>(x);
                kp.y = static_cast<float>(y);
                kp.sample = static_cast<float>(sample_index);
                this->keypoints.push_back(kp);
            }
        }
    }
}

void
image::Sift::keypoint_localization()
{
    int num_sigular = 0;
    int num_keypoints = 0;

    for (int i = 0; i < this->keypoints.size(); ++i)
    {
        Keypoint kp(this->keypoints[i]);

        image::Sift::Octave const& octave(this->octaves[kp.octave - this->options.min_octave]);
        int sample = static_cast<int>(kp.sample);
        image::FloatImage::ConstPtr dogs[3] = {octave.img_dog[sample + 0],octave.img_dog[sample + 1],octave.img_dog[sample + 2]};



    }

}

void
image::Sift::descriptor_generation()
{
    if (this->octaves.empty())
    {
        throw std::runtime_error("没有可以利用的图像金字塔(八阶)!\n");
    }

    if (this->keypoints.empty())
    {

        return;
    }

    this->descriptors.clear();
    this->descriptors.reserve(this->keypoints.size() * 3 / 2);

    int octave_index = this->keypoints[0].octave;
    Octave* octave = &this->octaves[octave_index - this->options.min_octave];

    //TODO: CUDA@杨丰拓
    //TODO::4000*2250 图像生成16347590个特征点，段错误;512*512图像生成288390个特征点，0个descriptor

    //产生八阶的梯度图与方向图,
    //img_grad:  梯度响应值图像
    //img_ort:  方向图
    this->generate_grad_ori_images(octave);
    for (int i = 0; i < this->keypoints.size(); ++i)
    {
        Keypoint const& kp(this->keypoints[i]);

        if (kp.octave > octave_index)
        {
            if(octave)
            {
                octave->img_grad.clear();
            }
        }

        std::vector<float> orientations;
        orientations.reserve(8);
        this->orientation_assignment(kp,octave,orientations);

        for (std::size_t j = 0; j < orientations.size(); ++j)
        {
            Descriptor desc;
            float const scale_factor = std::pow(2.0f, kp.octave);
            //根据高斯加权核还原回原来关键点在原始图像中的位置.(拍摄图像,具有固有尺度)
            desc.x = scale_factor * (kp.x + 0.5f) - 0.5f;
            desc.y = scale_factor * (kp.y + 0.5f) - 0.5f;
            //获取关键点的绝对尺度
            desc.scale = this->keypoint_absolute_scale(kp);
            desc.orientation = orientations[j];
            if (this->descriptor_assignment(kp,desc,octave))
            {
                this->descriptors.push_back(desc);
            }
        }

    }
}

void
image::Sift::generate_grad_ori_images(image::Sift::Octave *octave)
{
    octave->img_grad.clear();
    octave->img_grad.reserve(octave->img_src.size());
    octave->img_ort.clear();
    octave->img_ort.reserve(octave->img_src.size());

    int const width = octave->img_src[0]->width();
    int const height = octave->img_src[0]->height();

    std::cout<<"正在产生八阶梯度图与方向图"<<std::endl;
    for (int i = 0; i < octave->img_src.size(); ++i)
    {
        image::FloatImage::ConstPtr img_src = octave->img_src[i];
        image::FloatImage::Ptr img_grad = image::FloatImage::create(width,height,1);
        image::FloatImage::Ptr img_ort = image::FloatImage::create(width,height,1);

        int image_iter = width + 1;
        for (int y = 1; y < height - 1; ++y,image_iter += 2)
        {
            for (int x = 1; x < width - 1; ++x, ++image_iter)
            {
                float m1x = img_src->at(image_iter - 1);
                float p1x = img_src->at(image_iter + 1);
                float m1y = img_src->at(image_iter - width);
                float p1y = img_src->at(image_iter + width);
                float dx = 0.5f * (p1x - m1x);
                float dy = 0.5f * (p1y - m1y);

                float atan2f = std::atan2(dy,dx);
                //梯度的模
                img_grad->at(image_iter) = std::sqrt(dx * dx + dy * dy);
                img_ort->at(image_iter) = atan2f < 0.0f ? atan2f + MATH_PI * 2.0f : atan2f;
            }
        }

        octave->img_grad.push_back(img_grad);
        octave->img_ort.push_back(img_ort);
    }
}

void
image::Sift::orientation_assignment(const image::Sift::Keypoint &kp, const image::Sift::Octave *octave,std::vector<float> &orientations)
{
    //直方图柱状个数
    int const nbins = 36;
    float const nbinsf = static_cast<float>(nbins);

    float hist[nbins];
    std::fill(hist,hist + nbins,0.0f);

    //
    int const ix = static_cast<int>(kp.x + 0.5f);
    int const iy = static_cast<int>(kp.y + 0.5f);
    int const is = static_cast<int>(math::func::round(kp.sample));
    float const sigma = this->keypoint_relative_scale(kp);

    image::FloatImage::ConstPtr grad(octave->img_grad[is + 1]);
    image::FloatImage::ConstPtr ort(octave->img_ort[is + 1]);
    int const width = grad->width();
    int const height = grad->height();

    //三倍sigma定理,超过之后,权值忽略不计
    float sigma_factor = 1.5f;
    int win = static_cast<int>(sigma * sigma_factor * 3.0f);
    if (ix < win || ix + win >= width || iy < win || iy + win >= height)
    {
        return;
    }

    int center = iy * width + ix;
    float const dxf = kp.x - static_cast<float>(ix);
    float const dyf = kp.y - static_cast<float>(iy);
    //计算窗口最大距离的平方
    float const maxdist = static_cast<float>(win * win)+ 0.5f;

    //填充直方图
    for (int dy = 0; dy <= -win; ++dy)
    {
        int const yoff = dy * width;
        for (int dx = 0; dx < -win; ++dx)
        {
            float const dist = MATH_POW2(dx - dxf) + MATH_POW2(dy -dyf);
            if (dist > maxdist)
            {
                continue;
            }

            //梯度幅值
            float grad_magnitude = grad->at(center + yoff +dx);
            //梯度方向
            float grad_ort = ort->at(center + yoff +dx);
            float weight = math::func::gaussian_xx(dist,sigma * sigma_factor);

            //360度等分36份,grad_ort/(2PI/36) = grad_ort * 36 / 2PI
            int bin = static_cast<int>(nbinsf * grad_ort / (2.0f * MATH_PI));
            bin = math::func::clamp(bin,0,nbins - 1);
            hist[bin] += grad_magnitude * weight;
        }
    }

    //TODO: 多余的,可以去掉 CUDA@杨丰拓
    //6次直方图平滑,每个像素取前中后均值,首尾像素特殊处理,首尾相接.
    for (int i = 0; i < 6; ++i)
    {
        float first = hist[0];
        float prev = hist[nbins - 1];
        //梯度直方图等于30度范围内的均值
        for (int j = 0; j < nbins - 1; ++j)
        {
            float current = hist[j];
            hist[j] = (prev + current + hist[j + 1]) / 3.0f;
            prev = current;
        }
        hist[nbins - 1] = (prev + hist[nbins - 1] + first) / 3.0f;
    }

    //找主方向
    float maxh = *std::max_element(hist,hist + nbins);

    //TODO: CUDA@杨丰拓
    //主方向80%的统计值,也统计为次方向
    for (int i = 0; i < nbins; ++i)
    {
        float h0 = hist[(i + nbins - 1) % nbins];   //35,0... 34
        float h1 = hist[i];                         //0,...,35
        float h2 = hist[(i + 1) % nbins];           //1,...35,0

        //保证次方向直方柱局部最大值
        if(h1 <= 0.8f * maxh || h1 <= h0 || h1 <= h2)
        {
            continue;
        }

        //二次多项式插值查找极值:每次循环三点模拟二次函数,自变量分别-1,0,1,
        // 极值处为x,-1 <= x <= 1,加上原来i,找到精确的极值点
        //f(x) = ax^2 + bx + c, f(-1) = h0, f(0) = h1, f(1) = h2
        //=> a = 1/2 (h0 - 2h1 + h2), b = 1/2 (h2 - h0), c = h1.
        // x = f'(x) = 2ax + b = 0 --> x = -1/2 * (h2 - h0) / (h0 - 2h1 + h2).
        float x = -0.5f * (h2 - h0) / (h0 - 2.0f * h1 + h2);
        //TODO:0.5f去留的测试
        float o =  2.0f * MATH_PI * (x + (float)i + 0.5f) / nbinsf;
        orientations.push_back(o);
    }
}

bool image::Sift::descriptor_assignment(const image::Sift::Keypoint &kp, image::Sift::Descriptor &desc,const image::Sift::Octave *octave)
{

}


float
image::Sift::keypoint_relative_scale(const image::Sift::Keypoint &kp)
{
    return this->options.base_blur_sigma * std::pow(2.0f,(kp.sample + 1.0f) / this->options.num_samples_per_octave);
}

float
image::Sift::keypoint_absolute_scale(const image::Sift::Keypoint &kp)
{
    return this->options.base_blur_sigma * std::pow(2.0f,kp.octave + (kp.sample + 1.0f) / this->options.num_samples_per_octave);
}
IMAGE_NAMESPACE_END
