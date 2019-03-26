/*
 * @desc    SIFT特征
 * @author  安帅
 * @date    2019-1-17
 * @e-mail   1028792866@qq.com
*/


#ifndef IMAGE_SIFT_H
#define IMAGE_SIFT_H

#include <string>
#include <vector>
#include "define.h"
#include "Matrix/vector.hpp"
#include "image.hpp"

IMAGE_NAMESPACE_BEGIN

/********************************************************************
*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Sift类的声明~~~~~~~~~~~~~~~~~~~~~~~~~~~~
********************************************************************/
/*
 * @property    图像Sift类
 * @func        Sift关键点、描述子特征提取等
 * @param
 * @param
 * @param
*/
    class Sift {
/********************************************************************
*~~~~~~~~~~~~~~~~~~~~~~~~~常用结构体声明及定义~~~~~~~~~~~~~~~~~~~~~~~~~~
********************************************************************/
    public:
        /*
        * @property    图像Sift类
        * @func        Sift关键点、描述子特征提取等
        * @param
        * @param
        * @param
        */
        struct Options {
            //void 避免函数进行检测参数，加速
            Options(void);

            //每阶有效差分个数默认为3
            int num_samples_per_octave;

            //设置最小阶的ID.
            // 默认为0，即原始图像；
            // -1，原图上采样2倍；
            // >0，原图下采样min_octave*2 倍
            int min_octave;

            //设置最大阶ID，默认为4，也就是一共5阶分别是0,1,2,3,4,,下采样4次
            int max_octave;

            //消除边界响应，trace(H)^2/Det(H) < [(r+1)^2]/r 默认为10
            float edge_ratio_threshold;

            //图像固有尺度，假设图像拍摄就有默认尺度为0.5的模糊。即实际原图为I,拍摄图像为L = I * G(inherent_blur_sigma)
            float inherent_blur_sigma;

            //基本图像的尺度，默认1.6 第0阶第一张图的基本尺度,可通过固有尺度模糊到该尺度
            float base_blur_sigma;

            //是否输出程序运行的状态信息到控制台
            bool verbose_output;

            //输出更多的状态信息到控制台
            bool debug_output;

        };

        struct Keypoint {
            //关键点的阶索引，第octave阶
            int octave;

            //有效高斯差分索引，{0,S-1} 高斯差分{-1,S}
            float sample;

            //关键点x坐标
            float x;

            //关键点y坐标
            float y;
        };

        struct Descriptor {
            //关键点亚像素x坐标
            float x;

            //关键点亚像素y的坐标
            float y;

            //关Options键点的尺度
            float scale;

            //关键点的方向：[0,2PI]
            float orientation;

            //描述子的数据，[0.0,1.0]
            matrix::Vector<float, 128> data;
        };
/********************************************************************
*~~~~~~~~~~~~~~~~~~~~~~~~~常用数据类型别名定义~~~~~~~~~~~~~~~~~~~~~~~~~~
********************************************************************/
    public:
        typedef std::vector<image::Sift::Keypoint> Keypoints;
        typedef std::vector<image::Sift::Descriptor> Descriptors;
        //typedef std::vector<image::FloatImage::Ptr> ImageVector;

    public:
        /*
        *  @property   Sift构造函数
        *  @func       为Sift特征分配内存空间，初始化Sift类变量
        *  @param_in   options          Sift特征的初始化数据选项
        *  @explict    显示转换，在构造函数中使用，防止非类对象隐式转换修改成员变量
        */
        explicit Sift(image::Sift::Options const &options);

        /*
        *  @property    图像转换为float图
        *  @func        设置输入图像　灰度图不变　RGB->HSL
        *  @param_in    img       输入图
        *  @return      void
        */
        void set_image(image::ByteImage::ConstPtr img);

        /*
        *  @property    图像转换
        *  @func        设置输入图像　灰度图不变　RGB->HSL
        *  @param_in    img       输入图
        *  @return      void
        */
        void set_float_image(image::FloatImage::ConstPtr img);

        /*
        *  @property    图像处理
        *  @func        检测图像的SIFT关键点与描述子，并对其进行滤波处理
        *               最后清除中间过程生成的八阶。
        *  @return      void
        */
        void process(void);

        /*
        *  @property    关键点获取
        *  @func        获取图像的SIFT关键点
        *  @return      Keypoints
        */
        Keypoints const& get_keypoints(void) const;

        /*
        *  @property    描述子获取
        *  @func        获取图像的SIFT关键点的描述子
        *  @return      Descriptors
        */
        Descriptors const& get_descriptors(void) const;

        /*
        *  @property    描述子获取
        *  @func        获取图像的SIFT关键点的描述子
        *  @param_in    filename    描述子文件名
        *  @param_In    result      描述子容器   一幅图像的描述子结果存在此中
        *  @return      void
        */
        static void load_lowe_descriptors(std::string const& filename, Descriptors* result);
/********************************************************************
*~~~~~~~~~~~~~~~~~~~~~~~~~常用结构体声明及定义~~~~~~~~~~~~~~~~~~~~~~~~~~
********************************************************************/
    protected:
        struct Octave {
            typedef std::vector<image::FloatImage::Ptr> ImageVector;

            ImageVector img_src;    //每阶原图像数        s+3
            ImageVector img_dog;    //每阶高斯差分图像数   s+2
            ImageVector img_grad;   //每阶梯度图像数      s+3
            ImageVector img_ort;    //每阶旋转图像        s+3
        };

    protected:
        typedef std::vector<Octave> Octaves;

    protected:
        /*
        *  @property    图像转换
        *  @func        设置输入图像　灰度图不变　RGB->HSL
        *  @param_in    img       输入图
        *  @return      void
        */
        void create_octaves(void);

        /*
        *  @property    图像增添八阶
        *  @func        图像建立八阶，即高斯空间，高斯差分空间，
        *               图像固有尺度为has_sigma,默认为0.5,八阶中高斯空间第一个高斯空间图像的尺度为 target_sigma,
        *  @param_in    image           输入图
        *  @param_in    has_sigma       固有尺度
        *  @param_in    target_sigma    目标尺度
        *  @return      void
        */
        void add_octave(image::FloatImage::ConstPtr image, float has_sigma, float target_sigma);

        /*
        *  @property    图像增添八阶
        *  @func        图像建立八阶，即高斯空间，高斯差分空间，直接用平方进行计算，不用进行开方计算
        *               图像固有尺度为has_sigma,默认为0.5,八阶中高斯空间第一个高斯空间图像的尺度为 target_sigma,
        *  @param_in    image            输入图
        *  @param_in    has_sigma2       固有尺度平方
        *  @param_in    target_sigma2    目标尺度平方
        *  @return      void
        */
        void add_octave2(image::FloatImage::ConstPtr image, float has_sigma2, float target_sigma2);

        void extrema_detection(void);

        void extrema_detection(image::FloatImage::ConstPtr s[3],int octave_index, int sample_index);

        void keypoint_localization(void);

        void descriptor_generation(void);

        void generate_grad_ori_images(Octave* octave);

        void orientation_assignment(Keypoint const& kp, Octave const* octave, std::vector<float>& orientations);


        bool descriptor_assignment(Keypoint const& kp, Descriptor& desc,Octave const* octave);

        float keypoint_relative_scale(Keypoint const& kp);

        /*
        *  @property    图像关键点绝对尺度
        *  @func        获取图像关键点绝对尺度,第octave阶,第sample个dog图像的尺度,其中 S为sample总数,
        *               s为关键点的sample位置下一层,sample[3]分别为s:s+1:s+2,因此尺度为2^octave * k^(s+1) * sigma
        *               其中,k = 2^(1/S)
        *  @param_in    kp               图像关键点引用
        *  @return      float   关键点的绝对尺度,也就是相对于无模糊图像的尺度.
        */
        float keypoint_absolute_scale(Keypoint const& kp);


    private:
        Options options;
        image::FloatImage::ConstPtr srcImg;
        Octaves octaves;
        Keypoints keypoints;
        Descriptors descriptors;

    };


    inline
    Sift::Options::Options(void) :
            num_samples_per_octave(3),
            min_octave(0),
            max_octave(4),
            edge_ratio_threshold(10.0f),
            inherent_blur_sigma(0.5f),
            base_blur_sigma(1.6f),
            verbose_output(false),
            debug_output(false) {

    }


IMAGE_NAMESPACE_END


#endif //IMAGE_SIFT_H
