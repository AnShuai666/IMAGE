/*
 * @desc    特征匹配
 * @author  安帅
 * @date    2019-04-02
 * @e-mail   1028792866@qq.com
*/

#ifndef IMAGE_MATCH_H
#define IMAGE_MATCH_H

#include <vector>
#include <limits>
#include "define.h"
IMAGE_NAMESPACE_BEGIN


class Matching
{
public:
    struct Options
    {
        //描述子长度
       int descriptor_length;

       //要求最佳匹配距离和次最佳匹配距离之间的比率低于某个阈值。
       // 如果此比率接近1，则匹配不明确。
       // 好的值是sift:0.8,surf:0.7
       // 设置为1.0以禁用测试。
       float lowe_ratio_threshold;

       //不接受距离大于这个阈值，设置为FLOAT_MAX 禁用测试
       float distance_threshold;
    };

    struct Result
    {
        std::vector<int> matches_1_2;
        std::vector<int> matches_2_1;
    };

public:

    template <typename T>
    static void
    oneway_match(Options const& options,T const* set_1,int set_1_size,T const* set_2,int set_2_size,std::vector<int>* result);

    template <typename T>
    static void
    twoway_match(Options const& options,T const* set_1,int set_1_size,T const* set_2,int set_2_size,Result* matches);

    static void
    remove_inconsistent_matches(Result* matches);

    static int
    count_consistent_matches(Result const& matches);

    static void
    combine_results(Result const& sift_result, Result const& surf_result,Matching::Result* result);
};

class MatchingBase
{
public:
    struct Options
    {
        Matching::Options sift_matching_options
        {
            128,0.8f,std::numeric_limits<float>::max()
        };
        Matching::Options surf_mathing_options
        {
            64, 0.7f, std::numeric_limits<float>::max()
        };
    };
    //用于显式要求编译器提供合成版本的四大函数(构造、拷贝、析构、赋值)
    virtual ~MatchingBase(void) = default;

    virtual void init();

    // 匹配所有特征类型，生成一个匹配结果。
    virtual void pairwise_match(int view_1_id,int view_2_id,
            Matching::Result* result) const = 0;

    //匹配n个最低分辨率特征点并返回匹配的数目。可以用作完全匹配的估计
    //500特征至多3个匹配 300个特征2个匹配 为有用值。
    virtual int pairwise_match_low_res(int view_1_id,int view_2_id,
            std::size_t num_features) const = 0;

    Options options;
};

class ExhausitiveMatching : public MatchingBase
{

};

class CascadeHashingMatching : public ExhausitiveMatching
{

};


template <typename T>
void
Matching::oneway_match(const image::Matching::Options &options, T const *set_1, int set_1_size, T const *set_2,
                            int set_2_size, std::vector<int> *result)
{

}

template <typename T>
void Matching::twoway_match(const image::Matching::Options &options, T const *set_1, int set_1_size, T const *set_2,
                            int set_2_size, image::Matching::Result *matches)
{

}
IMAGE_NAMESPACE_END
#endif //IMAGE_MATCH_H
