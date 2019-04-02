/*
 * @desc    特征匹配
 * @author  安帅
 * @date    2019-04-02
 * @e-mail   1028792866@qq.com
*/

#ifndef IMAGE_MATCH_H
#define IMAGE_MATCH_H

#include <vector>
#include "define.h"
IMAGE_NAMESPACE_BEGIN

class Matching
{
public:
    struct Options
    {
       int descriptor_length;

       float lowe_ratio_threshold;

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

template <typename T>
void
Matching::oneway_match(const image::Matching::Options &options, T const *set_1, int set_1_size, T const *set_2,
                            int set_2_size, std::vector<int> *result)
{

}

template <typename T>
void Matching::twoway_match(const image::Matching::Options &options, T const *set_1, int set_1_size, T const *set_2,
                            int set_2_size, image::Matching::Result *matches)
{}
IMAGE_NAMESPACE_END
#endif //IMAGE_MATCH_H
