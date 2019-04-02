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

};

IMAGE_NAMESPACE_END
#endif //IMAGE_MATCH_H
