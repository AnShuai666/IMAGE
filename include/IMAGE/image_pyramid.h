/*
 * @desc    图像金字塔声明
 * @author  安帅
 * @date    2019-05-14
 * @e-mail   1028792866@qq.com
*/

#ifndef IMAGE_IMAGE_PYRAMID_H
#define IMAGE_IMAGE_PYRAMID_H

#include <mutex>
#include "define.h"
#include "image.hpp"

IMAGE_NAMESPACE_BEGIN
struct ImagePyramidLevel
{
    int width,height;
    image::ByteImage::ConstPtr image;

};

typedef std::vector<ImagePyramidLevel> ImagePyramidLevels;

class ImagePyramid : public ImagePyramidLevels
{
public:
    typedef std::shared_ptr<ImagePyramid> Ptr;
    typedef std::shared_ptr<ImagePyramid const> ConstPtr;
};

class ImagePyramidCache
{
public:
    static ImagePyramid::ConstPtr get();
    static void cleanup();

private:
    static std::mutex metadataMutex;
    //static image::Sc
    static std::string cachedEmbedding;
};

IMAGE_NAMESPACE_END
#endif //IMAGE_IMAGE_PYRAMID_H
