/*
 * @desc    可视化
 * @author  安帅
 * @date    2019-04-01
 * @email   1028792866@qq.com
*/

#ifndef IMAGE_VISUALLIZER_H
#define IMAGE_VISUALLIZER_H

#include "define.h"
IMAGE_NAMESPACE_BEGIN
class Visualizer
{
public:
    struct Keypoint
    {
        float x;
        float y;
        float radius;
        float orientation;
    };

    enum KeypointStytle
    {
        RADIUS_BOX_ORIENTATION,
        RADIUS_CIRCLE_ORIENTATION,
        SMALL_CIRCLE_STATIC,
        SMALL_DOT_STATIC
    };

public:
    static void draw_keypoint(image::ByteImage& image, Keypoint const& keypoint,KeypointStytle style,uint8_t const* color);




};

IMAGE_NAMESPACE_END


#endif //IMAGE_VISUALLIZER_H
