/*
 * @desc    可视化
 * @author  安帅
 * @date    2019-04-01
 * @email   1028792866@qq.com
*/

#include "include/visualizer.h"
#include "include/image_process.hpp"
IMAGE_NAMESPACE_BEGIN

namespace
{
    unsigned char color_table[12][3] =
            {
                    { 255, 0,   0 }, { 0, 255,   0 }, { 0, 0, 255   },
                    { 255, 255, 0 }, { 255, 0, 255 }, { 0, 255, 255 },
                    { 127, 255, 0 }, { 255, 127, 0 }, { 127, 0, 255 },
                    { 255, 0, 127 }, { 0, 127, 255 }, { 0, 255, 127 }
            };

    void
    draw_box (ByteImage& image, float x, float y,
              float size, float orientation, uint8_t const* color)
    {
        float const sin_ori = std::sin(orientation);
        float const cos_ori = std::cos(orientation);

        float const x0 = (cos_ori * -size - sin_ori * -size);
        float const y0 = (sin_ori * -size + cos_ori * -size);
        float const x1 = (cos_ori * +size - sin_ori * -size);
        float const y1 = (sin_ori * +size + cos_ori * -size);
        float const x2 = (cos_ori * +size - sin_ori * +size);
        float const y2 = (sin_ori * +size + cos_ori * +size);
        float const x3 = (cos_ori * -size - sin_ori * +size);
        float const y3 = (sin_ori * -size + cos_ori * +size);

        image::draw_line(image, static_cast<int>(x + x0 + 0.5f),
                               static_cast<int>(y + y0 + 0.5f), static_cast<int>(x + x1 + 0.5f),
                               static_cast<int>(y + y1 + 0.5f), color);
        image::draw_line(image, static_cast<int>(x + x1 + 0.5f),
                               static_cast<int>(y + y1 + 0.5f), static_cast<int>(x + x2 + 0.5f),
                               static_cast<int>(y + y2 + 0.5f), color);
        image::draw_line(image, static_cast<int>(x + x2 + 0.5f),
                               static_cast<int>(y + y2 + 0.5f), static_cast<int>(x + x3 + 0.5f),
                               static_cast<int>(y + y3 + 0.5f), color);
        image::draw_line(image, static_cast<int>(x + x3 + 0.5f),
                               static_cast<int>(y + y3 + 0.5f), static_cast<int>(x + x0 + 0.5f),
                               static_cast<int>(y + y0 + 0.5f), color);
    }

}  // namespace

/* ---------------------------------------------------------------- */

void Visualizer<uint8_t >::draw_keypoint<>(image::ByteImage &image, const Visualizer::Keypoint &keypoint,
                               Visualizer<uint8_t>::KeypointStyle style, uint8_t const *color) {
    int const x = static_cast<int>(keypoint.x + 0.5);
    int const y = static_cast<int>(keypoint.y + 0.5);
    int const width = image.width();
    int const height = image.height();
    int const channels = image.channels();

    if (x < 0 || x >= width || y < 0 || y >= height)
        return;

    int required_space = 0;
    bool draw_orientation = false;
    switch (style)
    {
        default:
        case SMALL_DOT_STATIC:
            required_space = 0;
            draw_orientation = false;
            break;
        case SMALL_CIRCLE_STATIC:
            required_space = 3;
            draw_orientation = false;
            break;
        case RADIUS_BOX_ORIENTATION:
            required_space = static_cast<int>(std::sqrt
                    (2.0f * keypoint.radius * keypoint.radius)) + 1;
            draw_orientation = true;
            break;
        case RADIUS_CIRCLE_ORIENTATION:
            required_space = static_cast<int>(keypoint.radius);
            draw_orientation = true;
            break;
    }

    if (x < required_space || x >= width - required_space
        || y < required_space || y >= height - required_space)
    {
        style = SMALL_DOT_STATIC;
        required_space = 0;
        draw_orientation = false;
    }

    switch (style)
    {
        default:
        case SMALL_DOT_STATIC:
            std::copy(color, color + channels, &image.at(x, y, 0));
            break;
        case SMALL_CIRCLE_STATIC:
            image::draw_circle(image, x, y, 3, color);
            break;
        case RADIUS_BOX_ORIENTATION:
            draw_box(image, keypoint.x, keypoint.y,
                     keypoint.radius, keypoint.orientation, color);
            break;
        case RADIUS_CIRCLE_ORIENTATION:
            image::draw_circle(image, x, y, required_space, color);
            break;
    }

    if (draw_orientation)
    {
        float const sin_ori = std::sin(keypoint.orientation);
        float const cos_ori = std::cos(keypoint.orientation);
        float const x1 = (cos_ori * keypoint.radius);
        float const y1 = (sin_ori * keypoint.radius);
        image::draw_line(image, static_cast<int>(keypoint.x + 0.5f),
                               static_cast<int>(keypoint.y + 0.5f),
                               static_cast<int>(keypoint.x + x1 + 0.5f),
                               static_cast<int>(keypoint.y + y1 + 0.5f), color);
    }
}

/* ---------------------------------------------------------------- */

ByteImage::Ptr
Visualizer<>::draw_keypoints(ByteImage::ConstPtr image,
                           std::vector<Visualizer<T>::Keypoint> const& matches,
                           Visualizer::KeypointStyle style)
{
    ByteImage::Ptr ret;
    if (image->channels() == 3)
    {
        ret = image::desaturate<unsigned char>(image, core::image::DESATURATE_AVERAGE);
        ret = image::expand_grayscale<unsigned char>(ret);
    }
    else if (image->channels() == 1)
    {
        ret = image::expand_grayscale<unsigned char>(image);
    }

    uint8_t* color = color_table[3];
    for (std::size_t i = 0; i < matches.size(); ++i)
    {
        Visualizer::draw_keypoint(*ret, matches[i], style, color);
    }

    return ret;
}


IMAGE_NAMESPACE_END