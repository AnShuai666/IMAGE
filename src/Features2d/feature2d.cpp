/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/


#include "Features2d/features2d.h"
#include <vector>
using namespace std;
namespace features2d
{
float  atan(float y,float x){
    const float PI=3.1415926535897932384626433832795;
    //反正切的角度等于X轴与通过原点和给定坐标点(x, y)的直线之间的夹角。
    // 结果为正表示从X轴逆时针旋转的角度，结果为负表示从X轴顺时针旋转的角度
    float angle_pi=std::atan2(y,x);
    float angle=angle_pi/PI*180.f;
    if(angle<0)
        angle=360+angle;
    return angle;
}
int  round(double x){
    if(x<0)
        x-=0.5;
    if(x>0)
        x+=0.5;
    return (int)x;
}
int  floor(double x){
    if((x-round(x))<EPSILON)
        return round(x);
    if(x<0)
        x-=1;
    return (int)x;
}
int  ceil(double x){
    if((x-round(x))<EPSILON)
        return round(x);
    if(x>0)
        x+=1;
    return (int)x;
}
KeyPoint::KeyPoint()
        : m_pt(0,0), m_size(0), m_angle(-1), m_response(0), m_octave(0), m_class_id(-1) {}


KeyPoint::KeyPoint(Point2f _pt, float _size, float _angle, float _response, int _octave, int _class_id)
        : m_pt(_pt), m_size(_size), m_angle(_angle), m_response(_response), m_octave(_octave), m_class_id(_class_id) {}


KeyPoint::KeyPoint(float x, float y, float _size, float _angle, float _response, int _octave, int _class_id)
        : m_pt(x, y), m_size(_size), m_angle(_angle), m_response(_response), m_octave(_octave), m_class_id(_class_id) {}



Feature2D::~Feature2D() {}

/*
 * Detect keypoints in an image.
 * image        The image.
 * keypoints    The detected keypoints.
 * mask         Mask specifying where to look for keypoints (optional). Must be a char
 *              matrix with non-zero values in the region of interest.
 */
void Feature2D::detect( UCInputArray& image,
                        std::vector<KeyPoint>& keypoints,
                        UCInputArray& mask )
{
    if( image.empty() )
    {
        keypoints.clear();
        return;
    }
    if(!mask.empty())
    {
        if(mask.rows()!=image.rows()
        ||mask.cols()!=image.cols()
        ||mask.channels()!=1){
            throw("Mask size error\n");
        }
    }
    Mat des;
    detectOrCompute(image, mask, keypoints, des, false);
}
/*
 * Compute the descriptors for a set of keypoints in an image.
 * image        The image.
 * keypoints    The input keypoints. Keypoints for which a descriptor cannot be computed are removed.
 * descriptors  Copmputed descriptors. Row i is the descriptor for keypoint i.
 */
void Feature2D::compute( UCInputArray& image,
                         std::vector<KeyPoint>& keypoints,
                         UCOutputArray& descriptors ){
    if( image.empty()||keypoints.empty())
    {
        return;
    }
    if(descriptors.empty())
    {
        int dsize = descriptorSize();
        descriptors.resize(dsize, (int)keypoints.size(), 1);
    }
    UCMat mask;
    detectOrCompute(image, mask, keypoints, descriptors, true);
}
void Feature2D::compute( UCInputArray& image,
                         std::vector<KeyPoint>& keypoints,
                         OutputArray& descriptors ){
    if( image.empty()||keypoints.empty())
    {
        return;
    }
    if(descriptors.empty())
    {
        int dsize = descriptorSize();
        descriptors.resize(dsize, (int)keypoints.size(), 1);
    }
    UCMat mask;
    detectOrCompute(image, mask, keypoints, descriptors, true);
}
void Feature2D::detectAndCompute( UCInputArray& image, UCInputArray& mask,
                                std::vector<KeyPoint>& keypoints,
                                OutputArray& descriptors){
    if( image.empty() )
    {
        return;
    }
    if(descriptors.empty())//descriptors.empty() 作为标志位判断是否计算描述子
    {
        int dsize = descriptorSize();
        descriptors.resize(dsize, 1, 1);
    }
    if(!mask.empty())
    {
        if(mask.rows()!=image.rows()
           ||mask.cols()!=image.cols()
           ||mask.channels()!=1){
            throw("Mask size error\n");
        }
    }
    detectOrCompute(image, mask, keypoints, descriptors, false);
}
void Feature2D::detectAndCompute( UCInputArray& image, UCInputArray& mask,
                                  std::vector<KeyPoint>& keypoints,
                                  UCOutputArray& descriptors){
    if( image.empty() )
    {
        return;
    }
    if(descriptors.empty())//descriptors.empty() 作为标志位判断是否计算描述子
    {
        int dsize = descriptorSize();
        descriptors.resize(dsize, 1, 1);
    }
    if(!mask.empty())
    {
        if(mask.rows()!=image.rows()
           ||mask.cols()!=image.cols()
           ||mask.channels()!=1){
            throw("Mask size error\n");
        }
    }
    detectOrCompute(image, mask, keypoints, descriptors, false);
}
/* Detects keypoints and computes the descriptors */
void Feature2D::detectOrCompute( UCInputArray&, UCInputArray&,
                                  std::vector<KeyPoint>&,
                                  OutputArray&,
                                  bool){

    throw("Error::StsNotImplemented");
}
void Feature2D::detectOrCompute( UCInputArray&, UCInputArray&,
                                 std::vector<KeyPoint>&,
                                 UCOutputArray&,
                                 bool){

    throw("Error::StsNotImplemented");
}

int Feature2D::descriptorSize() const
{
    return 0;
}

int Feature2D::descriptorType() const
{
    return 0;
}

int Feature2D::defaultNorm() const
{
    int tp = descriptorType();
    return tp == 2;
}




struct KeypointResponseGreaterThanThreshold
{
    KeypointResponseGreaterThanThreshold(float _value) :
            value(_value)
    {
    }
    inline bool operator()(const KeyPoint& kpt) const
    {
        return kpt.m_response >= value;
    }
    float value;
};

struct KeypointResponseGreater
{
    inline bool operator()(const KeyPoint& kp1, const KeyPoint& kp2) const
    {
        return kp1.m_response > kp2.m_response;
    }
};


// takes keypoints and culls them by the response
void KeyPointsFilter::retainBest(std::vector<KeyPoint>& keypoints, int n_points)
{
    //this is only necessary if the keypoints size is greater than the number of desired points.
    if( n_points >= 0 && keypoints.size() > (size_t)n_points )
    {
        if (n_points==0)
        {
            keypoints.clear();
            return;
        }
        //first use nth element to partition the keypoints into the best and worst.
        std::nth_element(keypoints.begin(), keypoints.begin() + n_points - 1, keypoints.end(), KeypointResponseGreater());
        //this is the boundary response, and in the case of FAST may be ambiguous
        float ambiguous_response = keypoints[n_points - 1].m_response;
        //use std::partition to grab all of the keypoints with the boundary response.
        std::vector<KeyPoint>::const_iterator new_end =
                std::partition(keypoints.begin() + n_points, keypoints.end(),
                               KeypointResponseGreaterThanThreshold(ambiguous_response));
        //resize the keypoints, given this new end point. nth_element and partition reordered the points inplace
        keypoints.resize(new_end - keypoints.begin());
    }
}

struct RoiPredicate
{
    RoiPredicate( const Rect& _r ) : r(_r)
    {}

    bool operator()( const KeyPoint& keyPt ) const
    {
        return !r.contains( keyPt.m_pt );
    }

    Rect r;
};

void KeyPointsFilter::runByImageBorder( std::vector<KeyPoint>& keypoints, Size imageSize, int borderSize )
{
    if( borderSize > 0)
    {
        if (imageSize.height <= borderSize * 2 || imageSize.width <= borderSize * 2)
            keypoints.clear();
        else
            keypoints.erase( std::remove_if(keypoints.begin(), keypoints.end(),
                                            RoiPredicate(Rect(Point(borderSize, borderSize),
                                                              Point(imageSize.width - borderSize, imageSize.height - borderSize)))),
                             keypoints.end() );
    }
}

struct SizePredicate
{
    SizePredicate( float _minSize, float _maxSize ) : minSize(_minSize), maxSize(_maxSize)
    {}

    bool operator()( const KeyPoint& keyPt ) const
    {
        float size = keyPt.m_size;
        return (size < minSize) || (size > maxSize);
    }

    float minSize, maxSize;
};

void KeyPointsFilter::runByKeypointSize( std::vector<KeyPoint>& keypoints, float minSize, float maxSize )
{
    if(minSize<0||maxSize<0||maxSize<minSize)
        throw("runByKeypoint Size error\n");

    keypoints.erase( std::remove_if(keypoints.begin(), keypoints.end(), SizePredicate(minSize, maxSize)),
                     keypoints.end() );
}

class MaskPredicate
{
public:
    MaskPredicate( const UCMat& _mask ) : mask(_mask) {}
    bool operator() (const KeyPoint& key_pt) const
    {
        return mask.at( (int)(key_pt.m_pt.x + 0.5f), (int)(key_pt.m_pt.y + 0.5f),0 ) == 0;
    }

private:
    const UCMat mask;
    MaskPredicate& operator=(const MaskPredicate&);
};

void KeyPointsFilter::runByPixelsMask( std::vector<KeyPoint>& keypoints, const UCMat& mask )
{

    if( mask.empty() )
        return;

    keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(), MaskPredicate(mask)), keypoints.end());
}

struct KeyPoint_LessThan
{
    KeyPoint_LessThan(const std::vector<KeyPoint>& _kp) : kp(&_kp) {}
    bool operator()(int i, int j) const
    {
        const KeyPoint& kp1 = (*kp)[i];
        const KeyPoint& kp2 = (*kp)[j];
        if( kp1.m_pt.x != kp2.m_pt.x )
            return kp1.m_pt.x < kp2.m_pt.x;
        if( kp1.m_pt.y != kp2.m_pt.y )
            return kp1.m_pt.y < kp2.m_pt.y;
        if( kp1.m_size != kp2.m_size )
            return kp1.m_size > kp2.m_size;
        if( kp1.m_angle != kp2.m_angle )
            return kp1.m_angle < kp2.m_angle;
        if( kp1.m_response != kp2.m_response )
            return kp1.m_response > kp2.m_response;
        if( kp1.m_octave != kp2.m_octave )
            return kp1.m_octave > kp2.m_octave;
        if( kp1.m_class_id != kp2.m_class_id )
            return kp1.m_class_id > kp2.m_class_id;

        return i < j;
    }
    const std::vector<KeyPoint>* kp;
};

void KeyPointsFilter::removeDuplicated( std::vector<KeyPoint>& keypoints )
{
    int i, j, n = (int)keypoints.size();
    std::vector<int> kpidx(n);
    std::vector<unsigned char> mask(n, (unsigned char)1);

    for( i = 0; i < n; i++ )
        kpidx[i] = i;
    std::sort(kpidx.begin(), kpidx.end(), KeyPoint_LessThan(keypoints));
    for( i = 1, j = 0; i < n; i++ )
    {
        KeyPoint& kp1 = keypoints[kpidx[i]];
        KeyPoint& kp2 = keypoints[kpidx[j]];
        if( kp1.m_pt.x != kp2.m_pt.x || kp1.m_pt.y != kp2.m_pt.y ||
            kp1.m_size != kp2.m_size || kp1.m_angle != kp2.m_angle )
            j = i;
        else
            mask[kpidx[i]] = 0;
    }

    for( i = j = 0; i < n; i++ )
    {
        if( mask[i] )
        {
            if( i != j )
                keypoints[j] = keypoints[i];
            j++;
        }
    }
    keypoints.resize(j);
}
struct KeyPoint12_LessThan
{
    bool operator()(const KeyPoint &kp1, const KeyPoint &kp2) const
    {
        if( kp1.m_pt.x != kp2.m_pt.x )
            return kp1.m_pt.x < kp2.m_pt.x;
        if( kp1.m_pt.y != kp2.m_pt.y )
            return kp1.m_pt.y < kp2.m_pt.y;
        if( kp1.m_size != kp2.m_size )
            return kp1.m_size > kp2.m_size;
        if( kp1.m_angle != kp2.m_angle )
            return kp1.m_angle < kp2.m_angle;
        if( kp1.m_response != kp2.m_response )
            return kp1.m_response > kp2.m_response;
        if( kp1.m_octave != kp2.m_octave )
            return kp1.m_octave > kp2.m_octave;
        return kp1.m_class_id > kp2.m_class_id;
    }
};

void KeyPointsFilter::removeDuplicatedSorted( std::vector<KeyPoint>& keypoints )
{
    int i, j, n = (int)keypoints.size();

    if (n < 2) return;

    std::sort(keypoints.begin(), keypoints.end(), KeyPoint12_LessThan());

    for( i = 0, j = 1; j < n; ++j )
    {
        const KeyPoint& kp1 = keypoints[i];
        const KeyPoint& kp2 = keypoints[j];
        if( kp1.m_pt.x != kp2.m_pt.x || kp1.m_pt.y != kp2.m_pt.y ||
            kp1.m_size != kp2.m_size || kp1.m_angle != kp2.m_angle ) {
            keypoints[++i] = keypoints[j];
        }
    }
    keypoints.resize(i + 1);
}
}
