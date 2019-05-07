/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
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

#ifndef FEATURES2D_HPP
#define FEATURES2D_HPP
#include <Matrix/matrix.hpp>
#include "types.hpp"
#include "IMAGE/image.hpp"


namespace features2d
{

typedef image::Image<float>        Mat;
typedef image::Image<float>        InputArray;
typedef image::Image<float>        OutputArray;
static const float EPSILON =1.192093e-007;
class  KeyPoint
{
        public:
        //! the default constructor
        KeyPoint();
        /**
        @param _pt x & y coordinates of the keypoint
        @param _size keypoint diameter
        @param _angle keypoint orientation
        @param _response keypoint detector response on the keypoint (that is, strength of the keypoint)
        @param _octave pyramid octave in which the keypoint has been detected
        @param _class_id object id
         */
        KeyPoint(Point2f _pt, float _size, float _angle=-1, float _response=0, int _octave=0, int _class_id=-1);
        /**
        @param x x-coordinate of the keypoint
        @param y y-coordinate of the keypoint
        @param _size keypoint diameter
        @param _angle keypoint orientation
        @param _response keypoint detector response on the keypoint (that is, strength of the keypoint)
        @param _octave pyramid octave in which the keypoint has been detected
        @param _class_id object id
         */
        KeyPoint(float x, float y, float _size, float _angle=-1, float _response=0, int _octave=0, int _class_id=-1);


        Point2f pt; //!< coordinates of the keypoints
        float size; //!< diameter of the meaningful keypoint neighborhood//GSS中尺度值,sig*K^o*2,缩放到原图尺寸。
        float angle; //!< computed orientation of the keypoint (-1 if not applicable);
        //!< it's in [0,360) degrees and measured relative to
        //!< image coordinate system, ie in clockwise.
        float response; //!< the response by which the most strong keypoints have been selected. Can be used for the further sorting or subsampling
        int octave; //!< octave (pyramid layer) from which the keypoint has been extracted
        int class_id; //!< object class (if the keypoints need to be clustered by an object they belong to)
};


/** @brief Abstract base class for 2D image feature detectors and descriptor extractors*/
class  Feature2D
{
public:
    virtual ~Feature2D();

    /** @brief Detects keypoints in an image (first variant) or image set (second variant).

    @param image Image.
    @param keypoints The detected keypoints. In the second variant of the method keypoints[i] is a set
    of keypoints detected in images[i] .
    @param mask Mask specifying where to look for keypoints (optional). It must be a 8-bit integer
    matrix with non-zero values in the region of interest.
     */

    virtual void detect( InputArray& image,std::vector<KeyPoint>& keypoints,
                                 InputArray& mask );

 /** @brief Computes the descriptors for a set of keypoints detected in an image (first variant) or image set
    (second variant).

    @param image Image.
    @param keypoints Input collection of keypoints. Keypoints for which a descriptor cannot be
    computed are removed. Sometimes new keypoints can be added, for example: SIFT duplicates keypoint
    with several dominant orientations (for each orientation).
    @param descriptors Computed descriptors. In the second variant of the method descriptors[i] are
    descriptors computed for a keypoints[i]. Row j is the keypoints (or keypoints[i]) is the
    descriptor for keypoint j-th keypoint.
     */
     virtual void compute( InputArray& image,
                                  std::vector<KeyPoint>& keypoints,
                                  OutputArray& descriptors );




    /** Detects keypoints and computes the descriptors */
     virtual void detectAndCompute( InputArray& image, InputArray& mask,
                                           std::vector<KeyPoint>& keypoints,
                                           OutputArray& descriptors,
                                           bool useProvidedKeypoints=false );

     virtual int descriptorSize() const;
     virtual int descriptorType() const;
     virtual int defaultNorm() const;
};

/** @brief Class for extracting keypoints and computing descriptors using the Scale Invariant Feature Transform
(SIFT) algorithm by D. Lowe @cite Lowe04 .
*/
class SIFT : public Feature2D
{
public:
    /**
    @param nfeatures The number of best features to retain. The features are ranked by their scores
    (measured in SIFT algorithm as the local contrast)

    @param nOctaveLayers The number of layers in each octave. 3 is the value used in D. Lowe paper. The
    number of octaves is computed automatically from the image resolution.

    @param contrastThreshold The contrast threshold used to filter out weak features in semi-uniform
    (low-contrast) regions. The larger the threshold, the less features are produced by the detector.

    @param edgeThreshold The threshold used to filter out edge-like features. Note that the its meaning
    is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are
    filtered out (more features are retained).

    @param sigma The sigma of the Gaussian applied to the input image at the octave \#0. If your image
    is captured with a weak camera with soft lenses, you might want to reduce the number.
     */
    static shared_ptr<SIFT> create( int nfeatures = 0, int nOctaveLayers = 3,
                                     double contrastThreshold = 0.04, double edgeThreshold = 10,
                                     double sigma = 1.6);
};

typedef SIFT SiftFeatureDetector;
typedef SIFT SiftDescriptorExtractor;

/** @brief Class for extracting Speeded Up Robust Features from an image @cite Bay06 .

The algorithm parameters:
-   member int extended
-   0 means that the basic descriptors (64 elements each) shall be computed
-   1 means that the extended descriptors (128 elements each) shall be computed
-   member int upright
-   0 means that detector computes orientation of each feature.
-   1 means that the orientation is not computed (which is much, much faster). For example,
if you match images from a stereo pair, or do image stitching, the matched features
likely have very similar angles, and you can speed up feature extraction by setting
upright=1.
-   member double hessianThreshold
Threshold for the keypoint detector. Only features, whose hessian is larger than
hessianThreshold are retained by the detector. Therefore, the larger the value, the less
keypoints you will get. A good default value could be from 300 to 500, depending from the
image contrast.
-   member int nOctaves
The number of a gaussian pyramid octaves that the detector uses. It is set to 4 by default.
If you want to get very large features, use the larger value. If you want just small
features, decrease it.
-   member int nOctaveLayers
The number of images within each octave of a gaussian pyramid. It is set to 2 by default.
@note
-   An example using the SURF feature detector can be found at
opencv_source_code/samples/cpp/generic_descriptor_match.cpp
-   Another example using the SURF feature detector, extractor and matcher can be found at
opencv_source_code/samples/cpp/matcher_simple.cpp
*/
class  SURF : public Feature2D
{
public:
    /**
    @param hessianThreshold Threshold for hessian keypoint detector used in SURF.
    @param nOctaves Number of pyramid octaves the keypoint detector will use.
    @param nOctaveLayers Number of octave layers within each octave.
    @param extended Extended descriptor flag (true - use extended 128-element descriptors; false - use
    64-element descriptors).
    @param upright Up-right or rotated features flag (true - do not compute orientation of features;
    false - compute orientation).
     */
     static shared_ptr<SURF> create(double hessianThreshold=100,
                                    int nOctaves = 4, int nOctaveLayers = 3,
                                    bool extended = false, bool upright = false);

     virtual void setHessianThreshold(double hessianThreshold) = 0;
     virtual double getHessianThreshold() const = 0;

     virtual void setNOctaves(int nOctaves) = 0;
     virtual int getNOctaves() const = 0;

     virtual void setNOctaveLayers(int nOctaveLayers) = 0;
     virtual int getNOctaveLayers() const = 0;

     virtual void setExtended(bool extended) = 0;
     virtual bool getExtended() const = 0;

     virtual void setUpright(bool upright) = 0;
     virtual bool getUpright() const = 0;
};

typedef SURF SurfFeatureDetector;
typedef SURF SurfDescriptorExtractor;

/** @brief A class filters a vector of keypoints.

 Because now it is difficult to provide a convenient interface for all usage scenarios of the
 keypoints filter class, it has only several needed by now static methods.
 */
class  KeyPointsFilter
{
        public:
        KeyPointsFilter(){}

        /*
         * Remove keypoints within borderPixels of an image edge.
         */
        static void runByImageBorder( std::vector<KeyPoint>& keypoints, Size imageSize, int borderSize );
        /*
         * Remove keypoints of sizes out of range.
         */
        static void runByKeypointSize( std::vector<KeyPoint>& keypoints, float minSize,
        float maxSize=3.402823466e+38F);
        /*
         * Remove keypoints from some image by mask for pixels of this image.
         */
        static void runByPixelsMask( std::vector<KeyPoint>& keypoints, const Mat& mask );
        /*
         * Remove duplicated keypoints.
         */
        static void removeDuplicated( std::vector<KeyPoint>& keypoints );
        /*
         * Remove duplicated keypoints and sort the remaining keypoints
         */
        static void removeDuplicatedSorted( std::vector<KeyPoint>& keypoints );

        /*
         * Retain the specified number of the best keypoints (according to the response)
         */
        static void retainBest( std::vector<KeyPoint>& keypoints, int npoints );
};

}

#endif //FEATURES2D_HPP