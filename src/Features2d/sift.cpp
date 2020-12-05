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

/**********************************************************************************************\
 Implementation of SIFT is based on the code from http://blogs.oregonstate.edu/hess/code/sift/
 Below is the original copyright.

//    Copyright (c) 2006-2010, Rob Hess <hess@eecs.oregonstate.edu>
//    All rights reserved.

//    The following patent has been issued for methods embodied in this
//    software: "Method and apparatus for identifying scale invariant features
//    in an image and use of same for locating an object in an image," David
//    G. Lowe, US Patent 6,711,293 (March 23, 2004). Provisional application
//    filed March 8, 1999. Asignee: The University of British Columbia. For
//    further details, contact David Lowe (lowe@cs.ubc.ca) or the
//    University-Industry Liaison Office of the University of British
//    Columbia.

//    Note that restrictions imposed by this patent (and possibly others)
//    exist independently of and may be in conflict with the freedoms granted
//    in this license, which refers to copyright of the program, not patents
//    for any methods that it implements.  Both copyright and patent law must
//    be obeyed to legally use and redistribute this program and it is not the
//    purpose of this license to induce you to infringe any patents or other
//    property right claims or to contest validity of any such claims.  If you
//    redistribute or use the program, then this license merely protects you
//    from committing copyright infringement.  It does not protect you from
//    committing patent infringement.  So, before you do anything with this
//    program, make sure that you have permission to do so not merely in terms
//    of copyright, but also in terms of patent law.

//    Please note that this license is not to be understood as a guarantee
//    either.  If you use the program according to this license, but in
//    conflict with patent law, it does not mean that the licensor will refund
//    you for any losses that you incur if you are sued for your patent
//    infringement.

//    Redistribution and use in source and binary forms, with or without
//    modification, are permitted provided that the following conditions are
//    met:
//        * Redistributions of source code must retain the above copyright and
//          patent notices, this list of conditions and the following
//          disclaimer.
//        * Redistributions in binary form must reproduce the above copyright
//          notice, this list of conditions and the following disclaimer in
//          the documentation and/or other materials provided with the
//          distribution.
//        * Neither the name of Oregon State University nor the names of its
//          contributors may be used to endorse or promote products derived
//          from this software without specific prior written permission.

//    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
//    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
//    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
//    HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
\**********************************************************************************************/

#include "Features2d/features2d.h"
#include <iostream>
#include <stdarg.h>
#include <MATH/Matrix/matrix.hpp>
#include <MATH/Matrix/matrix_lu.hpp>
#include "IMAGE/image_process.hpp"
#include <math.h>
#define CV_AVX2 1
namespace features2d
{

/*!
 SIFT implementation.

 The class implements SIFT algorithm by D. Lowe.
 */

class SiftImpl : public SIFT
{
public:
    explicit SiftImpl( int features_num = 0, int octave_layers = 3,
                          double contrast_threshold = 0.04, double edge_threshold = 10,
                          double sigma = 1.6);

    //! returns the descriptor size in floats (128)
    int descriptorSize() const ;

    //! returns the descriptor type
    int descriptorType() const ;

    //! returns the default norm type
    int defaultNorm() const ;


private:
    void buildGaussianPyramid( const Mat& base, std::vector<Mat>& pyramid, int octaves_num ) const;
    void buildDoGPyramid( const std::vector<Mat>& pyramid, std::vector<Mat>& dogpyramid ) const;
    void findScaleSpaceExtrema( const std::vector<Mat>& gauss_pyramid, const std::vector<Mat>& dog_pyramid,
                                std::vector<KeyPoint>& keypoints ) const;
    //! finds the keypoints and computes descriptors for them using SIFT algorithm.
    //! Optionally it can compute descriptors for the user-provided keypoints
    void detectOrCompute(UCInputArray& img, UCInputArray& mask,
                          std::vector<KeyPoint>& keypoints,
                          OutputArray& descriptors,
                          bool use_provided_keypoints = false) ;
protected:
     int m_features_num;
     int m_octave_layers;
     double m_contrast_threshold;
     double m_edge_threshold;
     double m_sigma;
};

shared_ptr<SIFT> SIFT::create( int features_num, int octave_num,
                     double contrast_threshold, double edge_threshold, double sigma )
{
    SIFT* sift = new SiftImpl(features_num, octave_num, contrast_threshold, edge_threshold, sigma);
    return shared_ptr<SIFT>(sift);
}

/******************************* Defs and macros *****************************/

// default width of descriptor histogram array
static const int SIFT_DESCR_WIDTH = 4;

// default number of bins per histogram in descriptor array
static const int SIFT_DESCR_HIST_BINS = 8;

// assumed gaussian blur for input image
static const float SIFT_INIT_SIGMA = 0.5f;

// width of border in which to ignore keypoints
static const int SIFT_IMG_BORDER = 5;

// maximum steps of keypoint interpolation before failure
static const int SIFT_MAX_INTERP_STEPS = 5;

// default number of bins in histogram for orientation assignment
static const int SIFT_ORI_HIST_BINS = 36;

// determines gaussian sigma for orientation assignment
static const float SIFT_ORI_SIG_FCTR = 1.5f;

// determines the radius of the region used in orientation assignment
static const float SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR;

// orientation magnitude relative to max that results in new feature
static const float SIFT_ORI_PEAK_RATIO = 0.8f;

// determines the size of a single descriptor orientation histogram
static const float SIFT_DESCR_SCL_FCTR = 3.f;

// threshold on magnitude of elements of descriptor vector
static const float SIFT_DESCR_MAG_THR = 0.2f;

// factor used to convert floating-point descriptor to unsigned char
static const float SIFT_INT_DESCR_FCTR = 512.f;

#define DoG_TYPE_SHORT 0
#if DoG_TYPE_SHORT
// intermediate type used for DoG pyramids
typedef short sift_wt;
static const int SIFT_FIXPT_SCALE = 48;
#else
// intermediate type used for DoG pyramids
typedef float sift_wt;
static const int SIFT_FIXPT_SCALE = 1;
#endif

static inline void
unpackOctave(const KeyPoint& kpt, int& octave, int& layer, float& scale)
{
    octave = kpt.m_octave & 255;
    layer = (kpt.m_octave >> 8) & 255;
    octave = octave < 128 ? octave : (-128 | octave);
    scale = octave >= 0 ? 1.f/(1 << octave) : (float)(1 << -octave);
}

static Mat createInitialImage(const UCMat& gray_uc, bool double_image_size, float sigma )
{
    float sig_diff;
    Mat gray_fpt;
    image::converTo(gray_uc,gray_fpt);
    if( double_image_size )
    {
        sig_diff = sqrtf( std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01f) );
        Mat dbl;
        //resize(gray_fpt, dbl, gray_fpt.cols*2, gray_fpt.rows*2);
        image::imageResize(gray_fpt,dbl,gray_fpt.cols()*2, gray_fpt.rows()*2);
        //Resize(gray_fpt,dbl,gray_fpt.cols()*2, gray_fpt.rows()*2);

        //GaussianBlur(dbl, dbl, Size(), sig_diff, sig_diff);
        Mat blur_dbl;
        image::blurGaussian<float>(&dbl,&blur_dbl,sig_diff);
        //GaussBulr(dbl,blur_dbl,sig_diff);
        return blur_dbl;
    }
    else
    {
        sig_diff = sqrtf( std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01f) );
        //GaussianBlur(gray_fpt, gray_fpt, Size(), sig_diff, sig_diff);
        Mat blur_gray;
        image::blurGaussian<float>(&gray_fpt,&blur_gray,sig_diff);
        return blur_gray;
    }
}


void SiftImpl::buildGaussianPyramid( const Mat& base, std::vector<Mat>& pyramid, int octaves_num ) const
{
    std::vector<double> sig(m_octave_layers + 3);
    pyramid.resize(octaves_num*(m_octave_layers + 3));

    // precompute Gaussian sigmas using the following formula:
    //  \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
    //sig[i] = k^(i-1)*sig*sqrt(k^2-1)
    sig[0] = m_sigma;
    double k = std::pow( 2., 1. / m_octave_layers );
    for( int i = 1; i < m_octave_layers + 3; i++ )
    {
        double sig_prev = std::pow(k, (double)(i-1))*m_sigma;
        double sig_total = sig_prev*k;
        sig[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
    }

    for( int o = 0; o < octaves_num; o++ )
    {
        for( int i = 0; i < m_octave_layers + 3; i++ )
        {
            Mat& dst = pyramid[o*(m_octave_layers + 3) + i];
            if( o == 0  &&  i == 0 )
                dst = base;
            // base of new octave is halved image from end of previous octave
            else if( i == 0 )
            {
                const Mat& src = pyramid[(o-1)*(m_octave_layers + 3) + m_octave_layers];
                //resize(src, dst, Size(src.cols/2, src.rows/2),0, 0, INTER_NEAREST);
                image::imageResize(src,dst,src.cols()/2, src.rows()/2);
                //Resize(src,dst,src.cols()/2, src.rows()/2);
            }
            else
            {
                const Mat& src = pyramid[o*(m_octave_layers + 3) + i-1];
                //GaussianBlur(src, dst, Size(), sig[i], sig[i]);
                image::blurGaussian<float>(&src,&dst,sig[i]);
                //GaussBulr(src,dst,sig[i]);
            }
        }
    }
}


class buildDoGPyramidComputer
{
public:
    buildDoGPyramidComputer(
        int octave_layers,
        const std::vector<Mat>& pyramid,
        std::vector<Mat>& dogpyramid)
        : m_octave_layers(octave_layers),
          m_pyramid(pyramid),
          m_dogpyr(dogpyramid) { }

    void operator()( const int begin,const int end) const
    {
        for( int a = begin; a < end; a++ )
        {
            const int o = a / (m_octave_layers + 2);
            const int i = a % (m_octave_layers + 2);

            const Mat& src1 = m_pyramid[o*(m_octave_layers + 3) + i];
            const Mat& src2 = m_pyramid[o*(m_octave_layers + 3) + i + 1];
            Mat& dst = m_dogpyr[o*(m_octave_layers + 2) + i];
            //subtract(src2, src1, dst, noArray(), DataType<sift_wt>::type);
            image::subtract<float>(src2,src1,dst);
        }
    }

private:
    int m_octave_layers;
    const std::vector<Mat>& m_pyramid;
    std::vector<Mat>& m_dogpyr;
};

void SiftImpl::buildDoGPyramid( const std::vector<Mat>& pyramid, std::vector<Mat>& dogpyramid ) const
{
    int octaves_num = (int)pyramid.size()/(m_octave_layers + 3);
    dogpyramid.resize( octaves_num*(m_octave_layers + 2) );

    //parallel_for_(Range(0, octaves_num * (m_octave_layers + 2)), buildDoGPyramidComputer(m_octave_layers, pyramid, dogpyramid));
    buildDoGPyramidComputer dog_computer(m_octave_layers, pyramid, dogpyramid);
    dog_computer(0,octaves_num * (m_octave_layers + 2));

}

// Computes a gradient orientation histogram at a specified pixel
static float calcOrientationHist( const Mat& img, Point pt, int radius,
                                  float sigma, float* hist, int n )
{
    int i=0, j=0, k=0, len = (radius*2+1)*(radius*2+1);

    float expf_scale = -1.f/(2.f * sigma * sigma);
    float* buf=new float[len*5 + n+4];
    float *X = buf, *Y = X + len, *Ori = Y + len, *W = Ori + len, *Mag = W+len;
    float* temphist = W + len + 2;

    for( i = 0; i < n; i++ )
        temphist[i] = 0.f;

    int y_begain=max(pt.y-radius,1);
    int y_end=min(pt.y+radius,img.rows()-2);
    int x_begain=max(pt.x-radius,1);
    int x_end=min(pt.x+radius,img.cols()-2);
    for( int y = y_begain; y<=y_end; y++ )
    {
        const float *prev=img.ptr(y-1);
        const float *cur=img.ptr(y);
        const float *next=img.ptr(y+1);
        for( int x=x_begain;x<=x_end;x++)
        {
            float dx=cur[x+1]-cur[x-1];
            float dy=prev[x]-next[x];
            i=abs(x-pt.x);
            j=abs(y-pt.y);
            X[k] = dx; Y[k] = dy; W[k] = (i*i + j*j)*expf_scale;
            k++;
        }
    }
    len = k;

    // compute gradient values, orientations and the weights over the pixel neighborhood
    //mag=sqrt(dx*dx+dy*dy)
    //orientation=arctan(dy/dx)
    //weight=e^(-(i*i+j*j)/(2*(1.5sig)^2)
    k = 0;
    for(;k<len;k++){
        W[k]=exp(W[k]);
        Ori[k]=features2d::atan(Y[k],X[k]);
        Mag[k]=sqrt(X[i]*X[i]+Y[i]*Y[i]);
    }
    k=0;
#if CV_AVX2
    __m256 __nd360 = _mm256_set1_ps(n/360.f);
    __m256i __n = _mm256_set1_epi32(n);
    int CV_DECL_ALIGNED(32) bin_buf[8];
    float CV_DECL_ALIGNED(32) w_mul_mag_buf[8];
    for ( ; k <= len - 8; k+=8 )
    {
        __m256i __bin = _mm256_cvtps_epi32(_mm256_mul_ps(__nd360, _mm256_loadu_ps(&Ori[k])));

        __bin = _mm256_sub_epi32(__bin, _mm256_andnot_si256(_mm256_cmpgt_epi32(__n, __bin), __n));
        __bin = _mm256_add_epi32(__bin, _mm256_and_si256(__n, _mm256_cmpgt_epi32(_mm256_setzero_si256(), __bin)));

        __m256 __w_mul_mag = _mm256_mul_ps(_mm256_loadu_ps(&W[k]), _mm256_loadu_ps(&Mag[k]));

        _mm256_store_si256((__m256i *) bin_buf, __bin);
        _mm256_store_ps(w_mul_mag_buf, __w_mul_mag);

        temphist[bin_buf[0]] += w_mul_mag_buf[0];
        temphist[bin_buf[1]] += w_mul_mag_buf[1];
        temphist[bin_buf[2]] += w_mul_mag_buf[2];
        temphist[bin_buf[3]] += w_mul_mag_buf[3];
        temphist[bin_buf[4]] += w_mul_mag_buf[4];
        temphist[bin_buf[5]] += w_mul_mag_buf[5];
        temphist[bin_buf[6]] += w_mul_mag_buf[6];
        temphist[bin_buf[7]] += w_mul_mag_buf[7];
    }
#endif
    for( ; k < len; k++ )
    {
        int bin = round((n/360.f)*Ori[k]);
        if( bin >= n )
            bin -= n;
        if( bin < 0 )
            bin += n;
        temphist[bin] += W[k]*Mag[k];
    }


    // smooth the histogram
    // 数据按照 (n-2),(n-1),0,1,2,3,4,.....,n-1,0,1放置，方便一维滤波。
    temphist[-1] = temphist[n-1];
    temphist[-2] = temphist[n-2];
    temphist[n] = temphist[0];
    temphist[n+1] = temphist[1];

    i = 0;
#if CV_AVX2
    __m256 __d_1_16 = _mm256_set1_ps(1.f/16.f);
    __m256 __d_4_16 = _mm256_set1_ps(4.f/16.f);
    __m256 __d_6_16 = _mm256_set1_ps(6.f/16.f);
    for( ; i <= n - 8; i+=8 )
    {

        __m256 __hist = _mm256_add_ps(
                _mm256_mul_ps(_mm256_add_ps(_mm256_loadu_ps(&temphist[i-2]), _mm256_loadu_ps(&temphist[i+2])),__d_1_16),
                _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_loadu_ps(&temphist[i-1]), _mm256_loadu_ps(&temphist[i+1])),__d_4_16),
                _mm256_mul_ps(_mm256_loadu_ps(&temphist[i]), __d_6_16)));
        _mm256_storeu_ps(&hist[i], __hist);
    }
#endif
    for( ; i < n; i++ )
    {
        hist[i] = (temphist[i-2] + temphist[i+2])*(1.f/16.f) +
            (temphist[i-1] + temphist[i+1])*(4.f/16.f) +
            temphist[i]*(6.f/16.f);
    }

    float maxval = hist[0];
    for( i = 1; i < n; i++ )
        maxval = std::max(maxval, hist[i]);

    delete[] buf;
    return maxval;
}


//
// Interpolates a scale-space extremum's location and scale to subpixel
// accuracy to form an image feature. Rejects features with low contrast.
// Based on Section 4 of Lowe's paper.
static bool adjustLocalExtrema( const std::vector<Mat>& dog_pyramid, KeyPoint& kpt, int octv,
                                int& layer, int& r, int& c, int nOctaveLayers,
                                float contrast_threshold, float edge_threshold, float sigma )
{
    const float img_scale = 1.f/(255*SIFT_FIXPT_SCALE);//float 图像归一化到[0-1]
    const float deriv_scale = img_scale*0.5f;
    const float second_deriv_scale = img_scale;
    const float cross_deriv_scale = img_scale*0.25f;

    float xi=0, xr=0, xc=0, contr=0;
    int i = 0;
    for( ; i < SIFT_MAX_INTERP_STEPS; i++ )
    {
        int idx = octv*(nOctaveLayers+2) + layer;
        const Mat& img = dog_pyramid[idx];
        const Mat& prev = dog_pyramid[idx-1];
        const Mat& next = dog_pyramid[idx+1];

        float dD[3]{(img.at(c+1,r,0) - img.at(c-1,r, 0))*deriv_scale,
                         (img.at(c,r+1, 0) - img.at(c,r-1, 0))*deriv_scale,
                         (next.at(c,r, 0) - prev.at(c,r, 0))*deriv_scale};

        float v2 = (float)img.at(c,r, 0)*2;
        float dxx = (img.at(c+1,r, 0) + img.at(c-1,r, 0) - v2)*second_deriv_scale;
        float dyy = (img.at(c,r+1,0) + img.at(c,r-1, 0) - v2)*second_deriv_scale;
        float dss = (next.at(c,r, 0) + prev.at(c,r, 0) - v2)*second_deriv_scale;
        float dxy = (img.at(c+1,r+1, 0) - img.at(c-1,r+1, 0) -
                     img.at(c+1,r-1, 0) + img.at(c-1,r-1, 0))*cross_deriv_scale;
        float dxs = (next.at(c+1,r, 0) - next.at(c-1,r, 0) -
                     prev.at(c+1,r, 0) + prev.at(c-1,r, 0))*cross_deriv_scale;
        float dys = (next.at(c,r+1, 0) - next.at(c,r-1, 0) -
                     prev.at(c,r+1, 0) + prev.at(c,r-1, 0))*cross_deriv_scale;

//        Matx33f H(dxx, dxy, dxs,
//                  dxy, dyy, dys,
//                  dxs, dys, dss);
//
//        Vec3f X = H.solve(dD, DECOMP_LU);

        float H[9]={dxx, dxy, dxs,
                    dxy, dyy, dys,
                    dxs, dys, dss};
        float X[3];
        lupSolve(H,dD,X,3);
        xi = -X[2];
        xr = -X[1];
        xc = -X[0];

        if( std::abs(xi) < 0.5f && std::abs(xr) < 0.5f && std::abs(xc) < 0.5f )
            break;
        float th=INTMAX_MAX/3;
        if( std::abs(xi) > th ||
            std::abs(xr) > th ||
            std::abs(xc) > th )
            return false;

        c += round(xc);
        r += round(xr);
        layer += round(xi);

        if( layer < 1 || layer > nOctaveLayers ||
            c < SIFT_IMG_BORDER || c >= img.cols() - SIFT_IMG_BORDER  ||
            r < SIFT_IMG_BORDER || r >= img.rows() - SIFT_IMG_BORDER )
            return false;
    }

    // ensure convergence of interpolation
    if( i >= SIFT_MAX_INTERP_STEPS )
        return false;
    {
        int idx = octv*(nOctaveLayers+2) + layer;
        const Mat& img = dog_pyramid[idx];
        const Mat& prev = dog_pyramid[idx-1];
        const Mat& next = dog_pyramid[idx+1];

        /*
        * 极值点contr的求解公式contr = f(x0) + 0.5* delta.dot(D)
        * 其中
        * f(x0)--表示插值点(ix, iy, is) 处的DoG值，可通过dogs[1]->at(ix, iy, 0)获取
        * delta--为上述求得的delta=[delta_x, delta_y, delta_s]
        * D--为一阶导数，表示为(Dx, Dy, Ds)
        */
        float dD[3]{(img.at(c+1,r, 0) - img.at(c-1,r, 0))*deriv_scale,
                    (img.at(c,r+1, 0) - img.at(c,r-1, 0))*deriv_scale,
                    (next.at(c,r, 0) - prev.at(c,r, 0))*deriv_scale};
        float t = dD[0]*xc+dD[1]*xr+dD[2]*xi;//dD.dot(Matx31f(xc, xr, xi));

        contr = img.at(c, r, 0)*img_scale + t * 0.5f;
        if( std::abs( contr ) * nOctaveLayers < contrast_threshold )
            return false;

        // principal curvatures are computed using the trace and det of Hessian
        float v2 = (float)img.at(c,r, 0)*2;
        float dxx = (img.at(c+1,r, 0) + img.at(c-1,r, 0) - v2)*second_deriv_scale;
        float dyy = (img.at(c,r+1, 0) + img.at(c,r-1, 0) - v2)*second_deriv_scale;
        float dxy = (img.at(c+1,r+1, 0) - img.at(c-1,r+1, 0) -
                     img.at(c+1,r-1, 0) + img.at(c-1,r-1, 0))*cross_deriv_scale;

        float tr = dxx + dyy;
        float det = dxx * dyy - dxy * dxy;

        if( det <= 0 || tr*tr*edge_threshold >= (edge_threshold + 1)*(edge_threshold + 1)*det )
            return false;
    }
    kpt.m_pt.x = (c + xc) * (1 << octv);
    kpt.m_pt.y = (r + xr) * (1 << octv);
    kpt.m_octave = octv + (layer << 8) + (round((xi + 0.5)*255) << 16);
    kpt.m_size = sigma*powf(2.f, (layer + xi) / nOctaveLayers)*(1 << octv)*2;
    kpt.m_response = std::abs(contr);

    return true;
}


class findScaleSpaceExtremaComputer
{
public:
    findScaleSpaceExtremaComputer(
        int threshold,
        int octave_num,
        double contrast_threshold,
        double edge_threshold,
        double sigma,
        const std::vector<Mat>& gauss_pyr,
        const std::vector<Mat>& dog_pyr,
        std::vector<KeyPoint> & tls_kpts_struct):
          m_threshold(threshold),
          m_octave_layers(octave_num),
          m_contrast_threshold(contrast_threshold),
          m_edge_threshold(edge_threshold),
          m_sigma(sigma),
          m_gauss_pyramid(gauss_pyr),
          m_dogpyramid(dog_pyr),
          m_tls_kpts(tls_kpts_struct) { }
    void operator()(int o,int i) const
    {
        static const int n = SIFT_ORI_HIST_BINS;
        float hist[n];
        int idx = o*(m_octave_layers+2)+i;
        const Mat& img = m_dogpyramid[idx];
        const Mat& prev = m_dogpyramid[idx-1];
        const Mat& next = m_dogpyramid[idx+1];
        int step=img.cols()*img.channels();

        KeyPoint kpt;
        for( int r = SIFT_IMG_BORDER; r < img.rows()-SIFT_IMG_BORDER; r++)
        {
            const sift_wt* currptr = img.ptr(r);
            const sift_wt* prevptr = prev.ptr(r);
            const sift_wt* nextptr = next.ptr(r);

            for( int c = SIFT_IMG_BORDER; c < img.cols()-SIFT_IMG_BORDER; c++)
            {
                sift_wt val = currptr[c];
                // find local extrema with pixel accuracy
                // 相邻层3×3邻域内最大或最小
                if( std::abs(val) > m_threshold &&
                   ((val > 0 && val >= currptr[c-1] && val >= currptr[c+1] &&
                     val >= currptr[c-step-1] && val >= currptr[c-step] && val >= currptr[c-step+1] &&
                     val >= currptr[c+step-1] && val >= currptr[c+step] && val >= currptr[c+step+1] &&
                     val >= nextptr[c] && val >= nextptr[c-1] && val >= nextptr[c+1] &&
                     val >= nextptr[c-step-1] && val >= nextptr[c-step] && val >= nextptr[c-step+1] &&
                     val >= nextptr[c+step-1] && val >= nextptr[c+step] && val >= nextptr[c+step+1] &&
                     val >= prevptr[c] && val >= prevptr[c-1] && val >= prevptr[c+1] &&
                     val >= prevptr[c-step-1] && val >= prevptr[c-step] && val >= prevptr[c-step+1] &&
                     val >= prevptr[c+step-1] && val >= prevptr[c+step] && val >= prevptr[c+step+1]) ||
                    (val < 0 && val <= currptr[c-1] && val <= currptr[c+1] &&
                     val <= currptr[c-step-1] && val <= currptr[c-step] && val <= currptr[c-step+1] &&
                     val <= currptr[c+step-1] && val <= currptr[c+step] && val <= currptr[c+step+1] &&
                     val <= nextptr[c] && val <= nextptr[c-1] && val <= nextptr[c+1] &&
                     val <= nextptr[c-step-1] && val <= nextptr[c-step] && val <= nextptr[c-step+1] &&
                     val <= nextptr[c+step-1] && val <= nextptr[c+step] && val <= nextptr[c+step+1] &&
                     val <= prevptr[c] && val <= prevptr[c-1] && val <= prevptr[c+1] &&
                     val <= prevptr[c-step-1] && val <= prevptr[c-step] && val <= prevptr[c-step+1] &&
                     val <= prevptr[c+step-1] && val <= prevptr[c+step] && val <= prevptr[c+step+1])))
                {

                    int r1 = r, c1 = c, layer = i;
                    if( !adjustLocalExtrema(m_dogpyramid, kpt, o, layer, r1, c1,
                                            m_octave_layers, (float)m_contrast_threshold,
                                            (float)m_edge_threshold, (float)m_sigma) )
                        continue;

                    float scl_octv = kpt.m_size*0.5f/(1 << o);
                    float omax = calcOrientationHist(m_gauss_pyramid[o*(m_octave_layers+3) + layer],
                                                     Point(c1, r1),
                                                     round(SIFT_ORI_RADIUS * scl_octv),
                                                     SIFT_ORI_SIG_FCTR * scl_octv,
                                                     hist, n);
                    float mag_thr = (float)(omax * SIFT_ORI_PEAK_RATIO);
                   // const float FLT_EPSILON =1.192093e-007;
                    for( int j = 0; j < n; j++ )
                    {
                        int l = j > 0 ? j - 1 : n - 1;
                        int r2 = j < n-1 ? j + 1 : 0;

                        if( hist[j] > hist[l]  &&  hist[j] > hist[r2]  &&  hist[j] >= mag_thr )
                        {
                            float bin = j + 0.5f * (hist[l]-hist[r2]) / (hist[l] - 2*hist[j] + hist[r2]);
                            bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;
                            kpt.m_angle = 360.f - (float)((360.f/n) * bin);
                            if(std::abs(kpt.m_angle - 360.f) < EPSILON)
                                kpt.m_angle = 0.f;
                            {
                                m_tls_kpts.push_back(kpt);
                            }
                        }
                    }
                }
            }
        }
    }
private:
    int m_threshold;
    int m_octave_layers;
    double m_contrast_threshold;
    double m_edge_threshold;
    double m_sigma;
    const std::vector<Mat>& m_gauss_pyramid;
    const std::vector<Mat>& m_dogpyramid;
    std::vector<KeyPoint>&  m_tls_kpts;
};

//
// Detects features at extrema in DoG scale space.  Bad features are discarded
// based on contrast and ratio of principal curvatures.
void SiftImpl::findScaleSpaceExtrema( const std::vector<Mat>& gauss_pyramid, const std::vector<Mat>& dog_pyramid,
                                  std::vector<KeyPoint>& keypoints ) const
{
    const int octaves_num = (int)gauss_pyramid.size()/(m_octave_layers + 3);
    const int threshold = floor(0.5 * m_contrast_threshold / m_octave_layers * 255 * SIFT_FIXPT_SCALE);

    keypoints.clear();
    std::vector<KeyPoint>  tls_kpts_struct;
    findScaleSpaceExtremaComputer computer(
            threshold,
            m_octave_layers,
            m_contrast_threshold,
            m_edge_threshold,
            m_sigma,
            gauss_pyramid, dog_pyramid, tls_kpts_struct);

    for( int o = 0; o < octaves_num; o++ )
        for( int i = 1; i <= m_octave_layers; i++ )
        {
            computer(o,i);
        }

    keypoints.insert(keypoints.end(),tls_kpts_struct.begin(),tls_kpts_struct.end());
}


static void calcSIFTDescriptor( const Mat& img, Point2f ptf, float ori, float scale,
                               int d, int n, float* dst )
{
    const float PI=3.141592653;
    Point pt(round(ptf.x), round(ptf.y));
    float cos_t = cosf(ori*(float)(PI/180));
    float sin_t = sinf(ori*(float)(PI/180));
    float bins_per_rad = n / 360.f;
    float exp_scale = -1.f/(d * d * 0.5f);
    float hist_width = SIFT_DESCR_SCL_FCTR * scale;
    int radius = round(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);
    // Clip the radius to the diagonal of the image to avoid autobuffer too large exception
    radius = std::min(radius, (int) sqrt(((double) img.cols())*img.cols() + ((double) img.rows())*img.rows()));
    cos_t /= hist_width;
    sin_t /= hist_width;

    int i, j, k, len = (radius*2+1)*(radius*2+1), histlen = (d+2)*(d+2)*(n+2);
    int rows = img.rows(), cols = img.cols();


    float* buf=new float[len*8+histlen];
    float *X = buf, *Y = X + len, *Mag = Y + len, *Ori = Mag + len, *W = Ori + len;
    float *RBin = W + len, *CBin = RBin + len, *hist = CBin + len;

    for( i = 0; i < d+2; i++ )
    {
        for( j = 0; j < d+2; j++ )
            for( k = 0; k < n+2; k++ )
                hist[(i*(d+2) + j)*(n+2) + k] = 0.;
    }

    for( i = -radius, k = 0; i <= radius; i++ ) {
        int r = pt.y + i;
        if(r <= 1 || r >= rows - 1)
            continue;
        const float *prev = img.ptr(r - 1);
        const float *cur=img.ptr(r);
        const float *next=img.ptr(r+1);
        for (j = -radius; j <= radius; j++) {
            int c = pt.x + j;
            if( c <= 1 || c >= cols - 1)
                continue;
            // Calculate sample's histogram array coords rotated relative to ori.
            // Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
            // r_rot = 1.5) have full weight placed in row 1 after interpolation.
            float c_rot = j * cos_t - i * sin_t;
            float r_rot = j * sin_t + i * cos_t;
            float rbin = r_rot + d / 2 - 0.5f;
            float cbin = c_rot + d / 2 - 0.5f;

            if (rbin > -1 && rbin < d && cbin > -1 && cbin < d) {
                float dx=cur[c+1]-cur[c-1];
                float dy=prev[c]-next[c];
                X[k] = dx;
                Y[k] = dy;
                RBin[k] = rbin;
                CBin[k] = cbin;
                W[k] = (c_rot * c_rot + r_rot * r_rot) * exp_scale;
                k++;
            }
        }
    }
    len = k;
    //cv::hal::fastAtan2(Y, X, Ori, len, true);
    //cv::hal::magnitude32f(X, Y, Mag, len);
    //cv::hal::exp32f(W, W, len);
    k = 0;
    for(;k<len;k++){
        Mag[k]=sqrt(X[i]*X[i]+Y[i]*Y[i]);
        Ori[k]=features2d::atan(Y[k],X[k]);
        W[k]=exp(W[k]);
    }

    k = 0;

#if CV_AVX2

    int CV_DECL_ALIGNED(32) idx_buf[8];
    float CV_DECL_ALIGNED(32) rco_buf[64];
    const __m256 __ori = _mm256_set1_ps(ori);
    const __m256 __bins_per_rad = _mm256_set1_ps(bins_per_rad);
    const __m256i __n = _mm256_set1_epi32(n);
    for( ; k <= len - 8; k+=8 )
    {
        __m256 __rbin = _mm256_loadu_ps(&RBin[k]);
        __m256 __cbin = _mm256_loadu_ps(&CBin[k]);
        __m256 __obin = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(&Ori[k]), __ori), __bins_per_rad);
        __m256 __mag = _mm256_mul_ps(_mm256_loadu_ps(&Mag[k]), _mm256_loadu_ps(&W[k]));

        __m256 __r0 = _mm256_floor_ps(__rbin);
        __rbin = _mm256_sub_ps(__rbin, __r0);
        __m256 __c0 = _mm256_floor_ps(__cbin);
        __cbin = _mm256_sub_ps(__cbin, __c0);
        __m256 __o0 = _mm256_floor_ps(__obin);
        __obin = _mm256_sub_ps(__obin, __o0);

        __m256i __o0i = _mm256_cvtps_epi32(__o0);
        __o0i = _mm256_add_epi32(__o0i, _mm256_and_si256(__n, _mm256_cmpgt_epi32(_mm256_setzero_si256(), __o0i)));
        __o0i = _mm256_sub_epi32(__o0i, _mm256_andnot_si256(_mm256_cmpgt_epi32(__n, __o0i), __n));

        __m256 __v_r1 = _mm256_mul_ps(__mag, __rbin);
        __m256 __v_r0 = _mm256_sub_ps(__mag, __v_r1);

        __m256 __v_rc11 = _mm256_mul_ps(__v_r1, __cbin);
        __m256 __v_rc10 = _mm256_sub_ps(__v_r1, __v_rc11);

        __m256 __v_rc01 = _mm256_mul_ps(__v_r0, __cbin);
        __m256 __v_rc00 = _mm256_sub_ps(__v_r0, __v_rc01);

        __m256 __v_rco111 = _mm256_mul_ps(__v_rc11, __obin);
        __m256 __v_rco110 = _mm256_sub_ps(__v_rc11, __v_rco111);

        __m256 __v_rco101 = _mm256_mul_ps(__v_rc10, __obin);
        __m256 __v_rco100 = _mm256_sub_ps(__v_rc10, __v_rco101);

        __m256 __v_rco011 = _mm256_mul_ps(__v_rc01, __obin);
        __m256 __v_rco010 = _mm256_sub_ps(__v_rc01, __v_rco011);

        __m256 __v_rco001 = _mm256_mul_ps(__v_rc00, __obin);
        __m256 __v_rco000 = _mm256_sub_ps(__v_rc00, __v_rco001);

        __m256i __one = _mm256_set1_epi32(1);
        __m256i __idx = _mm256_add_epi32(
            _mm256_mullo_epi32(
                _mm256_add_epi32(
                    _mm256_mullo_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(__r0), __one), _mm256_set1_epi32(d + 2)),
                    _mm256_add_epi32(_mm256_cvtps_epi32(__c0), __one)),
                _mm256_set1_epi32(n + 2)),
            __o0i);

        _mm256_store_si256((__m256i *)idx_buf, __idx);

        _mm256_store_ps(&(rco_buf[0]),  __v_rco000);
        _mm256_store_ps(&(rco_buf[8]),  __v_rco001);
        _mm256_store_ps(&(rco_buf[16]), __v_rco010);
        _mm256_store_ps(&(rco_buf[24]), __v_rco011);
        _mm256_store_ps(&(rco_buf[32]), __v_rco100);
        _mm256_store_ps(&(rco_buf[40]), __v_rco101);
        _mm256_store_ps(&(rco_buf[48]), __v_rco110);
        _mm256_store_ps(&(rco_buf[56]), __v_rco111);
        #define HIST_SUM_HELPER(id)                                  \
            hist[idx_buf[(id)]] += rco_buf[(id)];                    \
            hist[idx_buf[(id)]+1] += rco_buf[8 + (id)];              \
            hist[idx_buf[(id)]+(n+2)] += rco_buf[16 + (id)];         \
            hist[idx_buf[(id)]+(n+3)] += rco_buf[24 + (id)];         \
            hist[idx_buf[(id)]+(d+2)*(n+2)] += rco_buf[32 + (id)];   \
            hist[idx_buf[(id)]+(d+2)*(n+2)+1] += rco_buf[40 + (id)]; \
            hist[idx_buf[(id)]+(d+3)*(n+2)] += rco_buf[48 + (id)];   \
            hist[idx_buf[(id)]+(d+3)*(n+2)+1] += rco_buf[56 + (id)];

        HIST_SUM_HELPER(0);
        HIST_SUM_HELPER(1);
        HIST_SUM_HELPER(2);
        HIST_SUM_HELPER(3);
        HIST_SUM_HELPER(4);
        HIST_SUM_HELPER(5);
        HIST_SUM_HELPER(6);
        HIST_SUM_HELPER(7);

        #undef HIST_SUM_HELPER
    }
#endif
    for( ; k < len; k++ )
    {
        float rbin = RBin[k], cbin = CBin[k];
        float obin = (Ori[k] - ori)*bins_per_rad;
        float mag = Mag[k]*W[k];

        int r0 = floor( rbin );
        int c0 = floor( cbin );
        int o0 = floor( obin );
        rbin -= r0;
        cbin -= c0;
        obin -= o0;

        if( o0 < 0 )
            o0 += n;
        if( o0 >= n )
            o0 -= n;

        // histogram update using tri-linear interpolation
        float v_r1 = mag*rbin, v_r0 = mag - v_r1;
        float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
        float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
        float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
        float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
        float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
        float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

        int idx = ((r0+1)*(d+2) + c0+1)*(n+2) + o0;
        hist[idx] += v_rco000;
        hist[idx+1] += v_rco001;
        hist[idx+(n+2)] += v_rco010;
        hist[idx+(n+3)] += v_rco011;
        hist[idx+(d+2)*(n+2)] += v_rco100;
        hist[idx+(d+2)*(n+2)+1] += v_rco101;
        hist[idx+(d+3)*(n+2)] += v_rco110;
        hist[idx+(d+3)*(n+2)+1] += v_rco111;
    }

    // finalize histogram, since the orientation histograms are circular
    for( i = 0; i < d; i++ )
        for( j = 0; j < d; j++ )
        {
            int idx = ((i+1)*(d+2) + (j+1))*(n+2);
            hist[idx] += hist[idx+n];
            hist[idx+1] += hist[idx+n+1];
            for( k = 0; k < n; k++ )
                dst[(i*d + j)*n + k] = hist[idx+k];
        }
    // copy histogram to the descriptor,
    // apply hysteresis thresholding
    // and scale the result, so that it can be easily converted
    // to byte array
    float nrm2 = 0;
    len = d*d*n;
    k = 0;
#if CV_AVX2
        float CV_DECL_ALIGNED(32) nrm2_buf[8];
        __m256 __nrm2 = _mm256_setzero_ps();
        __m256 __dst;
        for( ; k <= len - 8; k += 8 )
        {
            __dst = _mm256_loadu_ps(&dst[k]);
            __nrm2 = _mm256_add_ps(__nrm2, _mm256_mul_ps(__dst, __dst));

        }
        _mm256_store_ps(nrm2_buf, __nrm2);
        nrm2 = nrm2_buf[0] + nrm2_buf[1] + nrm2_buf[2] + nrm2_buf[3] +
               nrm2_buf[4] + nrm2_buf[5] + nrm2_buf[6] + nrm2_buf[7];
#endif
    for( ; k < len; k++ )
        nrm2 += dst[k]*dst[k];


    float thr = std::sqrt(nrm2)*SIFT_DESCR_MAG_THR;
    i = 0, nrm2 = 0;
    for( ; i < len; i++ )
    {
        float val = std::min(dst[i], thr);
        dst[i] = val;
        nrm2 += val*val;
    }
    nrm2 = SIFT_INT_DESCR_FCTR/std::max(std::sqrt(nrm2), EPSILON);

    k = 0;
    for( ; k < len; k++ )
    {
        dst[k] = (dst[k]*nrm2);
        //cout<<dst[k]<<endl;
    }

    delete[] buf;
}

class calcDescriptorsComputer
{
public:
    calcDescriptorsComputer(const std::vector<Mat>& pyramid,
                            const std::vector<KeyPoint>& keypoints,
                            Mat& descriptors,
                            int octave_num,
                            int firstOctave)
        : m_pyramid(pyramid),
          m_keypoints(keypoints),
          m_descriptors(descriptors),
          m_octave_layers(octave_num),
          m_firstOctave(firstOctave) { }

    void operator()( const int begin,const int end ) const
    {

        static const int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;
        //const float FLT_EPSILON =1.192093e-007;
        for ( int i = begin; i<end; i++ )
        {
            KeyPoint kpt = m_keypoints[i];
            int octave, layer;
            float scale;
            unpackOctave(kpt, octave, layer, scale);
            if(!(octave >= m_firstOctave && layer <= m_octave_layers+2))
                throw("octave or layer index error\n");
            float size=kpt.m_size*scale;
            Point2f ptf(kpt.m_pt.x*scale, kpt.m_pt.y*scale);
            const Mat& img = m_pyramid[(octave - m_firstOctave)*(m_octave_layers + 3) + layer];

            float angle = 360.f - kpt.m_angle;
            if(std::abs(angle - 360.f) < EPSILON)
                angle = 0.f;
            calcSIFTDescriptor(img, ptf, angle, size*0.5f, d, n, m_descriptors.ptr((int)i));
        }
    }
private:
    const std::vector<Mat>& m_pyramid;
    const std::vector<KeyPoint>& m_keypoints;
    Mat& m_descriptors;
    int m_octave_layers;
    int m_firstOctave;
};

static void calcDescriptors(const std::vector<Mat>& pyramid, const std::vector<KeyPoint>& keypoints,
                            Mat& descriptors, int nOctaveLayers, int firstOctave )
{
    //parallel_for_(Range(0, static_cast<int>(keypoints.size())), calcDescriptorsComputer(pyramid, keypoints, descriptors, nOctaveLayers, firstOctave));
    calcDescriptorsComputer descriptorsComputer(pyramid, keypoints, descriptors, nOctaveLayers, firstOctave);
    descriptorsComputer(0,keypoints.size());

}

//////////////////////////////////////////////////////////////////////////////////////////

SiftImpl::SiftImpl( int features_num, int octave_num,
           double contrast_threshold, double edge_threshold, double sigma )
    : m_features_num(features_num), m_octave_layers(octave_num),
    m_contrast_threshold(contrast_threshold), m_edge_threshold(edge_threshold), m_sigma(sigma)
{
}

int SiftImpl::descriptorSize() const
{
    return SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS;
}

int SiftImpl::descriptorType() const
{
    return CV_32F;
}

int SiftImpl::defaultNorm() const
{
    return NORM_L2;
}


void SiftImpl::detectOrCompute(UCInputArray& image, UCInputArray& mask,
                      std::vector<KeyPoint>& keypoints,
                      OutputArray& descriptors,
                      bool use_provided_keypoints)
{
    int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;


    if( image.empty() || image.channels() != 1 )
        throw("image is empty or has incorrect depth" );

    if( !mask.empty() && mask.channels() != 1 )
        throw("mask has incorrect depth" );

    if( use_provided_keypoints )
    {
        firstOctave = 0;
        int maxOctave = -1;
        for( size_t i = 0; i < keypoints.size(); i++ )
        {
            int octave, layer;
            float scale;
            unpackOctave(keypoints[i], octave, layer, scale);
            firstOctave = std::min(firstOctave, octave);
            maxOctave = std::max(maxOctave, octave);
            actualNLayers = std::max(actualNLayers, layer-2);
        }

        firstOctave = std::min(firstOctave, 0);
        if(!( firstOctave >= -1 && actualNLayers <= m_octave_layers )){
            throw("( firstOctave >= -1 && actualNLayers <= m_octave_layers ) must be true\n");
        };
        actualNOctaves = maxOctave - firstOctave + 1;
    }

    Mat base = createInitialImage(image, firstOctave < 0, (float)m_sigma);
    std::vector<Mat> pyramid, dogpyramid;
    //octaves_num = log2(min(rows,cols))-a;采样后最小尺寸为2^(a+1);
    //firstOctave==-1时，图像下采样为两倍尺寸
    int octaves_num = actualNOctaves > 0 ? actualNOctaves : round(std::log( (double)std::min( base.cols(), base.rows() ) ) / std::log(2.) - 2) - firstOctave;
    if(octaves_num<1)
    {
        throw("image too small to SIFT\n");
    }

    double t, tf = CLOCKS_PER_SEC;
    t = (double)clock();
    buildGaussianPyramid(base, pyramid, octaves_num);
    buildDoGPyramid(pyramid, dogpyramid);

    t = (double)clock() - t;
    printf("pyramid construction time: %g\n", t*1000./tf);

    if( !use_provided_keypoints )
    {
        t = (double)clock();
        findScaleSpaceExtrema(pyramid, dogpyramid, keypoints);
        KeyPointsFilter::removeDuplicatedSorted( keypoints );

        if( m_features_num > 0 )
            KeyPointsFilter::retainBest(keypoints, m_features_num);
        t = (double)clock() - t;
        printf("keypoint detection time: %g\n", t*1000./tf);

        if( firstOctave < 0 )
            for( size_t i = 0; i < keypoints.size(); i++ )
            {
                KeyPoint& kpt = keypoints[i];
                float scale = 1.f/(float)(1 << -firstOctave);
                kpt.m_octave = (kpt.m_octave & ~255) | ((kpt.m_octave + firstOctave) & 255);
                kpt.m_pt *= scale;
                kpt.m_size *= scale;
            }

        if( !mask.empty() )
            KeyPointsFilter::runByPixelsMask( keypoints, mask );
    }
    if( !descriptors.empty())
    {
        t = (double)clock();
        int dsize = descriptorSize();
        descriptors.resize(dsize, (int)keypoints.size(), 1);


        calcDescriptors(pyramid, keypoints, descriptors, m_octave_layers, firstOctave);
        t = (double)clock() - t;
        printf("descriptor extraction time: %g\n", t*1000./tf);
    }
}



}

