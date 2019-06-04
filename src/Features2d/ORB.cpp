/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/** Authors: Ethan Rublee, Vincent Rabaud, Gary Bradski */

#include "Features2d/features2d.h"
#include <iostream>
#include <stdarg.h>
#include <MATH/Matrix/matrix.hpp>
#include <MATH/Matrix/matrix_lu.hpp>
#include "IMAGE/image_process.hpp"
#include <math.h>
#ifndef CV_IMPL_ADD
#define CV_IMPL_ADD(x)
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace features2d
{

    const float kHarris = 0.04f;


#ifdef HAVE_OPENCL
    static bool
ocl_HarrisResponses(const UMat& imgbuf,
                    const UMat& layerinfo,
                    const UMat& keypoints,
                    UMat& responses,
                    int keypoints_num, int blockSize, float harris_k)
{
    size_t globalSize[] = {(size_t)keypoints_num};

    float scale = 1.f/((1 << 2) * blockSize * 255.f);
    float scale_sq_sq = scale * scale * scale * scale;

    ocl::Kernel hr_ker("ORB_HarrisResponses", ocl::features2d::orb_oclsrc,
                format("-D ORB_RESPONSES -D blockSize=%d -D scale_sq_sq=%.12ef -D kHarris=%.12ff", blockSize, scale_sq_sq, harris_k));
    if( hr_ker.empty() )
        return false;

    return hr_ker.args(ocl::KernelArg::ReadOnlyNoSize(imgbuf),
                ocl::KernelArg::PtrReadOnly(layerinfo),
                ocl::KernelArg::PtrReadOnly(keypoints),
                ocl::KernelArg::PtrWriteOnly(responses),
                keypoints_num).run(1, globalSize, 0, true);
}

static bool
ocl_ICAngles(const UMat& imgbuf, const UMat& layerinfo,
             const UMat& keypoints, UMat& responses,
             const UMat& umax, int keypoints_num, int half_k)
{
    size_t globalSize[] = {(size_t)keypoints_num};

    ocl::Kernel icangle_ker("ORB_ICAngle", ocl::features2d::orb_oclsrc, "-D ORB_ANGLES");
    if( icangle_ker.empty() )
        return false;

    return icangle_ker.args(ocl::KernelArg::ReadOnlyNoSize(imgbuf),
                ocl::KernelArg::PtrReadOnly(layerinfo),
                ocl::KernelArg::PtrReadOnly(keypoints),
                ocl::KernelArg::PtrWriteOnly(responses),
                ocl::KernelArg::PtrReadOnly(umax),
                keypoints_num, half_k).run(1, globalSize, 0, true);
}


static bool
ocl_computeOrbDescriptors(const UMat& imgbuf, const UMat& layerInfo,
                          const UMat& keypoints, UMat& desc, const UMat& pattern,
                          int keypoints_num, int dsize, int wta_k)
{
    size_t globalSize[] = {(size_t)keypoints_num};

    ocl::Kernel desc_ker("ORB_computeDescriptor", ocl::features2d::orb_oclsrc,
                         format("-D ORB_DESCRIPTORS -D WTA_K=%d", wta_k));
    if( desc_ker.empty() )
        return false;

    return desc_ker.args(ocl::KernelArg::ReadOnlyNoSize(imgbuf),
                         ocl::KernelArg::PtrReadOnly(layerInfo),
                         ocl::KernelArg::PtrReadOnly(keypoints),
                         ocl::KernelArg::PtrWriteOnly(desc),
                         ocl::KernelArg::PtrReadOnly(pattern),
                         keypoints_num, dsize).run(1, globalSize, 0, true);
}
#endif

/**
 * Function that computes the Harris responses in a
 * blockSize x blockSize patch at given points in the image
 */
static void
harrisResponses(const vector<UCMat>& image_pyramid, std::vector<KeyPoint>& pts, int blockSize, float harris_k)
{
    if(  blockSize*blockSize > 2048 ){
        throw ("blockSize too big\n");
    }

    size_t ptidx, ptsize = pts.size();
    float scale = 1.f/((1 << 2) * blockSize * 255.f);
    float scale_sq_sq = scale * scale * scale * scale;
    int r = blockSize/2;
    for( ptidx = 0; ptidx < ptsize; ptidx++ )
    {
        int x0 = round(pts[ptidx].m_pt.x);
        int y0 = round(pts[ptidx].m_pt.y);
        int z = pts[ptidx].m_octave;
        const UCMat& img=image_pyramid[z];
        int a = 0, b = 0, c = 0;
        int step=img.cols();
        for( int y = y0-r; y <= y0+r; y++ ){
            const unsigned char* ptr=img.ptr(y);
            for(int x=x0-r;x<=x0+r;x++) {
                int Ix = (ptr[x+1] - ptr[x-1]) * 2 + (ptr[x-step + 1] - ptr[x-step - 1]) + (ptr[x+step + 1] - ptr[x+step - 1]);
                int Iy = (ptr[x+step] - ptr[x-step]) * 2 + (ptr[x+step - 1] - ptr[x-step - 1]) +
                         (ptr[x+step + 1] - ptr[x-step + 1]);
                a += Ix * Ix;
                b += Iy * Iy;
                c += Ix * Iy;
            }
        }
        pts[ptidx].m_response = ((float)a * b - (float)c * c -
                               harris_k * ((float)a + b) * ((float)a + b))*scale_sq_sq;
    }

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static void getAngles(const vector<UCMat>& image_pyramid,std::vector<KeyPoint>& pts, int half_k)
{
    size_t ptidx, ptsize = pts.size();


    for( ptidx = 0; ptidx < ptsize; ptidx++ )
    {
        int m_01 = 0, m_10 = 0;
        int x0 = round(pts[ptidx].m_pt.x);
        int y0 = round(pts[ptidx].m_pt.y);
        const UCMat& img=image_pyramid[pts[ptidx].m_octave];
        const unsigned char* center=img.ptr(y0)+x0;
        // Treat the center line differently, v=0
        for (int u = -half_k; u <= half_k; ++u)
            m_10 += u * center[u];
        // Go line by line in the circular patch
        for (int v = 1; v <= half_k; ++v)
        {
            // 同时计算中心对称的上下两行
            int v_sum = 0;
            const unsigned char* top=img.ptr(y0-v)+x0;
            const unsigned char* down=img.ptr(y0+v)+x0;
            for (int u = -half_k; u <= half_k; ++u)
            {
                int val_plus = down[u], val_minus = top[u];
                v_sum += (val_plus - val_minus);
                m_10 += u * (val_plus + val_minus);
            }
            m_01 += v * v_sum;
        }

        pts[ptidx].m_angle = atan((float)m_01, (float)m_10);

    }

//    for( ptidx = 0; ptidx < ptsize; ptidx++ )
//    {
//        int x0 = round(pts[ptidx].m_pt.x);
//        int y0 = round(pts[ptidx].m_pt.y);
//        const UCMat& img=image_pyramid[pts[ptidx].m_octave];
//        int m_01 = 0, m_10 = 0;
//
//        for (int y = -half_k; y <= half_k; ++y)
//        {
//            const unsigned char* ptr=img.ptr(y+y0);
//            for (int x = -half_k; x <= half_k; ++x)
//            {
//               m_01+=y*ptr[x0+x];
//               m_10+=x*ptr[x0+x];
//            }
//        }
//
//        pts[ptidx].m_angle = atan((float)m_01, (float)m_10);
//    }
}


static int s_bit_pattern_31[256*4] =
{
    8,-3, 9,5/*mean (0), correlation (0)*/,
    4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
    -11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
    7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
    2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
    1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
    -2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
    -13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
    -13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
    10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
    -13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
    -11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
    7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
    -4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
    -13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
    -9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
    12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
    -3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
    -6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
    11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
    4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
    5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
    3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
    -8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
    -2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
    -13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
    -7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
    -4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
    -10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
    5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
    5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
    1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
    9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
    4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
    2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
    -4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
    -8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
    4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
    0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
    -13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
    -3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
    -6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
    8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
    0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
    7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
    -13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
    10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
    -6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
    10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
    -13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
    -13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
    3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
    5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
    -1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
    3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
    2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
    -13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
    -13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
    -13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
    -7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
    6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
    -9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
    -2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
    -12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
    3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
    -7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
    -3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
    2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
    -11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
    -1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
    5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
    -4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
    -9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
    -12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
    10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
    7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
    -7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
    -4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
    7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
    -7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
    -13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
    -3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
    7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
    -13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
    1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
    2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
    -4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
    -1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
    7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
    1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
    9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
    -1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
    -13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
    7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
    12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
    6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
    5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
    2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
    3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
    2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
    9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
    -8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
    -11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
    1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
    6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
    2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
    6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
    3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
    7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
    -11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
    -10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
    -5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
    -10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
    8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
    4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
    -10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
    4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
    -2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
    -5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
    7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
    -9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
    -5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
    8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
    -9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
    1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
    7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
    -2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
    11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
    -12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
    3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
    5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
    0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
    -9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
    0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
    -1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
    5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
    3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
    -13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
    -5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
    -4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
    6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
    -7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
    -13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
    1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
    4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
    -2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
    2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
    -2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
    4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
    -6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
    -3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
    7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
    4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
    -13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
    7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
    7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
    -7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
    -8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
    -13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
    2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
    10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
    -6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
    8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
    2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
    -11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
    -12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
    -11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
    5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
    -2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
    -1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
    -13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
    -10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
    -3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
    2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
    -9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
    -4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
    -4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
    -6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
    6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
    -13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
    11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
    7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
    -1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
    -4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
    -7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
    -13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
    -7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
    -8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
    -5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
    -13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
    1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
    1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
    9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
    5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
    -1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
    -9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
    -1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
    -13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
    8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
    2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
    7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
    -10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
    -10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
    4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
    3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
    -4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
    5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
    4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
    -9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
    0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
    -12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
    3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
    -10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
    8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
    -8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
    2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
    10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
    6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
    -7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
    -3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
    -1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
    -3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
    -8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
    4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
    2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
    6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
    3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
    11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
    -3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
    4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
    2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
    -10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
    -13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
    -13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
    6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
    0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
    -13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
    -9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
    -13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
    5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
    2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
    -1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
    9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
    11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
    3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
    -1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
    3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
    -13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
    5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
    8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
    7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
    -10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
    7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
    9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
    7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
    -1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
};


static inline float getScale(int level, int firstLevel, double scaleFactor)
{
    return (float)std::pow(scaleFactor, (double)(level - firstLevel));
}

class ORBImpl : public ORB
{
    public:
    explicit ORBImpl(int _nfeatures, float _scaleFactor, int _nlevels, int _edgeThreshold,
                      int _firstLevel, int _scoreType, int _fastThreshold) :
            m_features_num(_nfeatures), m_scale_factor(_scaleFactor), m_nlevels(_nlevels),
            m_edge_threshold(_edgeThreshold), m_first_level(_firstLevel),
            m_score_type(_scoreType), m_fast_threshold(_fastThreshold)
    {
        m_patch_size=31;
        m_wta_k=2;
    }

    void setMaxFeatures(int maxFeatures) override { m_features_num = maxFeatures; }
    int getMaxFeatures() const override { return m_features_num; }

    void setScaleFactor(double scaleFactor_) override { m_scale_factor = scaleFactor_; }
    double getScaleFactor() const override { return m_scale_factor; }

    void setNLevels(int nlevels_) override { m_nlevels = nlevels_; }
    int getNLevels() const override { return m_nlevels; }

    void setEdgeThreshold(int edgeThreshold_) override { m_edge_threshold = edgeThreshold_; }
    int getEdgeThreshold() const override { return m_edge_threshold; }

    void setFirstLevel(int firstLevel_) override {  m_first_level = std::max(firstLevel_,0); }
    int getFirstLevel() const override { return m_first_level; }

    void setWTA_K(int wta_k_) override { m_wta_k = wta_k_; }
    int getWTA_K() const override { return m_wta_k; }

    void setScoreType(int scoreType_) override { m_score_type = scoreType_; }
    int getScoreType() const override { return m_score_type; }

    void setPatchSize(int patchSize_) override { m_patch_size = patchSize_; }
    int getPatchSize() const override { return m_patch_size; }

    void setFastThreshold(int fastThreshold_) override { m_fast_threshold = fastThreshold_; }
    int getFastThreshold() const override { return m_fast_threshold; }

    // returns the descriptor size in bytes
    int descriptorSize() const override;
    // returns the descriptor type
    int descriptorType() const override;
    // returns the default norm type
    int defaultNorm() const override;

    protected:
    int m_features_num;
    double m_scale_factor;
    int m_nlevels;
    int m_edge_threshold;
    int m_first_level;
    int m_wta_k;
    int m_score_type;
    int m_patch_size;
    int m_fast_threshold;
private:

    void computeOrbDescriptors( const vector<UCMat>& image_pyramid, std::vector<KeyPoint>& keypoints,
                               Mat& descriptors, const std::vector<Point>& _pattern, int dsize );
    void computeKeyPoints(vector<UCMat>& image_pyramid,std::vector<KeyPoint>& all_keypoints,UCInputArray& mask);
    void buildPyramid( const UCMat& base, std::vector<UCMat>& pyr, int nlayers ) const;
    // Compute the ORBImpl features and descriptors on an image
    void detectOrCompute( UCInputArray& image, UCInputArray& mask,
                          std::vector<KeyPoint>& keypoints,
                          OutputArray& descriptors,
                          bool use_provided_keypoints=false )override;

};

int ORBImpl::descriptorSize() const
{
    return kBytes;
}

int ORBImpl::descriptorType() const
{
    return CV_8U;
}

int ORBImpl::defaultNorm() const
{
    return NORM_HAMMING;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ORBImpl::
computeOrbDescriptors( const vector<UCMat>& image_pyramid, std::vector<KeyPoint>& keypoints,
                       Mat& descriptors, const std::vector<Point>& pattern_vec, int dsize )
{
    int j, i, keypoints_num = (int)keypoints.size();
    const float PI=3.141592653;
    for( j = 0; j < keypoints_num; j++ )
    {
        const KeyPoint& kpt = keypoints[j];
        float angle = kpt.m_angle;
        float scale = 1.f/getScale(kpt.m_octave,m_first_level, m_scale_factor);
        angle *= (float)(PI/180.f);
        float a = (float)cos(angle), b = (float)sin(angle);

        const unsigned char* center = image_pyramid[kpt.m_octave].ptr(round(kpt.m_pt.y*scale))+round(kpt.m_pt.x*scale);
        float x, y;
        int ix, iy;
        const Point* pattern = &pattern_vec[0];
        float* desc = descriptors.ptr(j);
        int step=image_pyramid[kpt.m_octave].cols();
#define GET_VALUE(idx) \
       (x = pattern[idx].x*a - pattern[idx].y*b, \
        y = pattern[idx].x*b + pattern[idx].y*a, \
        ix = round(x), \
        iy = round(y), \
        *(center + iy*step + ix) )

        //256个点对，对比结果保存在32*8个bit中
        if( m_wta_k == 2 )
        {
            for (i = 0; i < dsize; ++i, pattern += 16)
            {
                int t0, t1, val;
                t0 = GET_VALUE(0); t1 = GET_VALUE(1);
                val = t0 < t1;
                t0 = GET_VALUE(2); t1 = GET_VALUE(3);
                val |= (t0 < t1) << 1;
                t0 = GET_VALUE(4); t1 = GET_VALUE(5);
                val |= (t0 < t1) << 2;
                t0 = GET_VALUE(6); t1 = GET_VALUE(7);
                val |= (t0 < t1) << 3;
                t0 = GET_VALUE(8); t1 = GET_VALUE(9);
                val |= (t0 < t1) << 4;
                t0 = GET_VALUE(10); t1 = GET_VALUE(11);
                val |= (t0 < t1) << 5;
                t0 = GET_VALUE(12); t1 = GET_VALUE(13);
                val |= (t0 < t1) << 6;
                t0 = GET_VALUE(14); t1 = GET_VALUE(15);
                val |= (t0 < t1) << 7;
                desc[i] = (float)val;
            }
        }
#undef GET_VALUE
    }
}

/** Compute the ORBImpl keypoints on an image
 * @param image_pyramid the image pyramid to compute the features and descriptors on
 * @param mask_pyramid the masks to apply at every level
 * @param keypoints the resulting keypoints, clustered per level
 */
void ORBImpl::computeKeyPoints(vector<UCMat>& image_pyramid,
                             std::vector<KeyPoint>& all_keypoints,UCInputArray& mask)
{
    int i, keypoints_num, level;
    unsigned long nlevels = image_pyramid.size();
    std::vector<int> features_num_per_level(nlevels);

    // fill the extractors and descriptors for the corresponding scales
    float factor = (float)(1.0 / m_scale_factor);
    float features_factor_per_level = m_features_num*(1 - factor)/(1 - (float)std::pow((double)factor, (double)nlevels));

    int features_num_sum = 0;
    for( level = 0; level < nlevels-1; level++ )
    {
        features_num_per_level[level] = round(features_factor_per_level);
        features_num_sum += features_num_per_level[level];
        features_factor_per_level *= factor;
    }
    features_num_per_level[nlevels-1] = std::max(m_features_num - features_num_sum, 0);


    all_keypoints.clear();
    std::vector<KeyPoint> keypoints;
    std::vector<int> counters(nlevels);

    // Detect FAST features, 20 is a good threshold
    shared_ptr<FastFeatureDetector> fd = FastFeatureDetector::create(m_fast_threshold, true);
    for( level = 0; level < nlevels; level++ )
    {
        int featuresNum = features_num_per_level[level];
        UCMat& img = image_pyramid[level];

        fd->detect(img, keypoints, mask);

        // Remove keypoints very close to the border
        KeyPointsFilter::runByImageBorder(keypoints, Size(img.width(),img.height()), m_edge_threshold);

        // Keep more points than necessary as FAST does not give amazing corners
        KeyPointsFilter::retainBest(keypoints, m_score_type == ORBImpl::HARRIS_SCORE ? 2 * featuresNum : featuresNum);

        keypoints_num = (int)keypoints.size();
        counters[level] = keypoints_num;

        float sf = getScale(level, m_first_level, m_scale_factor);
        for( i = 0; i < keypoints_num; i++ )
        {
            keypoints[i].m_octave = level;
            keypoints[i].m_size = m_patch_size*sf;
        }
        std::copy(keypoints.begin(), keypoints.end(), std::back_inserter(all_keypoints));
    }

    keypoints_num = (int)all_keypoints.size();
    if(keypoints_num == 0)
    {
        return;
    }

    // Select best features using the Harris cornerness (better scoring than FAST)
    if( m_score_type == ORBImpl::HARRIS_SCORE )
    {
        harrisResponses(image_pyramid, all_keypoints, 7, kHarris);

        std::vector<KeyPoint> new_all_keypoints;
        new_all_keypoints.reserve(features_num_per_level[0]*nlevels);

        int offset = 0;
        for( level = 0; level < nlevels; level++ )
        {
            int features_num = features_num_per_level[level];
            keypoints_num = counters[level];
            keypoints.resize(keypoints_num);
            std::copy(all_keypoints.begin() + offset,
                      all_keypoints.begin() + offset + keypoints_num,
                      keypoints.begin());
            offset += keypoints_num;

            //cull to the final desired level, using the new Harris scores.
            KeyPointsFilter::retainBest(keypoints, features_num);

            std::copy(keypoints.begin(), keypoints.end(), std::back_inserter(new_all_keypoints));
        }
        std::swap(all_keypoints, new_all_keypoints);
    }

    keypoints_num = (int)all_keypoints.size();
    // pre-compute the end of a row in a circular patch
    int half_patch_size = m_patch_size / 2;
    getAngles(image_pyramid, all_keypoints, half_patch_size);


    for( i = 0; i < keypoints_num; i++ )
    {
        float scale = getScale(all_keypoints[i].m_octave,m_first_level, m_scale_factor);
        all_keypoints[i].m_pt *= scale;
    }
}
void ORBImpl::buildPyramid(const UCMat &base, vector<UCMat> &pyr, int nLevels) const {

    pyr.resize(nLevels);
    pyr[0]=base;
    for(int level = 1; level < nLevels; level++ )
    {
        UCMat& dst=pyr[level];
        float scale = getScale(level, m_first_level, m_scale_factor);
        image::imageResize(base,dst,round(base.cols()/scale), round(base.rows()/scale));
    }
}

/** Compute the ORBImpl features and descriptors on an image
 * @param img the image to compute the features and descriptors on
 * @param mask the mask to apply
 * @param keypoints the resulting keypoints
 * @param descriptors the resulting descriptors
 * @param do_keypoints if true, the keypoints are computed, otherwise used as an input
 * @param do_descriptors if true, also computes the descriptors
 */
void ORBImpl::detectOrCompute( UCInputArray& image, UCInputArray& mask,
                                 std::vector<KeyPoint>& keypoints,
                                 OutputArray& descriptors, bool use_provided_keypoints )
{
    bool do_descriptors = !descriptors.empty();

    if( (use_provided_keypoints && do_descriptors) || image.empty() )
        return;

    int i, level, levels_num = this->m_nlevels, keypoints_num = (int)keypoints.size();
    bool sorted_by_level = true;

    if(use_provided_keypoints )
    {
        // if we have pre-computed keypoints, they may use more levels than it is set in parameters
        // !!!TODO!!! implement more correct method, independent from the used keypoint detector.
        // Namely, the detector should provide correct size of each keypoint. Based on the keypoint size
        // and the algorithm used (i.e. BRIEF, running on 31x31 patches) we should compute the approximate
        // scale-factor that we need to apply. Then we should cluster all the computed scale-factors and
        // for each cluster compute the corresponding image.
        //
        // In short, ultimately the descriptor should
        // ignore octave parameter and deal only with the keypoint size.
        levels_num = 0;
        for( i = 0; i < keypoints_num; i++ )
        {
            level = keypoints[i].m_octave;
            if(level < 0)
                throw("orb level must be positive\n");
            if( i > 0 && level < keypoints[i-1].m_octave )
                sorted_by_level = false;
            levels_num = std::max(levels_num, level);
        }
        levels_num++;
    }


    vector<UCMat> image_pyramid;
    buildPyramid(image,image_pyramid,levels_num);


    if( !use_provided_keypoints )
    {
        // Get keypoints, those will be far enough from the border that no check will be required for the descriptor
        computeKeyPoints(image_pyramid, keypoints,mask);
        KeyPointsFilter::removeDuplicatedSorted( keypoints );
    }
    else
    {
        KeyPointsFilter::runByImageBorder(keypoints, Size(image.width(),image.height()), m_edge_threshold);

        if( !sorted_by_level )
        {
            std::vector<std::vector<KeyPoint> > all_keypoints(levels_num);
            keypoints_num = (int)keypoints.size();
            for( i = 0; i < keypoints_num; i++ )
            {
                level = keypoints[i].m_octave;
                all_keypoints[level].push_back(keypoints[i]);
            }
            keypoints.clear();
            for( level = 0; level < levels_num; level++ )
                std::copy(all_keypoints[level].begin(), all_keypoints[level].end(), std::back_inserter(keypoints));
        }
    }

    if( do_descriptors )
    {
        int dsize = descriptorSize();

        keypoints_num = (int)keypoints.size();
        if( keypoints_num == 0 )
        {
            descriptors.release();
            return;
        }

        descriptors.resize(dsize,keypoints_num,1);
        std::vector<Point> pattern;

        const int npoints = 512;
        const Point* pattern0 = (const Point*)s_bit_pattern_31;
        std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));
        cout<<pattern.size()<<endl;
        for( level = 0; level < levels_num; level++ )
        {
            // preprocess the resized image
            const UCMat working_mat = image_pyramid[level];

            //boxFilter(working_mat, working_mat, working_mat.depth(), Size(5,5), Point(-1,-1), true, BORDER_REFLECT_101);
            //GaussianBlur(working_mat, image_pyramid[level], Size(7, 7), 2, 2, BORDER_REFLECT_101);
            //double t=clock();
            image::blurGaussian<unsigned char>(&working_mat,&image_pyramid[level],2);
            //cout<<"gauss time: "<<(clock()-t)/CLOCKS_PER_SEC*1000.0<<" ms"<<endl;
        }

        computeOrbDescriptors(image_pyramid,keypoints, descriptors, pattern, dsize);

    }
}

shared_ptr<ORB> ORB::create(int nfeatures, float scaleFactor, int nlevels, int edgeThreshold,
                     int firstLevel, int scoreType,  int fastThreshold)
{
    //CV_Assert(firstLevel >= 0);
    ORB * orb_impl=new ORBImpl(nfeatures, scaleFactor, nlevels, edgeThreshold,
                                     firstLevel, scoreType, fastThreshold);
    return shared_ptr<ORB>(orb_impl);
}

}
