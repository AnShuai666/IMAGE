//
// Created by doing on 19-5-22.
//
#include "features2d/features2d.h"
#include <memory.h>
#include <IMAGE/image_process.hpp>

namespace features2d {

int cornerScore_16(const unsigned char* ptr, const int pixel[], int threshold)
{
    const int K = 8, N = K*3 + 1;
    //v为当前像素值
    int k, v = ptr[0];
    short d[N];
    //计算当前像素值与其圆周像素值之间的差值
    for( k = 0; k < N; k++ )
        d[k] = (short)(v - ptr[pixel[k]]);

#if CV_SSE2
    __m128i q0 = _mm_set1_epi16(-1000), q1 = _mm_set1_epi16(1000);
for( k = 0; k < 16; k += 8 )
{
    __m128i v0 = _mm_loadu_si128((__m128i*)(d+k+1));
    __m128i v1 = _mm_loadu_si128((__m128i*)(d+k+2));
    __m128i a = _mm_min_epi16(v0, v1);
    __m128i b = _mm_max_epi16(v0, v1);
    v0 = _mm_loadu_si128((__m128i*)(d+k+3));
    a = _mm_min_epi16(a, v0);
    b = _mm_max_epi16(b, v0);
    v0 = _mm_loadu_si128((__m128i*)(d+k+4));
    a = _mm_min_epi16(a, v0);
    b = _mm_max_epi16(b, v0);
    v0 = _mm_loadu_si128((__m128i*)(d+k+5));
    a = _mm_min_epi16(a, v0);
    b = _mm_max_epi16(b, v0);
    v0 = _mm_loadu_si128((__m128i*)(d+k+6));
    a = _mm_min_epi16(a, v0);
    b = _mm_max_epi16(b, v0);
    v0 = _mm_loadu_si128((__m128i*)(d+k+7));
    a = _mm_min_epi16(a, v0);
    b = _mm_max_epi16(b, v0);
    v0 = _mm_loadu_si128((__m128i*)(d+k+8));
    a = _mm_min_epi16(a, v0);
    b = _mm_max_epi16(b, v0);
    v0 = _mm_loadu_si128((__m128i*)(d+k));
    q0 = _mm_max_epi16(q0, _mm_min_epi16(a, v0));
    q1 = _mm_min_epi16(q1, _mm_max_epi16(b, v0));
    v0 = _mm_loadu_si128((__m128i*)(d+k+9));
    q0 = _mm_max_epi16(q0, _mm_min_epi16(a, v0));
    q1 = _mm_min_epi16(q1, _mm_max_epi16(b, v0));
}
q0 = _mm_max_epi16(q0, _mm_sub_epi16(_mm_setzero_si128(), q1));
q0 = _mm_max_epi16(q0, _mm_unpackhi_epi64(q0, q0));
q0 = _mm_max_epi16(q0, _mm_srli_si128(q0, 4));
q0 = _mm_max_epi16(q0, _mm_srli_si128(q0, 2));
threshold = (short)_mm_cvtsi128_si32(q0) - 1;
#else
    //a0为阈值
    int a0 = threshold;
    //满足角点条件2时，更新阈值
    for( k = 0; k < 16; k += 2 )
    {
        //a为d[k+1]，d[k+2]和d[k+3]中的最小值
        int a = std::min((int)d[k+1], (int)d[k+2]);
        a = std::min(a, (int)d[k+3]);
        //如果a小于阈值，则进行下一次循环
        if( a <= a0 )
            continue;
        //更新阈值
        //a为从d[k+1]到d[k+8]中的最小值
        a = std::min(a, (int)d[k+4]);
        a = std::min(a, (int)d[k+5]);
        a = std::min(a, (int)d[k+6]);
        a = std::min(a, (int)d[k+7]);
        a = std::min(a, (int)d[k+8]);
        //从d[k]到d[k+9]中的最小值与a0比较，哪个大，哪个作为新的阈值
        a0 = std::max(a0, std::min(a, (int)d[k]));
        a0 = std::max(a0, std::min(a, (int)d[k+9]));
    }
    //满足角点条件1时，更新阈值
    int b0 = -a0;
    for( k = 0; k < 16; k += 2 )
    {
        int b = std::max((int)d[k+1], (int)d[k+2]);
        b = std::max(b, (int)d[k+3]);
        b = std::max(b, (int)d[k+4]);
        b = std::max(b, (int)d[k+5]);
        if( b >= b0 )
            continue;
        b = std::max(b, (int)d[k+6]);
        b = std::max(b, (int)d[k+7]);
        b = std::max(b, (int)d[k+8]);

        b0 = std::min(b0, std::max(b, (int)d[k]));
        b0 = std::min(b0, std::max(b, (int)d[k+9]));
    }

    threshold = -b0-1;
#endif

#if VERIFY_CORNERS
    testCorner(ptr, pixel, K, N, threshold);
#endif
    //更新后的阈值作为输出
    return threshold;
}

int cornerScore_12(const unsigned char* ptr, const int pixel[], int threshold)
{
    const int K = 6, N = K*3 + 1;
    int k, v = ptr[0];
    short d[N + 4];
    for( k = 0; k < N; k++ )
        d[k] = (short)(v - ptr[pixel[k]]);
#if CV_SIMD128
    for( k = 0; k < 4; k++ )
    d[N+k] = d[k];
#endif

#if CV_SIMD128
    if (hasSIMD128())
{
    v_int16x8 q0 = v_setall_s16(-1000), q1 = v_setall_s16(1000);
    for (k = 0; k < 16; k += 8)
    {
        v_int16x8 v0 = v_load(d + k + 1);
        v_int16x8 v1 = v_load(d + k + 2);
        v_int16x8 a = v_min(v0, v1);
        v_int16x8 b = v_max(v0, v1);
        v0 = v_load(d + k + 3);
        a = v_min(a, v0);
        b = v_max(b, v0);
        v0 = v_load(d + k + 4);
        a = v_min(a, v0);
        b = v_max(b, v0);
        v0 = v_load(d + k + 5);
        a = v_min(a, v0);
        b = v_max(b, v0);
        v0 = v_load(d + k + 6);
        a = v_min(a, v0);
        b = v_max(b, v0);
        v0 = v_load(d + k);
        q0 = v_max(q0, v_min(a, v0));
        q1 = v_min(q1, v_max(b, v0));
        v0 = v_load(d + k + 7);
        q0 = v_max(q0, v_min(a, v0));
        q1 = v_min(q1, v_max(b, v0));
    }
    q0 = v_max(q0, v_setzero_s16() - q1);
    threshold = v_reduce_max(q0) - 1;
}
else
#endif
    {
        int a0 = threshold;
        for( k = 0; k < 12; k += 2 )
        {
            int a = std::min((int)d[k+1], (int)d[k+2]);
            if( a <= a0 )
                continue;
            a = std::min(a, (int)d[k+3]);
            a = std::min(a, (int)d[k+4]);
            a = std::min(a, (int)d[k+5]);
            a = std::min(a, (int)d[k+6]);
            a0 = std::max(a0, std::min(a, (int)d[k]));
            a0 = std::max(a0, std::min(a, (int)d[k+7]));
        }

        int b0 = -a0;
        for( k = 0; k < 12; k += 2 )
        {
            int b = std::max((int)d[k+1], (int)d[k+2]);
            b = std::max(b, (int)d[k+3]);
            b = std::max(b, (int)d[k+4]);
            if( b >= b0 )
                continue;
            b = std::max(b, (int)d[k+5]);
            b = std::max(b, (int)d[k+6]);

            b0 = std::min(b0, std::max(b, (int)d[k]));
            b0 = std::min(b0, std::max(b, (int)d[k+7]));
        }

        threshold = -b0-1;
    }
#if VERIFY_CORNERS
    testCorner(ptr, pixel, K, N, threshold);
#endif
    return threshold;
}

int cornerScore_8(const unsigned char* ptr, const int pixel[], int threshold)
{
    const int K = 4, N = K * 3 + 1;
    int k, v = ptr[0];
    short d[N];
    for (k = 0; k < N; k++)
        d[k] = (short)(v - ptr[pixel[k]]);

#if CV_SIMD128
    if (hasSIMD128())
{
    v_int16x8 v0 = v_load(d + 1);
    v_int16x8 v1 = v_load(d + 2);
    v_int16x8 a = v_min(v0, v1);
    v_int16x8 b = v_max(v0, v1);
    v0 = v_load(d + 3);
    a = v_min(a, v0);
    b = v_max(b, v0);
    v0 = v_load(d + 4);
    a = v_min(a, v0);
    b = v_max(b, v0);
    v0 = v_load(d);
    v_int16x8 q0 = v_min(a, v0);
    v_int16x8 q1 = v_max(b, v0);
    v0 = v_load(d + 5);
    q0 = v_max(q0, v_min(a, v0));
    q1 = v_min(q1, v_max(b, v0));
    q0 = v_max(q0, v_setzero_s16() - q1);
    threshold = v_reduce_max(q0) - 1;
}
else
#endif
    {
        int a0 = threshold;
        for( k = 0; k < 8; k += 2 )
        {
            int a = std::min((int)d[k+1], (int)d[k+2]);
            if( a <= a0 )
                continue;
            a = std::min(a, (int)d[k+3]);
            a = std::min(a, (int)d[k+4]);
            a0 = std::max(a0, std::min(a, (int)d[k]));
            a0 = std::max(a0, std::min(a, (int)d[k+5]));
        }

        int b0 = -a0;
        for( k = 0; k < 8; k += 2 )
        {
            int b = std::max((int)d[k+1], (int)d[k+2]);
            b = std::max(b, (int)d[k+3]);
            if( b >= b0 )
                continue;
            b = std::max(b, (int)d[k+4]);

            b0 = std::min(b0, std::max(b, (int)d[k]));
            b0 = std::min(b0, std::max(b, (int)d[k+5]));
        }

        threshold = -b0-1;
    }

#if VERIFY_CORNERS
    testCorner(ptr, pixel, K, N, threshold);
#endif
    return threshold;
}
template <int patternSize>
int cornerScore(const unsigned char* ptr, const int pixel[], int threshold)
{
    int res=0;
    if(patternSize==16)
        res = cornerScore_16(ptr,pixel,threshold);
    else if(patternSize==12)
        res = cornerScore_12(ptr,pixel,threshold);
    else if(patternSize==8)
        res = cornerScore_8(ptr,pixel,threshold);
    else
        throw("pattern size error \n");
    return res;
}
void makeOffsets(int pixel[25], int rowStride, int patternSize)
{
    static const int offsets16[][2] =
            {
                    {0,  3}, { 1,  3}, { 2,  2}, { 3,  1}, { 3, 0}, { 3, -1}, { 2, -2}, { 1, -3},
                    {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3,  1}, {-2,  2}, {-1,  3}
            };

    static const int offsets12[][2] =
            {
                    {0,  2}, { 1,  2}, { 2,  1}, { 2, 0}, { 2, -1}, { 1, -2},
                    {0, -2}, {-1, -2}, {-2, -1}, {-2, 0}, {-2,  1}, {-1,  2}
            };

    static const int offsets8[][2] =
            {
                    {0,  1}, { 1,  1}, { 1, 0}, { 1, -1},
                    {0, -1}, {-1, -1}, {-1, 0}, {-1,  1}
            };
    //根据patternSize值，得到具体应用上面定义的哪个数组
    const int (*offsets)[2] = patternSize == 16 ? offsets16 :
                              patternSize == 12 ? offsets12 :
                              patternSize == 8  ? offsets8  : 0;

    //代入输入图像每行的像素个数，得到圆周像素的绝对坐标位置
    int k = 0;
    for( ; k < patternSize; k++ )
        pixel[k] = offsets[k][0] + offsets[k][1] * rowStride;
    //由于要计算连续的像素，因此要循环的多列出一些值
    for( ; k < 25; k++ )
        pixel[k] = pixel[k - patternSize];
}

template<typename _Tp> static inline _Tp* alignPtr(_Tp* ptr, int n=(int)sizeof(_Tp)) {
    return (_Tp*)(((size_t)ptr + n-1) & -n);
}
template<int patternSize>
void FAST_t(UCInputArray& img, std::vector<KeyPoint>& keypoints, int threshold, bool nonmax_suppression)
{
    //K为圆周连续像素的个数
    //N用于循环圆周的像素点，因为要首尾连接，所以N要比实际圆周像素数量多K+1个
    const int K = patternSize/2, N = patternSize + K + 1;
    int i, j, k, pixel[25];
    //找到圆周像素点相对于圆心的偏移量
    makeOffsets(pixel, (int)img.cols(), patternSize);

#if CV_SIMD128
    const int quarterPatternSize = patternSize/4;
v_uint8x16 delta = v_setall_u8(0x80), t = v_setall_u8((char)threshold), K16 = v_setall_u8((char)K);
bool hasSimd = hasSIMD128();
#if CV_TRY_AVX2
Ptr<opt_AVX2::FAST_t_patternSize16_AVX2> fast_t_impl_avx2;
if(CV_CPU_HAS_SUPPORT_AVX2)
    fast_t_impl_avx2 = opt_AVX2::FAST_t_patternSize16_AVX2::getImpl(img.cols, threshold, nonmax_suppression, pixel);
#endif

#endif

    keypoints.clear();

    threshold = std::min(std::max(threshold, 0), 255);
    // threshold_tab为阈值列表，在进行阈值比较的时候，只需查该表即可
    unsigned char threshold_tab[512];
    /*为阈值列表赋值，该表分为三段：
     * 第一段从threshold_tab[0]至threshold_tab[255 - threshold]，值为1，落在该区域的值表示满足角点判断条件2；
     * 第二段从threshold_tab[255 – threshold]至threshold_tab[255 + threshold]，值为0，落在该区域的值表示不是角点；
     * 第三段从threshold_tab[255 + threshold]至threshold_tab[511]，值为2，落在该区域的值表示满足角点判断条件1*/
    for( i = -255; i <= 255; i++ )
        threshold_tab[i+255] = (unsigned char)(i < -threshold ? 1 : i > threshold ? 2 : 0);
/*buf[0、buf[1]和buf[2]分别表示图像的前一行、当前行和后一行。因为在非极大值抑制的步骤2中，是要在3×3的角点邻域内进行比较，因此需要三行的图像数据。因为只有得到了当前行的数据，所以对于上一行来说，才凑够了连续三行的数据，因此输出的非极大值抑制的结果是上一行数据的处理结果*/
    unsigned char* _buf=new unsigned char[(img.cols()+16)*3*(sizeof(int) + sizeof(unsigned char)) + 128];
    unsigned char* buf[3];
    buf[0] = _buf; buf[1] = buf[0] + img.cols(); buf[2] = buf[1] + img.cols();
    //cpbuf存储角点的坐标位置，也是需要连续三行的数据
    int* cpbuf[3];
    cpbuf[0] = (int*)alignPtr(buf[2] + img.cols(), sizeof(int)) + 1;
    cpbuf[1] = cpbuf[0] + img.cols() + 1;
    cpbuf[2] = cpbuf[1] + img.cols() + 1;
    memset(buf[0], 0, img.cols()*3);

    for(i = 3; i < img.rows()-2; i++)
    {
        const unsigned char* ptr = img.ptr(i) + 3;
        unsigned char* curr = buf[(i - 3)%3];
        int* cornerpos = cpbuf[(i - 3)%3];
        memset(curr, 0, img.cols());
        int ncorners = 0;

        if( i < img.rows() - 3 )
        {

            for( j = 3; j < img.cols() - 3; j++, ptr++ )
            {
                //当前像素的灰度值
                int v = ptr[0];
                //由当前像素的灰度值，确定其在阈值列表中的位置
                const unsigned char* tab = &threshold_tab[0] - v + 255;
                //pixel[0]表示圆周上编号为0的像素相对于圆心坐标的偏移量
                //ptr[pixel[0]表示圆周上编号为0的像素值
                //tab[ptr[pixel[0]]]表示相对于当前像素（即圆心）圆周上编号为0的像素值在阈值列表threshold_tab中所查询得到的值，如果为1，说明I0 < Ip - t，如果为2，说明I0 > Ip + t，如果为0，说明 Ip – t < I0 < Ip + t。因此通过tab，就可以得到当前像素是否满足角点条件。
                //编号为0和8（即直径在圆周上的两个像素点）在列表中的值相或后得到d。d=0说明编号为0和8的值都是0；d=1说明编号为0和8的值至少有一个为1，而另一个不能为2；d=2说明编号为0和8的值至少有一个为2，而另一个不能为1；d=3说明编号为0和8的值有一个为1，另一个为2。只可能有这四种情况。
                int d = tab[ptr[pixel[0]]] | tab[ptr[pixel[8]]];
                //d=0说明圆周上不可能有连续12个像素满足角点条件，因此当前值一定不是角点，所以退出此次循环，进入下一次循环
                if( d == 0 )
                    continue;
                //继续进行其他直径上两个像素点的判断
                d &= tab[ptr[pixel[2]]] | tab[ptr[pixel[10]]];
                d &= tab[ptr[pixel[4]]] | tab[ptr[pixel[12]]];
                d &= tab[ptr[pixel[6]]] | tab[ptr[pixel[14]]];
                //d=0说明上述d中至少有一个d为0，所以肯定不是角点；另一种情况是一个d为2，而另一个d为1，相与后也为0，这说明一个是满足角点条件1，而另一个满足角点条件2，所以肯定也不会有连续12个像素满足同一个角点条件的，因此也一定不是角点。
                if( d == 0 )
                    continue;
                //继续判断圆周上剩余的像素点
                d &= tab[ptr[pixel[1]]] | tab[ptr[pixel[9]]];
                d &= tab[ptr[pixel[3]]] | tab[ptr[pixel[11]]];
                d &= tab[ptr[pixel[5]]] | tab[ptr[pixel[13]]];
                d &= tab[ptr[pixel[7]]] | tab[ptr[pixel[15]]];
                //如果满足if条件，则说明有可能满足角点条件2
                if( d & 1 )
                {
                    //vt为真正的角点条件，即Ip – t，count为连续像素的计数值
                    int vt = v - threshold, count = 0;
                    //遍历整个圆周
                    for( k = 0; k < N; k++ )
                    {
                        int x = ptr[pixel[k]];    //提取出圆周上的像素值
                        if(x < vt)    //如果满足条件2
                        {
                            //连续计数，并判断是否大于K（K为圆周像素的一半）
                            if( ++count > K )
                            {
                                //进入该if语句，说明已经得到一个角点
                                //保存该点的位置，并把当前行的角点数加1
                                cornerpos[ncorners++] = j;
                                //进行非极大值抑制的第一步，计算得分函数
                                if(nonmax_suppression)
                                    curr[j] = (unsigned char)cornerScore<patternSize>(ptr, pixel, threshold);
                                break;    //退出循环
                            }
                        }
                        else
                            count = 0;    //连续像素的计数值清零
                    }
                }
                //如果满足if条件，则说明有可能满足角点条件1
                if( d & 2 )
                {
                    //vt为真正的角点条件，即Ip + t，count为连续像素的计数值
                    int vt = v + threshold, count = 0;
                    //遍历整个圆周
                    for( k = 0; k < N; k++ )
                    {
                        int x = ptr[pixel[k]];    //提取出圆周上的像素值
                        if(x > vt)    //如果满足条件1
                        {
                            //连续计数，并判断是否大于K（K为圆周像素的一半）
                            if( ++count > K )
                            {
                                //进入该if语句，说明已经得到一个角点
                                //保存该点的位置，并把当前行的角点数加1
                                cornerpos[ncorners++] = j;
                                //进行非极大值抑制的第一步，计算得分函数
                                if(nonmax_suppression)
                                    curr[j] = (unsigned char)cornerScore<patternSize>(ptr, pixel, threshold);
                                break;    //退出循环
                            }
                        }
                        else
                            count = 0;    //连续像素的计数值清零
                    }
                }
            }
        }
        //保存当前行所检测到的角点数
        cornerpos[-1] = ncorners;
        //i=3说明只仅仅计算了一行的数据，还不能进行非极大值抑制的第二步，所以不进行下面代码的操作，直接进入下一次循环
        if( i == 3 )
            continue;
        //以下代码是进行非极大值抑制的第二步，即在3×3的角点邻域内对得分函数的值进行非极大值抑制。因为经过上面代码的计算，已经得到了当前行的数据，所以可以进行上一行的非极大值抑制。因此下面的代码进行的是上一行的非极大值抑制。
        //提取出上一行和上两行的图像像素
        const unsigned char* prev = buf[(i - 4 + 3)%3];
        const unsigned char* pprev = buf[(i - 5 + 3)%3];
        //提取出上一行所检测到的角点位置
        cornerpos = cpbuf[(i - 4 + 3)%3];
        //提取出上一行的角点数
        ncorners = cornerpos[-1];
        //在上一行内遍历整个检测到的角点
        for( k = 0; k < ncorners; k++ )
        {
            j = cornerpos[k];    //得到角点的位置
            int score = prev[j];    //得到该角点的得分函数值
            //在3×3的角点邻域内，计算当前角点是否为最大值，如果是则压入特性值向量中
            if( !nonmax_suppression ||
                (score > prev[j+1] && score > prev[j-1] &&
                 score > pprev[j-1] && score > pprev[j] && score > pprev[j+1] &&
                 score > curr[j-1] && score > curr[j] && score > curr[j+1]) )
            {
                keypoints.push_back(KeyPoint((float)j, (float)(i-1), 7.f, -1, (float)score));
            }
        }
    }
    delete[] _buf;
}

void FAST(UCInputArray _img, std::vector<KeyPoint>& keypoints, int threshold, bool nonmax_suppression, int type)
{

    switch(type) {
        case FastFeatureDetector::TYPE_5_8:
            FAST_t<8>(_img, keypoints, threshold, nonmax_suppression);
            break;
        case FastFeatureDetector::TYPE_7_12:
            FAST_t<12>(_img, keypoints, threshold, nonmax_suppression);
            break;
        case FastFeatureDetector::TYPE_9_16:
            FAST_t<16>(_img, keypoints, threshold, nonmax_suppression);
            break;
    }
}

class FastFeatureDetector_Impl : public FastFeatureDetector
{
    public:
    FastFeatureDetector_Impl( int _threshold, bool _nonmaxSuppression, int _type )
            : threshold(_threshold), nonmaxSuppression(_nonmaxSuppression), type((short)_type)
    {}

    void detect( UCInputArray& gray, std::vector<KeyPoint>& keypoints, UCInputArray& mask ) override
    {
        if(gray.empty()||gray.channels()!=1)
        {
            keypoints.clear();
            return;
        }


        FAST( gray, keypoints, threshold, nonmaxSuppression, type );
        if(!mask.empty())
            KeyPointsFilter::runByPixelsMask( keypoints, mask );
    }

    void set(int prop, double value)
    {
        if(prop == THRESHOLD)
            threshold = Round(value);
        else if(prop == NONMAX_SUPPRESSION)
            nonmaxSuppression = value != 0;
        else if(prop == FAST_N)
            type = Round(value);
    }

    double get(int prop) const
    {
        if(prop == THRESHOLD)
            return threshold;
        if(prop == NONMAX_SUPPRESSION)
            return nonmaxSuppression;
        if(prop == FAST_N)
            return type;
        return 0;
    }

    void setThreshold(int threshold_) override { threshold = threshold_; }
    int getThreshold() const override { return threshold; }

    void setNonmaxSuppression(bool f) override { nonmaxSuppression = f; }
    bool getNonmaxSuppression() const override { return nonmaxSuppression; }

    void setType(int type_) override { type = type_; }
    int getType() const override { return type; }

    int threshold;
    bool nonmaxSuppression;
    int type;
};

shared_ptr<FastFeatureDetector> FastFeatureDetector::create(int threshold, bool nonmaxSuppression, int type) {
    return make_shared<FastFeatureDetector_Impl>(threshold, nonmaxSuppression, type);
}
}