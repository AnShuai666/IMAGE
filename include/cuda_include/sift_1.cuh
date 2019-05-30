/**
 * @功能    sift.hpp文件实现
 * @姓名    杨丰拓
 * @日期    2019-05-20
 * @时间    15:40
*/
#ifndef _SIFT_1_CUH_
#define _SIFT_1_CUH_


void  extrema_detection_cu(int w,int h,int *noff,int octave_index,int sample_index,const float *s0,
                           const float *s1,const float *s2);

#endif //_SIFT_1_CUH_