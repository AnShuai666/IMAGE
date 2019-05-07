/*
 * @desc    最近邻算法
 * @author  安帅
 * @date    2019-04-02
 * @e-mail   1028792866@qq.com
*/

#ifndef IMAGE_NEAREST_NEIGHBOR_HPP
#define IMAGE_NEAREST_NEIGHBOR_HPP

#include "define.h"
#include <iostream>
#include <emmintrin.h>  //SSE2 加速整型
#include <pmmintrin.h>  //SSE3 加速浮点型
IMAGE_NAMESPACE_BEGIN

/********************************************************************
*~~~~~~~~~~~~~~~~~~~~~~~~NearestNeighbor类的声明~~~~~~~~~~~~~~~~~~~~~~
********************************************************************/
/*
 * @property    NearestNeighbor类
 * @func        对归一化向量进行最近邻搜索,求一个向量与另一个向量序列中最近的向量,
 *              取距离平方最小者为最近邻,第二小者为次近邻.
*/
template <typename T>
class NearestNeighbor
{
public:
    /*
     * @property    搜索结果,最近距离平方
     * @func        存储最近邻/次近邻距离与索引值
    */
    struct Result
    {
        T dist_1st_best;
        T dist_2nd_best;
        int index_1st_best;
        int index_2nd_best;
    };

public:
    NearestNeighbor(void);

    void set_elements(T const* elements);

    void set_element_dimentions(int element_dimentions);

    void set_num_elements(int num_elements);

    void find(T const* query, Result* result) const;

    int get_element_dimentions(void) const;

    void inner_prod(T const* query, typename NearestNeighbor<T>::Result* result,T const* elements, int num_elements,int dimensions);

    //void float_inner_prod()

private:
    int dimentions;
    int num_elements;
    T const* elements;
};

/********************************************************************
*~~~~~~~~~~~~~~~~~~~~~~~~NearestNeighbor类的实现~~~~~~~~~~~~~~~~~~~~~~
********************************************************************/

template <typename T>
inline NearestNeighbor<T>::NearestNeighbor(void):dimentions(64),num_elements(0),elements(NULL)
{

}

template <typename T>
inline void
NearestNeighbor<T>::set_elements(T const *elements)
{

}

template <typename T>
inline void NearestNeighbor<T>::set_element_dimentions(int element_dimentions)
{

}

template <typename T>
inline void NearestNeighbor<T>::set_num_elements(int num_elements)
{

}

template <typename T>
inline int
NearestNeighbor<T>::get_element_dimentions() const
{

}

template <>
void
NearestNeighbor<short>::find(short const *query, image::NearestNeighbor<short>::Result *result) const
{

}

template <>
void
NearestNeighbor<unsigned short>::find(unsigned short const *query, image::NearestNeighbor<unsigned short>::Result *result) const
{

}

template <>
void
NearestNeighbor<float>::find(float const *query, image::NearestNeighbor<float>::Result *result) const
{

}

template <typename T>
void NearestNeighbor<T>::inner_prod(T const *query, image::NearestNeighbor<T>::Result *result, T const *elements, int num_elements, int dimensions)
{

}

IMAGE_NAMESPACE_END

#endif //IMAGE_NEAREST_NEIGHBOR_HPP
