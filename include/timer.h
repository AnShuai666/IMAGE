/*
 * @desc    时间定义
 * @author  安帅
 * @date    2019-03-12
 * @email   1028792866@qq.com
*/
#ifndef IMAGE_TIMER_H
#define IMAGE_TIMER_H

#include <chrono>
#include "define.h"
IMAGE_NAMESPACE_BEGIN
/********************************************************************
 *~~~~~~~~~~~~~~~~~~~~~~~TimerHigh类的声明~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *******************************************************************/
class TimerHigh
{
/********************************************************************
*~~~~~~~~~~~~~~~~~~~~~常用向量类型别名声明~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*******************************************************************/
private:
    typedef std::chrono::high_resolution_clock TimerClock;
    typedef std::chrono::time_point<TimerClock> TimerTimePoint;
    typedef std::chrono::duration<double ,std::milli> TimerDurationMs;

    //初始化初始时间点
    TimerTimePoint start;

public:
    /*
    *  @property   默认构造函数
    *  @func       对start进行初始化
    */
    TimerHigh(void);

    /*
    *  @property   重置函数
    *  @func       对start进行初始化
    */
    void reset(void);

    /*
    *  @property   获取时间段
    *  @func       获取声明TimerHigh处到此刻的运行时间，单位为：毫秒
    *  @return     std::size_t      返回毫秒数
    */
    std::size_t get_elapsed(void) const;

    /*
    *  @property   获取时间段
    *  @func       获取声明TimerHigh处到此刻的运行时间，单位为：秒
    *  @return     std::size_t      返回秒数
    */
    float get_elapsed_sec(void) const;
};

/********************************************************************
 *~~~~~~~~~~~~~~~~~~~~~~~TimerLess类的声明~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *******************************************************************/
class TimerLess
{
private:
    //初始化初始时间点
    std::size_t start;

public:
    /*
    *  @property   默认构造函数
    *  @func       对start进行初始化
    */
    TimerLess(void);

    /*
    *  @property   重置函数
    *  @func       对start进行初始化
    */
    void reset(void);

    /*
    *  @property   获取时间段
    *  @func       获取此刻的时间戳，单位为：秒
    *  @return     std::size_t       返回时间戳：秒数
    */
    static float now_sec(void);

    /*
    *  @property   获取时间戳
    *  @func       获取此刻的时间戳，单位为：毫秒
    *  @return     std::size_t      返回时间戳：毫秒数
    */
    static std::size_t now(void);

    /*
    *  @property   获取时间段
    *  @func       获取声明TimerLess处到此刻的运行时间，单位为：毫秒
    *  @return     std::size_t      返回毫秒数
    */
    std::size_t get_elapsed(void) const;

    /*
    *  @property   获取时间段
    *  @func       获取声明TimerLess处到此刻的运行时间，单位为：秒
    *  @return     float      返回秒数
    */
    float get_elapsed_sec(void) const;
};

/********************************************************************
 *~~~~~~~~~~~~~~~~~~~~~~~TimerHigh类的实现~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *******************************************************************/
inline
TimerHigh::TimerHigh()
{
    this->reset();
}

inline void
TimerHigh::reset()
{
    this->start = TimerClock::now();
}

inline std::size_t
TimerHigh::get_elapsed() const
{
    TimerDurationMs interval = TimerClock::now() - this->start;
    return interval.count();
}

inline float
TimerHigh::get_elapsed_sec() const
{
    return static_cast<float>(this->get_elapsed()) / 1000.0f;
}

/********************************************************************
 *~~~~~~~~~~~~~~~~~~~~~~~TimerLess类的实现~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *******************************************************************/

inline
TimerLess::TimerLess()
{
    this->reset();
}

inline void
TimerLess::reset()
{
    this->start = TimerLess::now();
}

inline float
TimerLess::now_sec()
{
    return (float)std::clock() / (float)CLOCKS_PER_SEC;
}

inline std::size_t
TimerLess::now()
{
    return ((std::size_t)(std::clock()) * 1000) / (std::size_t)CLOCKS_PER_SEC;
}

inline std::size_t
TimerLess::get_elapsed() const
{
    return TimerLess::now() - this->start;
}

inline float
TimerLess::get_elapsed_sec() const
{
    return (1.0f / 1000.0f) * (float)this->get_elapsed();
}

IMAGE_NAMESPACE_END
#endif //IMAGE_TIMER_H
