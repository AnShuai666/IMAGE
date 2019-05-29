/*
 * @desc    测试image_io
 * @author  安帅
 * @date    2019-5-27
 * @e-mail   1028792866@qq.com
*/
#define BOOST_AUTO_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iostream>
#include "IMAGE/image_io.h"

BOOST_AUTO_TEST_SUITE(image_io_test)
BOOST_AUTO_TEST_CASE(example)
{
    std::string str = "12345";
    BOOST_CHECK( str == "12345");
}


BOOST_AUTO_TEST_SUITE_END()