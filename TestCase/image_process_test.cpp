//
// Created by doing on 19-5-31.
//

#define BOOST_AUTO_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iostream>
#include "IMAGE/image.hpp"
#include "IMAGE/image_process.hpp"
using namespace image;


BOOST_AUTO_TEST_SUITE(image_process)
    BOOST_AUTO_TEST_CASE(resize)
    {
        Image<float> image_test(100,100);
        Image<float> image_dst;
        imageResize(image_test,image_dst,50,50);
        int w=image_dst.width();
        int h=image_dst.height();
        BOOST_CHECK(w==50);
        BOOST_CHECK(h==50);
    }
    BOOST_AUTO_TEST_CASE(threshold)
    {
        Image<float> image_test(100,100);
        Image<float> image_dst;
        image_test.fill(10);
        imageThreshold(image_test,image_test,5,20,THRESH_BINARY);
        float v=image_test.at(0);
        BOOST_CHECK(v==20);
        imageThreshold(image_test,image_test,50,20,THRESH_BINARY);
        v=image_test.at(0);
        BOOST_CHECK(v==0);
        image_test.fill(10);
        imageThreshold(image_test,image_test,50,20,THRESH_BINARY_INV);
        v=image_test.at(0);
        BOOST_CHECK(v==20);
        imageThreshold(image_test,image_test,5,20,THRESH_BINARY_INV);
        v=image_test.at(0);
        BOOST_CHECK(v==0);
    }
BOOST_AUTO_TEST_SUITE_END()
