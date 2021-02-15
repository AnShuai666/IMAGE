//
// Created by doing on 19-5-31.
//
#define BOOST_AUTO_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iostream>
#include "IMAGE/image.hpp"

using namespace image;



BOOST_AUTO_TEST_SUITE(image_class_test)
    BOOST_AUTO_TEST_CASE(ImageSize)
    {
        Image<float> image_test(10,11,3);
        int w=image_test.width();
        int h=image_test.height();
        int c=image_test.channels();
        BOOST_CHECK( w == 10);
        BOOST_CHECK( h == 11);
        BOOST_CHECK( c == 3);
        image_test.resize(20,21,2);
        w=image_test.width();
        h=image_test.height();
        c=image_test.channels();
        BOOST_CHECK( w == 20);
        BOOST_CHECK( h == 21);
        BOOST_CHECK( c == 2);
        int num=image_test.getPixelAmount();
        BOOST_CHECK(num=w*h);
        num=image_test.getValueAmount();
        BOOST_CHECK(num==w*h*c);
        image_test.release();
        w=image_test.width();
        h=image_test.height();
        c=image_test.channels();
        BOOST_CHECK( w == 0);
        BOOST_CHECK( h == 0);
        BOOST_CHECK( c == 0);
    }
    BOOST_AUTO_TEST_CASE(fillColor)
    {
        Image<float> image_test(10,10);
        float color[1]={1};
        image_test.fill(color[0]);
        float val=image_test.at(0);
        BOOST_CHECK(val==1.0);
        color[0]=2.f;
        image_test.fillColor(color,1);
        val=image_test.at(0);
        BOOST_CHECK(val==2.f);
    }
    BOOST_AUTO_TEST_CASE(channels)
    {
        Image<float> image_test(10,10);
        image_test.addChannels(1,1);
        int c=image_test.channels();
        BOOST_CHECK(c==2);
        image_test.swapChannels(1,0);
        float c0=image_test.at(0,0);
        float c1=image_test.at(0,1);
        BOOST_CHECK(c0==1.f);
        BOOST_CHECK(c1==0.f);
        image_test.copyChannel(1,0);
        c0=image_test.at(0,0);
        c1=image_test.at(0,1);
        BOOST_CHECK(c0==c1);
        image_test.deleteChannel(1);
        c=image_test.channels();
        BOOST_CHECK(c==1);
    }

BOOST_AUTO_TEST_SUITE_END()