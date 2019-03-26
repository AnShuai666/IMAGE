#include "include/define.h"

void checkfile(FILE *fp, char const* const filename, int const linenum)
{
    if(fp == NULL)
    {
        std::cerr<<"FILE LOAD ERROR AT: "<<filename<<": "<<linenum<<std::endl;
    }
    return;
}