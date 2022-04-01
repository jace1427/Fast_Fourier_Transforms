#ifndef IMAGELOADER_H
#define IMAGELOADER_H
#include <string>
#include <complex>
#include <vector>
#include <omp.h>
#include "../include/lodepng.h"

using namespace std;

class ImageLoader
{
    public:
        ImageLoader(const char* path);
        void grayscaler();
        void doubleVector();
        void doubleVectorConvert(vector<complex<double>> input);
        const char* path;
        std::vector<unsigned char> image;
        std::vector<double> imageDoubles;
        unsigned width, height;
    private:
};

#endif