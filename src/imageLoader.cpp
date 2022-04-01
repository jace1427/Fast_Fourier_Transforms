#include "../include/imageLoader.h"
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>
#include <complex>
#include "../include/lodepng.h"

using namespace std;

ImageLoader::ImageLoader(const char* path)
{
    cout << "constructor called for image loader" << endl;
    cout << "loading image with path " << path << endl;

    vector<uint8_t> img;
    unsigned w, h;

    unsigned error = lodepng::decode(img, w, h, path);

    if (error) cout << "decoding error" << error << ": " << lodepng_error_text(error) << endl;
    else
    {
        image = img;
        vector<double> tmp(img.begin(), img.end());
        imageDoubles = tmp;
        width = w;
        height = h;
        cout << "Success" << endl;
        //for (int i = 0; i < image.size(); i+=4)
        //{
        //    cout << "Pixel " << i << " is : " 
        //    << "r:" << (int)image[i] << " " 
        //    << "g:" << (int)image[i + 1] << " "
        //    << "b:" << (int)image[i + 2] << " "
        //    << "a:" << (int)image[i + 3] << " "
        //    << endl;
        //}
    }
}

void ImageLoader::grayscaler()
{
    cout << "attempting to encode grayscale image" << endl;
    //0.02031 - Serial
    // - Best parallel
    //0.02106 - Best SIMD
    //#pragma omp parallel for schedule(static, 1024)
    #pragma omp simd
    for (int i = 0; i < image.size(); i+=4)
    {
        unsigned int gray = image[i] + image[i + 1] + image[i + 2];
        gray = gray / 3;
        image[i] = gray;
        image[i + 1] = gray;
        image[i + 2] = gray;
        //cout << "Pixel " << i << " is : " 
        //<< "r:" << (int)image[i] << " " 
        //<< "g:" << (int)image[i + 1] << " "
        //<< "b:" << (int)image[i + 2] << " "
        //<< "a:" << (int)image[i + 3] << " "
        //<< endl;
    }
    char* filename = "test.png";

    unsigned error = lodepng::encode(filename, image, width, height);

    //if there's an error, display it
    if(error) cout << "encoder error " << error << ": "<< lodepng_error_text(error) << endl;

    cout << "grayscale encoding success" << endl;
}

void ImageLoader::doubleVector()
{
    //copy the grayscale vector to double vector for dft
    vector<double> tmp(image.begin(), image.end());

    imageDoubles = tmp;
}

void ImageLoader::doubleVectorConvert(vector<complex<double>> input)
{
    //convert the vector back into an image
    char* filename = "doubleTest.png";

    //init vector for real part of input
    vector<uint8_t> real;

    //vector for real part of complex vector
    for (int i = 0; i < input.size(); i++) {
        real.push_back(ceil(input[i].real()));
        //cout << (int)real[i] << endl;
    }
    
    //convert datatype for image saving
    //vector<uint8_t> real(tmp.begin(), tmp.end());
    
    unsigned error = lodepng::encode(filename, real, width, height);
    
    //if there's an error, display it
    if(error) cout << "encoder error " << error << ": "<< lodepng_error_text(error) << endl;
}