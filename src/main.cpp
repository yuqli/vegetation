
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;


Mat test1(1000, 1000, CV_16U, Scalar(400));
imwrite("test.tiff", test1);
Mat test2 = imread("stam.tiff", CV_LOAD_IMAGE_ANYDEPTH);
cout << test1.depth() << " " << test2.depth() << endl;
cout << test2.at<unsigned short>(0,0) << endl;

int main() 
{ 

    TIFF *tif=TIFFOpen("../dat.tif", "r");

    if (tif) {
        int width, height;
        int channels = 3;

        // #define uint32 unsigned long

        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);           // uint32 width;
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);        // uint32 height;

        std::cout << "width " << width << " height " << height << std::endl;




        int npixels = width * height * channels;

        auto raster=(uint32*) _TIFFmalloc(npixels *sizeof(uint32));

        int x = TIFFReadRGBAImage(tif, width, height, raster, 0);
        std::cout << x << endl;

        // char X=(char )TIFFGetX(raster[i]);  // where X can be the channels R, G, B, and A.

        // for (int k = 0; k < channels; k++) {
        //     std::cout << tif
        // }

        _TIFFfree(raster);

        TIFFClose(tif);

    }




    // write to tiff
      
    return 0; 
} 

