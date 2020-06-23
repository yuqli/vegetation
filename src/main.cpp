#include "tiffio.h"
#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;


int main() 
{ 
    
    /*
    Mat mat = imread("../S2A_3Band_Cropped_int16_b1.tif", IMREAD_ANYDEPTH);  
    // Mat mat = imread("../dat.tif", IMREAD_ANYDEPTH | IMREAD_COLOR);  
    // Mat mat = imread("../samples/test.tif", IMREAD_COLOR | IMREAD_ANYDEPTH);  
    int rows = mat.rows;
    int cols = mat.cols;
    std::cout << "rows " << rows << " cols " << cols << " channels " << mat.channels() << std::endl;

    if(! mat.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    // namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    // imshow( "Display window", mat);                   // Show our image inside it.
    // waitKey(0);  

    // std::cout << mat.at<cv::Vec3i>(0, 0).val[0] << endl;
    std::cout << mat.at<int>(0, 0) << endl;
    */


    TIFF* tif = TIFFOpen("../dat.tif", "r");
    if (tif) {
        uint32 w, h;
        size_t npixels;
        uint32* raster;

        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);

        std::cout << "w " << w << " h " << h << std::endl;

        npixels = w * h;
        raster = (uint32*) _TIFFmalloc(npixels * sizeof (uint32));

        if (raster != NULL) {
            if (TIFFReadRGBAImage(tif, w, h, raster, 0)) {
            // ...process raster data...
            }
            _TIFFfree(raster);
        }
        TIFFClose(tif);
    }

    return 0; 
} 

