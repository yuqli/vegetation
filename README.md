# How to use
1. Clone this director to local workstation
2. Put sample file under root folder, i.e. outside `src` folder. Sample file name must be: "S2A_3Band_Cropped.tif". It's possible to make this program able to read other filenames in the future.
3. Run `run.sh` in `src` folder. It does three steps : 1) read `.tif` file in Python and save as `.txt` 2) compile and build c++ executable `main` and 3) run `main`

This software depends on :
1) scikit-image in Python
2) OpenCV in C++
3) CUDA

# Results

- Performs one run only for each funtion. 
- Time unit: microseconds

Note it's better to perform multiple runs and get average run time for each function when time permitted.

|            |   C++  | CUDA  | Speed up |
| ---------- | ------ | ----- | -------- |
| Vegetation | 187130 | 97878 | 47.70%   |
| Detection  | 138634 | 26794 | 80.67%   |


## Read .tiff issues

The I/O part turns out to be surprisingly tricky. Common image libraries in C++ handle 3-channel RGB .tiff (8-bit) fine, but they complain when trying to read 32-bit or 16-bit multiple channel or single channel .tiffs. 

I tried two libraries in C++:
- [X] TIFF
- [X] OpenCV

**With `TIFF`, the issues are :**

1. The recommended API is [TIFFReadRGBAImage](http://www.libtiff.org/man/TIFFReadRGBAImage.3t.html). However, this API cannot handle 32-bit images. More, when it tries to read non 8-bit images, it will scale pixel values (not in a straightforward way). The resulting pointer is a 32-bit pack of four 8-bit values : R, G, B and A.
2. There's also the `Scanline-based Image I/O` http://www.libtiff.org/libtiff.html but this a) requires knowledge about whether this image is organized as scanline and 2) further research how to unpack scanline results and 3) no random access at specific pixel locations.

In summary I did not have luck with `TIFF` for this particular file.

**With `OpenCV`, the issues are :**

1. flags for `imread` like `IMREAD_ANYDEPTH` or `IMREAD_COLOR` do not work as expected. Frequently they read wrong number of channels, or get the channels right but values wrong. 
2. Even after separating 3-channel 16-bit .tif as three seperate gray scal .tifs, OpenCV still can't get it right with the following code snippet:

```   
Mat mat = imread("../S2A_3Band_Cropped_int16_b1.tif", IMREAD_ANYDEPTH);  
std::cout << mat.at<int>(0, 0) << endl;
```
Conclusion: I see quite a few questions on StackOverflow and OpenCV forum on the same issue, that never gets solved. It's possible that OpenCV's support for this kind of images is somewhat basic. 

- https://answers.opencv.org/question/151798/can-opencv-correctly-read-a-16-bit-signed-tiff-image/
- https://answers.opencv.org/question/19272/unable-to-properly-read-16bit-tiff-image/
- https://answers.opencv.org/question/13115/cannot-view-16-bit-grey-level-images/
- https://stackoverflow.com/questions/22009312/read-16-bit-tif-with-opencv#comment71556342_22009981
- https://stackoverflow.com/questions/44771872/opencv-read-16-bits-tiff-image-in-c
- https://answers.opencv.org/question/105406/how-to-read-10-bit-tiff-satellite-image-in-opencv/

### Conclusion 

Due to time constraint, I decide to move to a hybrid approach : use Python to parse sample image and save as raw bytes. Read again in C++.**


### Recommendations: 

- [ ] Use geo-specific C++ image libraries like GDAL : https://gdal.org/. 
- [ ] Convert 16-bit .tiff to 16-bit .png in Python, or use other image processing softwares like Imagemagick. 

These two approaches have been used by people in the aforementioned questions to solve this I/O issue.

## Save .tiff issues
Similar to read 32-bit .tiff issues above, it's also tricky to save 32-bit .tiff in OpenCV. e.g. 

- https://stackoverflow.com/questions/17735863/opencv-save-cv-32fc1-images
- https://www.researchgate.net/post/how_to_save_an_image_of_type_float32_in_opencv
- https://answers.opencv.org/question/67165/save-floating-point-tiff-images/
- https://stackoverflow.com/questions/23019021/opencv-how-to-save-float-array-as-an-image
- https://stackoverflow.com/questions/23663180/how-can-i-write-float-image-in-opencv

The temporary solution is to save as .bmp files for float32. Save as .png for int8.


