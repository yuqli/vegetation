#include <iostream>
#include <cuda.h>
#include <opencv2/core.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <fstream>
#include <string>
#include <chrono>



using namespace cv;

__global__ 
void vegetationKernel(float *in, float *out, int row, int col, int channel);

__global__ 
void detectionKernel(float *in, int8_t *out, int row, int col);

// Get pixel offset in a 2D matrix at position (i, j)
// row-major memory layout
__host__ __device__
int offset2D(int i, int j, int col) {
    return i * col + j;
}

// Get pixel offset in a 3D matrix at position (i, j, k)
// row-major memory layout, but channel is the innermost loop increment unit 
__host__ __device__
int offset3D(int i, int j, int k, int col, int channel) {
    return (i * col + j) * channel + k; 
}


/************************************************************************************************************** 
 *                                                 C++ version 
 *************************************************************************************************************/

// Calculate vegetation index
void getVegetationIndex(float *img, float *out, int row, int col, int channel) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            float b2 = img[offset3D(i, j, 1, col, channel)];
            float b3 = img[offset3D(i, j, 2, col, channel)];

            if (b3 + b2 == 0) {
                out[offset2D(i, j, col)] = 0;
            } 
            else {
                out[offset2D(i, j, col)] = (b3 - b2) / (b3 + b2);
            }
        }
    }

    return;
}


// Calculate vegetation detection 
void getVegetationDetection(float *veg, int8_t *out, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            out[offset2D(i, j, col)] = veg[offset2D(i, j, col)] > 0.1 ? (int8_t)1 : (int8_t)0;
        }
    }

    return;
} 


/************************************************************************************************************** 
 *                                                 CUDA version 
 *************************************************************************************************************/

void getVegetationIndexCUDA(float *h_img, float *h_veg, int row, int col, int channel) {
    int img_size = row * col * channel * sizeof(float);
    int out_size = row * col * sizeof(float);

    float *d_img, *d_out;

    cudaError_t err1 =  cudaMalloc((void **) &d_img, img_size);
    cudaError_t err2 =  cudaMalloc((void **) &d_out, out_size);

    if (err1 != cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err1), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    if (err2 != cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err2), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(d_img, h_img, img_size, cudaMemcpyHostToDevice);

    dim3 blocksPerGrid(ceil(row/32.0), ceil(col/32.0), 1);   
    dim3 threadsPerBlock(32, 32, 1); // 1024 threads per block

    vegetationKernel<<<blocksPerGrid, threadsPerBlock>>>(d_img, d_out, row, col, channel);

    cudaMemcpy(h_veg, d_out, out_size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();

    if(error!=cudaSuccess)
    {
        fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
        exit(-1);
    }

    cudaFree(d_img); 
    cudaFree(d_out);
}


void getVegetationDetectionCUDA(float *h_veg, int8_t *h_out, int row, int col) {
    int veg_size = row * col * sizeof(float);
    int out_size = row * col * sizeof(int8_t);
    float *d_veg;
    int8_t *d_out;

    cudaError_t err1 =  cudaMalloc((void **) &d_veg, veg_size);
    cudaError_t err2 =  cudaMalloc((void **) &d_out, out_size);

    if (err1 != cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err1), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    if (err2 != cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err2), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(d_veg, h_veg, veg_size, cudaMemcpyHostToDevice);

    dim3 blocksPerGrid(ceil(row/32.0), ceil(col/32.0), 1);   
    dim3 threadsPerBlock(32, 32, 1); // 1024 threads per block

    detectionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_veg, d_out, row, col);

    cudaMemcpy(h_out, d_out, out_size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();

    if(error!=cudaSuccess)
    {
        fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
        exit(-1);
    }

    cudaFree(d_veg); 
    cudaFree(d_out);
} 


/************************************************************************************************************** 
 *                                                 Driver code 
 *************************************************************************************************************/
void printImg(float *d, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            std::cout << d[offset2D(i, j, col)] << " ";
        }
        std::cout << std::endl;
    }
}


void printImg(int8_t *d, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            std::cout << (int)d[offset2D(i, j, col)] << " ";
        }
        std::cout << std::endl;
    }
}

void printCube(float *d, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            std::cout << "( " << d[offset3D(i, j, 0, col, 3)] << " " << d[offset3D(i, j, 1, col, 3)] << " " << d[offset3D(i, j, 2, col, 3)] << " )"; 
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

int main() 
{ 
    // Image metadata is hard coded for now, but can be passed as arguments to main
    int height = 6058;
    int width = 3320; 
    int channel= 3;

    std::cout << "width " << width << " height " << height << std::endl;

    // Read from tmp .txt file
    int img_size = width * height * channel;
    float *dat = new float[img_size]();

    std::ifstream file("../tmp.txt");
    std::string str; 

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < channel; k++) {
                std::getline(file, str);
                dat[offset3D(i, j, k, width, channel)] = std::atof(str.c_str());
            }
        }
    }

    // // verify correct

    // for (int i = 0; i < 1; i++) {
    //     for (int j = 0; j < 1; j++) {
    //         for (int k = 0; k < channel; k++) {
    //             std::cout << dat[offset3D(i, j, k, width, channel)] << " ";
    //         }
    //     }
    // }


    // C++ version
    int out_size = height * width;

    std::cout << "C++ ======================================\n";
    float *veg_cpp = new float[out_size]();
    int8_t *det_cpp = new int8_t[out_size]();
    
    // time veg in cpp
    auto t1 = std::chrono::high_resolution_clock::now();

    getVegetationIndex(dat, veg_cpp, height, width, channel);

    auto t2 = std::chrono::high_resolution_clock::now();

    auto veg_cpp_duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

    // time det in cpp
    t1 = std::chrono::high_resolution_clock::now();

    getVegetationDetection(veg_cpp, det_cpp, height, width);

    t2 = std::chrono::high_resolution_clock::now();

    auto det_cpp_duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

    // verify results in cpp
    std::cout << "Veg time : cpp " << veg_cpp_duration << std::endl;
    std::cout << "Det time : cpp " << det_cpp_duration << std::endl;

    std::cout << "Veg results : cpp " << std::endl;
    printImg(veg_cpp, 5, 5);
    std::cout << "Det results : cpp " << std::endl;
    printImg(det_cpp, 5, 5);

    // CUDA version
    std::cout << "CUDA ======================================\n";

    float *veg_cuda = new float[out_size]();
    int8_t *det_cuda = new int8_t[out_size]();

    // time veg in cuda 
    t1 = std::chrono::high_resolution_clock::now();

    getVegetationIndexCUDA(dat, veg_cuda, height, width, channel);

    t2 = std::chrono::high_resolution_clock::now();

    auto veg_cuda_duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

    // time det in cuda 
    t1 = std::chrono::high_resolution_clock::now();

    getVegetationDetectionCUDA(veg_cuda, det_cuda, height, width);

    t2 = std::chrono::high_resolution_clock::now();

    auto det_cuda_duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

    // verify results in cuda 
    std::cout << "Veg time : cuda " << veg_cuda_duration << std::endl;
    std::cout << "Det time : cuda " << det_cuda_duration << std::endl;

    std::cout << "Veg results : cuda" << std::endl;
    printImg(veg_cuda, 5, 5);
    std::cout << "Det results : cuda" << std::endl;
    printImg(det_cuda, 5, 5);

    // save results
    std::cout << "Write results to image ...\n";
    cv::imwrite("../veg_index.bmp",  cv::Mat(height, width, CV_32FC1, veg_cpp));
    cv::imwrite("../veg_detection.tif",  cv::Mat(height, width, CV_8UC1, det_cpp));

    return 0; 
} 



/************************************************************************************************************** 
 *                                                 CUDA kernel 
 *************************************************************************************************************/

__global__ 
void vegetationKernel(float *in, float *out, int row, int col, int channel){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    if (i < row && j < col) {
        float a = (in[offset3D(i, j, 2, col, channel)] - in[offset3D(i, j, 1, col, channel)] ) ;
        float b = (in[offset3D(i, j, 2, col, channel)] + in[offset3D(i, j, 1, col, channel)] ) ;

        if (b != 0) {
            out[offset2D(i, j, col)] = a / b;
        }
        else {
            out[offset2D(i, j, col)] = 0;
        } 
    } 
}


__global__ 
void detectionKernel(float *in, int8_t *out, int row, int col){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    if (i < row && j < col) {
        out[offset2D(i, j, col)] = in[offset2D(i, j, col)] > 0.1 ? 1 : 0;
    } 
}