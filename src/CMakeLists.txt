cmake_minimum_required(VERSION 3.9)
project(main LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 11)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(TIFF)
include_directories( ${TIFF_INCLUDE_DIRS})

add_executable(main main.cu)
target_link_libraries( main ${OpenCV_LIBS} ${TIFF_LIBRARIES})


