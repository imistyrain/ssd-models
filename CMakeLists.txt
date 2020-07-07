cmake_minimum_required(VERSION 2.8)
set(ProjName ssd-models)
project(${ProjName})
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Caffe)
if(Caffe_LIBRARIES)
    include_directories(${Caffe_INCLUDE_DIRS})
    add_executable(fd_caffe cpp/ssd_caffe.cpp)
    target_link_libraries(fd_caffe ${OpenCV_LIBS} ${Caffe_LIBRARIES})
else()
	message("Caffe not found, ssd_caffe will not be built")
endif()

add_executable(fd_opencv cpp/ssd_opencv.cpp)
target_link_libraries(fd_opencv ${OpenCV_LIBS} ${Caffe_LIBRARIES})

add_executable(ssd cpp/main.cpp cpp/SSDDetector.cpp)
target_link_libraries(ssd ${OpenCV_LIBS} ${Caffe_LIBRARIES})

find_package(ncnn)
if(ncnn_FOUND)
    add_executable(demo ncnn/demo.cpp)
    target_link_libraries(demo ncnn ${OpenCV_LIBS})
endif()