cmake_minimum_required(VERSION 3.12)
project(CameraCalibration)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(camera_calibration calibrate.cpp)

target_link_libraries(camera_calibration ${OpenCV_LIBS})
