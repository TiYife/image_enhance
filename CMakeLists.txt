cmake_minimum_required(VERSION 3.12)
project(image_enhance)

set(CMAKE_CXX_STANDARD 14)

add_executable(image_enhance main.cpp MSDB.h CLAHE_DWT.h)

find_package(OpenCV)
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries(image_enhance ${OpenCV_LIBS} )

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(image_enhance ${ARMADILLO_LIBRARIES})