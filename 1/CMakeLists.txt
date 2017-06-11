cmake_minimum_required( VERSION 2.8 )
project( svo )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )


find_package( OpenCV 3.1 REQUIRED )
# find_package( OpenCV REQUIRED ) # use this if in OpenCV2

include_directories(${OpenCV_INCLUDE_DIRS}
                    "/usr/include/eigen3/")

file(GLOB viso
    "src/*.h"
    "src/*.cpp"
)
add_executable( svo ${viso} )
target_link_libraries( svo ${OpenCV_LIBS} )
