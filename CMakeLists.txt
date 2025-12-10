cmake_minimum_required(VERSION 2.8.3)
project(stereo_visual_odometry)

## Compile as C++11
#add_compile_options(-std=c++14)

find_package(OpenCV REQUIRED)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME}
        src/tdv_lab.cpp
)

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})


