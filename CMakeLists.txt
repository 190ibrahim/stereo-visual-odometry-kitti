cmake_minimum_required(VERSION 2.8.3)
project(stereo_visual_odometry)

## Compile as C++11
#add_compile_options(-std=c++14)

find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${YAML_CPP_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/feature_detector.cpp
    src/pose_estimator.cpp
    src/utils.cpp
    )

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} yaml-cpp)


