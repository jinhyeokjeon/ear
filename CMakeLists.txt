cmake_minimum_required(VERSION 3.10)
project(eye_tracker)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Dlib include directory
include_directories(${CMAKE_SOURCE_DIR}/dlib)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Dlib static lib
add_library(dlib STATIC IMPORTED)
set_target_properties(dlib PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/dlib/build/dlib/libdlib.a
)

# Main executable
add_executable(eye_tracker main.cpp)
target_link_libraries(eye_tracker dlib ${OpenCV_LIBS})