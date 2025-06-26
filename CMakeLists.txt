cmake_minimum_required(VERSION 3.10)
project(main)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# include directory
include_directories(${CMAKE_SOURCE_DIR}/include)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Dlib
include_directories(${CMAKE_SOURCE_DIR}/dlib)
add_library(dlib STATIC IMPORTED)
set_target_properties(dlib PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/dlib/build/dlib/libdlib.a
)

# Main executable
add_executable(main
 src/main.cpp
 src/functions.cpp
 )

target_link_libraries(main ${OpenCV_LIBS} dlib)