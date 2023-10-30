CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
PROJECT(inih)

set(inih_srcs
        cpp/INIReader.cpp
        ini.c)
include_directories(.)
add_library(inih ${inih_srcs})
