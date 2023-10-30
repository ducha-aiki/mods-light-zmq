CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

PROJECT(libNumerics)

INCLUDE_DIRECTORIES(..)

ADD_LIBRARY(Numerics
        computeH.cpp
        homography.cpp homography.h
        matrix.cpp matrix.h
        numerics.cpp numerics.h
        rodrigues.cpp rodrigues.h
        vector.cpp)
