CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
PROJECT(MSER_Detector)

IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-strict-aliasing")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wno-write-strings")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -ansi")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -funroll-loops")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

set(mser_srcs
        extrema/extrema.cpp
        extrema/sortPixels.cpp
        extrema/getExtrema.cpp
        extrema/libExtrema.cpp
        extrema/boundary.cpp
        extrema/suballoc.cpp
        extrema/optThresh.cpp
        extrema/preprocess.cpp
        LL/LLconsis.c
        LL/LLfile.c
        LL/LLio.c
        LL/LL.c
        LL/LLstr.c
        LL/LLmergeSort.c
        utls/matrix.cpp utls/timeutls.c
        )

include_directories(.
./extrema
./utls
./LL)

add_library(mser ${mser_srcs})
TARGET_LINK_LIBRARIES(mser ${OpenCV_LIBS})

