CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
PROJECT(helpers)

message("Looking for OpenCV")
FIND_PACKAGE(OpenCV REQUIRED )
if (OpenCV_FOUND)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenCV_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenCV_CXX_FLAGS}")
  include_directories(${OpenCV_INCLUDE_DIRS})
else(OpenCV_FOUND)
  message("OpenCV not found!")
endif (OpenCV_FOUND)

IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-strict-aliasing")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wno-write-strings")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -ansi")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -funroll-loops")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)
include_directories(.)

add_library(helpers helpers.cpp)

