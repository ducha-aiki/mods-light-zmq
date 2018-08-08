cmake_minimum_required(VERSION 2.8)

PROJECT(MODS_NEW)

add_definitions(-DA64)
add_definitions(-std=c++11)

FIND_PACKAGE(OpenMP)
if (OPENMP_FOUND)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif (OPENMP_FOUND)

message("Looking for OpenCV")
FIND_PACKAGE(OpenCV 2.4.9 REQUIRED )
if (OpenCV_FOUND)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenCV_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenCV_CXX_FLAGS}")
  include_directories(${OpenCV_INDLUDE_DIRS})
else(OpenCV_FOUND)
  message("OpenCV not found!")
endif (OpenCV_FOUND)

enable_language (Fortran)
FIND_PACKAGE(LAPACK REQUIRED)
if (LAPACK_FOUND)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LAPACK_CXX_FLAGS}")
endif (LAPACK_FOUND)

SET(CMAKE_BUILD_TYPE "RELEASE")

IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-strict-aliasing")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wno-write-strings")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -ansi")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -funroll-loops")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)


add_subdirectory( ./inih)
add_subdirectory( ./matutls)
add_subdirectory( ./detectors)
add_subdirectory( ./detectors/affinedetectors)
add_subdirectory( ./detectors/mser )
add_subdirectory( ./libMatch)
add_subdirectory( ./libNumerics)

####### DEGENSAC
include_directories(
.
./degensac)
set(degensac_srcs
        degensac/DegUtils.c
        degensac/exp_ranF.c
        degensac/exp_ranH.c
        degensac/Ftools.c
        degensac/hash.c
        degensac/Htools.c
        degensac/ranF.c
        degensac/ranH2el.c
        degensac/ranH.c
        degensac/rtools.c
        degensac/utools.c
        degensac/lapwrap.c
)
add_library(degensac ${degensac_srcs})
TARGET_LINK_LIBRARIES(degensac matutls ${LAPACK_LIBRARIES})
##### End of DEGENSAC

###################Matching
include_directories(.
./matching
 )

set(matching_srcs
        matching/siftdesc.cpp
        matching/matching.cpp
        orsa.cpp
)
add_library(matching ${matching_srcs})
TARGET_LINK_LIBRARIES(matching ${OpenCV_LIBS} degensac ${LAPACK_LIBRARIES} Match Numerics)

################## View synthesis

set(synth-detection_srcs
        synth-detection.cpp)

include_directories(.)

add_library(synth-detection ${synth-detection_srcs})

TARGET_LINK_LIBRARIES(synth-detection matching mser affinedetectors helpers ${OpenCV_LIBS})


add_executable(mods  mods.cpp correspondencebank.cpp imagerepresentation.cpp io_mods.cpp)
TARGET_LINK_LIBRARIES(mods inih matching synth-detection mser affinedetectors ${OpenCV_LIBS} zmq)

add_executable(mods-with-mask  mods-with-mask.cpp correspondencebank.cpp imagerepresentation.cpp io_mods.cpp)
TARGET_LINK_LIBRARIES(mods-with-mask inih matching synth-detection mser affinedetectors ${OpenCV_LIBS} zmq)


add_executable(extract_features extract_features.cpp correspondencebank.cpp imagerepresentation.cpp io_mods.cpp)
TARGET_LINK_LIBRARIES(extract_features inih matching synth-detection mser affinedetectors ${OpenCV_LIBS} zmq)

add_executable(extract_features_batch extract_features_batch.cpp correspondencebank.cpp imagerepresentation.cpp io_mods.cpp)
TARGET_LINK_LIBRARIES(extract_features_batch inih matching synth-detection mser affinedetectors ${OpenCV_LIBS} zmq)
