CMAKE_MINIMUM_REQUIRED(VERSION 3.9)
PROJECT(DEGENSAC C)

add_subdirectory( ./matutls)

include_directories(.)
set(degensac_srcs
        DegUtils.c
        exp_ranF.c
        exp_ranH.c
        Ftools.c
        hash.c
        Htools.c
        ranF.c
        ranH2el.c
        ranH.c
        rtools.c
        utools.c
        lapwrap.c
        
)
add_library(degensac  ${degensac_srcs})