cmake_minimum_required(VERSION 3.24)
project(gsplat C CXX)

if (NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "/usr/bin/clang++")
endif()
set(CMAKE_CUDA_ARCHITECTURES all)
enable_language(CUDA)

#--------------------------------------------
#               Main target
#--------------------------------------------
set(EXE_NAME gsplat_raytracer)
add_executable(${EXE_NAME})

#--------------------------------------------
#           Target sources
#--------------------------------------------
file(GLOB_RECURSE SRC_FILES
    "${CMAKE_SOURCE_DIR}/src/*.cu"
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/*.c"
)

target_sources(${EXE_NAME}
    PRIVATE
    ${SRC_FILES}
)

#--------------------------------------------
# Main target include and link directories
#--------------------------------------------

target_include_directories(${EXE_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(${EXE_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

#----------------------------------------
#           Target properies
#----------------------------------------

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set_target_properties(${EXE_NAME} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES all
        COMPILE_FLAGS "-xcuda --cuda-path=/usr/local/cuda"
    )
else()
    set_target_properties(${EXE_NAME} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES all
    )
endif()

#----------------------------------------
#           libraries
#----------------------------------------
