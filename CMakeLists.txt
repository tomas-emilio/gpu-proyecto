cmake_minimum_required(VERSION 3.18)
project(SimulacionTejido LANGUAGES CXX CUDA)

#estandares
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#archivos de headers
include_directories(${PROJECT_SOURCE_DIR}/include)

#cuda
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

#agrega subdirectorio con los fuentes
add_subdirectory(src)
