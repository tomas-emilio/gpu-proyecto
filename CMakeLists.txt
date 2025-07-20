cmake_minimum_required(VERSION 3.18)
project(SimulacionTejido LANGUAGES CXX CUDA)

# Estándares
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Archivos de headers
include_directories(${PROJECT_SOURCE_DIR}/include)

# Agrega subdirectorio con los fuentes
add_subdirectory(src)
