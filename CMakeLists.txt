cmake_minimum_required(VERSION 3.10)
project(SimulacionTejido)

# Habilita C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Incluye carpeta include/
include_directories(${PROJECT_SOURCE_DIR}/include)

# Agrega subdirectorio con código fuente
add_subdirectory(src)