# ==================================
# Archivos fuente
# ==================================
set(CORE_SOURCES
    core/mesh.cpp
    core/physics.cpp
    core/verlet.cpp
)

set(CPU_SOURCES
    cpu/cpu_simulation.cpp
)

set(CUDA_SOURCES
    cuda/cuda_simulation.cu
    cuda/kernels.cu
)

set(OPENGL_SOURCES
    opengl/shader_simulation.cpp
    opengl/utils.cpp
)

# ==================================
# Ejecutable
# ==================================
add_executable(simulacion
    ${CORE_SOURCES}
    ${CPU_SOURCES}
    ${CUDA_SOURCES}
    ${OPENGL_SOURCES}
    main.cpp
)

# ==================================
# Configuración CUDA
# ==================================
set_target_properties(simulacion PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "native"
)

# ==================================
# Includes
# ==================================
target_include_directories(simulacion PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/core
    ${PROJECT_SOURCE_DIR}/include/cpu
    ${PROJECT_SOURCE_DIR}/include/cuda
    ${PROJECT_SOURCE_DIR}/include/opengl
)

# ==================================
# Linkeo OpenGL, GLEW y GLFW
# ==================================
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

target_link_libraries(simulacion
    OpenGL::GL
    GLEW::GLEW
    glfw
)

# ==================================
# Configuración de shaders
# ==================================
file(GLOB SHADERS "${PROJECT_SOURCE_DIR}/shaders/*.glsl")
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/shaders)
foreach(shader ${SHADERS})
    get_filename_component(shader_name ${shader} NAME)
    configure_file(${shader} ${CMAKE_BINARY_DIR}/shaders/${shader_name} COPYONLY)
endforeach()