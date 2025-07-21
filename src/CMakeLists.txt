#archivos fuente
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
    opengl/renderer.cpp
    opengl/shader_simulation.cpp
    opengl/utils.cpp
)

set(APPLICATION_SRC
    application.cpp
)

#ejecutable
add_executable(simulacion
    ${CORE_SOURCES}
    ${CPU_SOURCES}
    ${CUDA_SOURCES}
    ${OPENGL_SOURCES}
    ${APPLICATION_SRC}
    main.cpp
)

#propiedades cuda
set_target_properties(simulacion PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "native"
)

#includes
target_include_directories(simulacion PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/core
    ${PROJECT_SOURCE_DIR}/include/cpu
    ${PROJECT_SOURCE_DIR}/include/cuda
    ${PROJECT_SOURCE_DIR}/include/opengl
)

#linkeo opengl, glew y glfw
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

target_link_libraries(simulacion
    OpenGL::GL
    GLEW::GLEW
    glfw
)

#configuracion de shaders
file(GLOB SHADERS "${PROJECT_SOURCE_DIR}/shaders/*.glsl")
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/shaders)
foreach(shader ${SHADERS})
    get_filename_component(shader_name ${shader} NAME)
    configure_file(${shader} ${CMAKE_BINARY_DIR}/shaders/${shader_name} COPYONLY)
endforeach()