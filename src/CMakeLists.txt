# ========================
# Archivos fuente CPU y Core
# ========================
set(CORE_SOURCES
    core/mesh.cpp
    core/physics.cpp
    core/verlet.cpp
)

set(CPU_SOURCES
    cpu/cpu_simulation.cpp
)

# ========================
# Archivos fuente CUDA
# ========================
set(CUDA_SOURCES
    cuda/cuda_simulation.cu
    cuda/kernels.cu
)

# ========================
# Ejecutable principal
# ========================
add_executable(simulacion
    ${CORE_SOURCES}
    ${CPU_SOURCES}
    ${CUDA_SOURCES}
    main.cpp
)

# ========================
# Configuración de CUDA
# ========================
# Para permitir separadores de host/device
set_target_properties(simulacion PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "native"
)

# ========================
# Headers
# ========================
target_include_directories(simulacion PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/core
    ${PROJECT_SOURCE_DIR}/include/cpu
    ${PROJECT_SOURCE_DIR}/include/cuda
)
