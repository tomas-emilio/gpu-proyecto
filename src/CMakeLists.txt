# Archivos fuente
set(CORE_SOURCES
    core/mesh.cpp
    core/physics.cpp
    core/verlet.cpp
)

set(CPU_SOURCES
    cpu/cpu_simulation.cpp
)

# Ejecutable principal
add_executable(simulacion
    ${CORE_SOURCES}
    ${CPU_SOURCES}
    main.cpp
)

# Incluye directorios para los headers
target_include_directories(simulacion PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/core
    ${PROJECT_SOURCE_DIR}/include/cpu
)
