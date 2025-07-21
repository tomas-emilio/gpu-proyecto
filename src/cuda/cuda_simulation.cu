#include "cuda_simulation.h"
#include "kernels.cuh"
#include <iostream>
#include <vector>
#include <cmath>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

CUDASimulation::CUDASimulation() 
    : d_positions(nullptr), d_oldPositions(nullptr), d_forces(nullptr), 
      d_springs(nullptr), numSprings(0), lastFrameTime(0.0) {
    
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
}

CUDASimulation::~CUDASimulation() {
    cleanup();
}

void CUDASimulation::initialize(int meshWidth, int meshHeight) {
    width = meshWidth;
    height = meshHeight;
    numVertices = width * height;
    spacing = 0.1f;
    
    // Alocar memoria en GPU
    CUDA_CHECK(cudaMalloc(&d_positions, numVertices * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_oldPositions, numVertices * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_forces, numVertices * sizeof(float3)));
    
    // Inicializar posiciones en host y copiar a device
    std::vector<float3> hostPositions(numVertices);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            hostPositions[idx] = make_float3(x * spacing, y * spacing, 0.0f);
        }
    }
    
    CUDA_CHECK(cudaMemcpy(d_positions, hostPositions.data(), 
                         numVertices * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_oldPositions, hostPositions.data(), 
                         numVertices * sizeof(float3), cudaMemcpyHostToDevice));
    
    generateSprings();
    
    std::cout << "CUDA Simulation initialized: " << width << "x" << height 
              << " (" << numVertices << " vertices, " << numSprings << " springs)" << std::endl;
}

void CUDASimulation::generateSprings() {
    std::vector<CudaSpring> hostSprings;
    extern TissueParams g_tissueParams;
    
    // Resortes estructurales
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int currentVertex = y * width + x;
            
            if (x < width - 1) {
                int rightVertex = y * width + (x + 1);
                CudaSpring spring = {currentVertex, rightVertex, spacing, g_tissueParams.structuralStiffness, 0};
                hostSprings.push_back(spring);
            }
            
            if (y < height - 1) {
                int bottomVertex = (y + 1) * width + x;
                CudaSpring spring = {currentVertex, bottomVertex, spacing, g_tissueParams.structuralStiffness, 0};
                hostSprings.push_back(spring);
            }
        }
    }
    
    // Resortes de corte
    for (int y = 0; y < height - 1; ++y) {
        for (int x = 0; x < width - 1; ++x) {
            int currentVertex = y * width + x;
            int diagVertex = (y + 1) * width + (x + 1);
            float diagLength = spacing * sqrtf(2.0f);
            
            CudaSpring spring1 = {currentVertex, diagVertex, diagLength, g_tissueParams.shearStiffness, 1};
            hostSprings.push_back(spring1);
            
            int rightVertex = y * width + (x + 1);
            int bottomLeftVertex = (y + 1) * width + x;
            CudaSpring spring2 = {rightVertex, bottomLeftVertex, diagLength, g_tissueParams.shearStiffness, 1};
            hostSprings.push_back(spring2);
        }
    }
    
    // Resortes virtuales
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int currentVertex = y * width + x;
            
            if (x < width - 2) {
                int farRightVertex = y * width + (x + 2);
                CudaSpring spring = {currentVertex, farRightVertex, spacing * 2.0f, g_tissueParams.virtualStiffness, 2};
                hostSprings.push_back(spring);
            }
            
            if (y < height - 2) {
                int farBottomVertex = (y + 2) * width + x;
                CudaSpring spring = {currentVertex, farBottomVertex, spacing * 2.0f, g_tissueParams.virtualStiffness, 2};
                hostSprings.push_back(spring);
            }
        }
    }
    
    numSprings = hostSprings.size();
    
    // Liberar memoria anterior si existe
    if (d_springs) {
        cudaFree(d_springs);
    }
    
    // Copiar nuevos resortes a GPU
    CUDA_CHECK(cudaMalloc(&d_springs, numSprings * sizeof(CudaSpring)));
    CUDA_CHECK(cudaMemcpy(d_springs, hostSprings.data(), 
                         numSprings * sizeof(CudaSpring), cudaMemcpyHostToDevice));
}

void CUDASimulation::update(float deltaTime) {
    CUDA_CHECK(cudaEventRecord(startEvent, stream));
    
    const int blockSize = 256;
    const int gridSizeVertices = (numVertices + blockSize - 1) / blockSize;
    const int gridSizeSprings = (numSprings + blockSize - 1) / blockSize;
    const int gridSizeConstraints = (numVertices + blockSize - 1) / blockSize;
    
    // Limpiar fuerzas
    clearForces<<<gridSizeVertices, blockSize, 0, stream>>>(d_forces, numVertices);
    
    // Calcular fuerzas de resortes
    calculateForces<<<gridSizeSprings, blockSize, 0, stream>>>(
        d_positions, d_forces, d_springs, numSprings);
    
    // Integración de Verlet
    verletIntegration<<<gridSizeVertices, blockSize, 0, stream>>>(
        d_positions, d_oldPositions, d_forces, deltaTime, numVertices, 1.0f);
    
    // Aplicar restricciones
    applyConstraints<<<gridSizeConstraints, blockSize, 0, stream>>>(
        d_positions, width, height);
    
    CUDA_CHECK(cudaEventRecord(stopEvent, stream));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    
    // Calcular tiempo
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));
    lastFrameTime = milliseconds;
}

void CUDASimulation::handleMouseInteraction(float x, float y, float force) {
    int meshX = static_cast<int>(x / spacing);
    int meshY = static_cast<int>(y / spacing);
    
    if (meshX >= 0 && meshX < width && meshY >= 0 && meshY < height) {
        int targetVertex = meshY * width + meshX;
        float3 forceVector = make_float3(0.0f, 0.0f, force);
        
        applyUserForce<<<1, 1, 0, stream>>>(d_positions, targetVertex, forceVector);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

void CUDASimulation::getMesh(Mesh& mesh) {
    std::vector<float3> hostPositions(numVertices);
    CUDA_CHECK(cudaMemcpy(hostPositions.data(), d_positions, 
                         numVertices * sizeof(float3), cudaMemcpyDeviceToHost));
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            float3 pos = hostPositions[idx];
            mesh.setVertex(x, y, glm::vec3(pos.x, pos.y, pos.z));
        }
    }
}

double CUDASimulation::getLastFrameTime() const {
    return lastFrameTime;
}

void CUDASimulation::cleanup() {
    if (d_positions) {
        cudaFree(d_positions);
        d_positions = nullptr;
    }
    if (d_oldPositions) {
        cudaFree(d_oldPositions);
        d_oldPositions = nullptr;
    }
    if (d_forces) {
        cudaFree(d_forces);
        d_forces = nullptr;
    }
    if (d_springs) {
        cudaFree(d_springs);
        d_springs = nullptr;
    }
    
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaStreamDestroy(stream);
}

void CUDASimulation::reset() {
    // Reinicializar posiciones
    std::vector<float3> hostPositions(numVertices);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            hostPositions[idx] = make_float3(x * spacing, y * spacing, 0.0f);
        }
    }
    
    CUDA_CHECK(cudaMemcpy(d_positions, hostPositions.data(), 
                         numVertices * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_oldPositions, hostPositions.data(), 
                         numVertices * sizeof(float3), cudaMemcpyHostToDevice));
}

void CUDASimulation::updateParams(const TissueParams& params) {
    // Regenerar resortes con nuevos parámetros
    generateSprings();
}