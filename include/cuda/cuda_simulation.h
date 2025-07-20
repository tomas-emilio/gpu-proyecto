#pragma once
#include "../core/mesh.h"
#include <cuda_runtime.h>
#include <chrono>

struct CudaSpring {
    int vertex1, vertex2;
    float restLength;
    float stiffness;
    int type; // 0=STRUCTURAL, 1=SHEAR, 2=VIRTUAL
};

class CUDASimulation {
private:
    // Host data
    int width, height, numVertices;
    float spacing;
    
    // Device pointers
    float3* d_positions;
    float3* d_oldPositions;
    float3* d_forces;
    CudaSpring* d_springs;
    int numSprings;
    
    // Timing
    std::chrono::high_resolution_clock::time_point lastTime;
    double lastFrameTime;
    
    // CUDA streams and events
    cudaStream_t stream;
    cudaEvent_t startEvent, stopEvent;
    
    void generateSprings();
    void copyToDevice(const Mesh& mesh);
    void copyFromDevice(Mesh& mesh);
    
public:
    CUDASimulation();
    ~CUDASimulation();
    
    void initialize(int meshWidth, int meshHeight);
    void update(float deltaTime);
    void handleMouseInteraction(float x, float y, float force);
    void getMesh(Mesh& mesh);
    
    double getLastFrameTime() const;
    void cleanup();
};