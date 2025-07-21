#pragma once
#include "../core/mesh.h"
#include "../core/tissue_params.h"
#include <cuda_runtime.h>
#include <chrono>

struct CudaSpring {
    int vertex1, vertex2;
    float restLength;
    float stiffness;
    int type; //0=estructural, 1=shear, 2=virtual
};

class CUDASimulation {
private:
    //host data
    int width, height, numVertices;
    float spacing;
    
    //device pointers
    float3* d_positions;
    float3* d_oldPositions;
    float3* d_forces;
    CudaSpring* d_springs;
    int numSprings;
    
    //timing
    std::chrono::high_resolution_clock::time_point lastTime;
    double lastFrameTime;
    
    //cuda streams y eventos
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
    void reset();
    void updateParams(const TissueParams& params);
};