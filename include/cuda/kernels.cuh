#pragma once
#include <cuda_runtime.h>
#include "cuda_simulation.h"

// Declaraciones de kernels
__global__ void clearForces(float3* forces, int numVertices);

__global__ void calculateForces(float3* positions, float3* forces, 
                               CudaSpring* springs, int numSprings);

__global__ void verletIntegration(float3* positions, float3* oldPositions,
                                 float3* forces, float deltaTime, int numVertices,
                                 float mass);

__global__ void applyConstraints(float3* positions, int width, int height);

__global__ void applyUserForce(float3* positions, int targetVertex, float3 force);