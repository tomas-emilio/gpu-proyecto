#include "kernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Kernel para limpiar fuerzas
__global__ void clearForces(float3* forces, int numVertices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVertices) {
        forces[idx] = make_float3(0.0f, 0.0f, 0.0f);
    }
}

// Kernel para calcular fuerzas de resortes
__global__ void calculateForces(float3* positions, float3* forces, 
                               CudaSpring* springs, int numSprings) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numSprings) {
        CudaSpring spring = springs[idx];
        
        float3 pos1 = positions[spring.vertex1];
        float3 pos2 = positions[spring.vertex2];
        
        float3 diff = make_float3(pos2.x - pos1.x, pos2.y - pos1.y, pos2.z - pos1.z);
        float currentLength = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
        
        if (currentLength > 0.0f) {
            float displacement = currentLength - spring.restLength;
            float3 direction = make_float3(diff.x / currentLength, 
                                         diff.y / currentLength, 
                                         diff.z / currentLength);
            
            float3 springForce = make_float3(direction.x * displacement * spring.stiffness,
                                           direction.y * displacement * spring.stiffness,
                                           direction.z * displacement * spring.stiffness);
            
            // Usar atomic operations para evitar race conditions
            atomicAdd(&forces[spring.vertex1].x, springForce.x);
            atomicAdd(&forces[spring.vertex1].y, springForce.y);
            atomicAdd(&forces[spring.vertex1].z, springForce.z);
            
            atomicAdd(&forces[spring.vertex2].x, -springForce.x);
            atomicAdd(&forces[spring.vertex2].y, -springForce.y);
            atomicAdd(&forces[spring.vertex2].z, -springForce.z);
        }
    }
}

// Kernel para integración Verlet
__global__ void verletIntegration(float3* positions, float3* oldPositions,
                                 float3* forces, float deltaTime, int numVertices,
                                 float mass) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numVertices) {
        float3 currentPos = positions[idx];
        float3 oldPos = oldPositions[idx];
        float3 force = forces[idx];
        
        // Calcular aceleración
        float3 acceleration = make_float3(force.x / mass, force.y / mass, force.z / mass);
        
        float dt2 = deltaTime * deltaTime;
        
        // Integración de Verlet: x(t+dt) = 2*x(t) - x(t-dt) + a(t)*dt^2
        float3 newPos = make_float3(
            2.0f * currentPos.x - oldPos.x + acceleration.x * dt2,
            2.0f * currentPos.y - oldPos.y + acceleration.y * dt2,
            2.0f * currentPos.z - oldPos.z + acceleration.z * dt2
        );
        
        // Actualizar posiciones
        oldPositions[idx] = currentPos;
        positions[idx] = newPos;
    }
}

// Kernel para aplicar restricciones
__global__ void applyConstraints(float3* positions, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < width * height) {
        int x = idx % width;
        int y = idx / width;
        
        // Fijar bordes superior e inferior
        if (y == 0 || y == height - 1) {
            positions[idx].z = 0.0f;
        }
    }
}

// Kernel para aplicar fuerza del usuario
__global__ void applyUserForce(float3* positions, int targetVertex, float3 force) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && targetVertex >= 0) {
        positions[targetVertex].x += force.x;
        positions[targetVertex].y += force.y;
        positions[targetVertex].z += force.z;
    }
}