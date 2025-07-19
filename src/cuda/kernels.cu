// Kernel para cálculo de fuerzas de resortes
__global__ void calculateForces(float3* positions, float3* forces, 
                               Spring* springs, int numSprings);

// Kernel para integración Verlet
__global__ void verletIntegration(float3* positions, float3* oldPositions,
                                 float3* forces, float deltaTime, int numVertices);

// Kernel para aplicar restricciones
__global__ void applyConstraints(float3* positions, int width, int height);

// Kernel para interacción del usuario
__global__ void applyUserForce(float3* positions, float3* forces,
                              int targetVertex, float3 force);