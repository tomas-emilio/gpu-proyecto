#pragma once
#include "../core/mesh.h"
#include "../core/physics.h"
#include "../core/verlet.h"
#include "../core/tissue_params.h"
#include <chrono>

class CPUSimulation {
private:
    Mesh mesh;
    PhysicsModel physics;
    VerletIntegrator integrator;
    
    std::chrono::high_resolution_clock::time_point lastTime;
    double lastFrameTime;
    
public:
    void initialize(int meshWidth, int meshHeight);
    void update(float deltaTime);
    void handleMouseInteraction(float x, float y, float force);
    
    const Mesh& getMesh() const;
    double getLastFrameTime() const;
    void reset();
    void updateParams(const TissueParams& params);
};