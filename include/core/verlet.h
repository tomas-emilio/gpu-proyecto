#pragma once
#include "mesh.h"
#include "physics.h"

class VerletIntegrator {
private:
    float mass;
    
public:
    VerletIntegrator() : mass(1.0f) {}
    
    void integrate(Mesh& mesh, PhysicsModel& physics, float deltaTime);
    void updatePreviousPositions(Mesh& mesh);
    
    void setMass(float m) { mass = m; }
};