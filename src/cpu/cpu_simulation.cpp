#include "cpu_simulation.h"
#include <iostream>

void CPUSimulation::initialize(int meshWidth, int meshHeight) {
    float spacing = 0.1f;
    
    mesh.initialize(meshWidth, meshHeight, spacing);
    physics.generateSprings(mesh);
    integrator.updatePreviousPositions(mesh);
    
    lastTime = std::chrono::high_resolution_clock::now();
    lastFrameTime = 0.0;
    
    std::cout << "CPU Simulation initialized: " << meshWidth << "x" << meshHeight 
              << " (" << (meshWidth * meshHeight) << " vertices)" << std::endl;
}

void CPUSimulation::update(float deltaTime) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    integrator.integrate(mesh, physics, deltaTime);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    lastFrameTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
}

void CPUSimulation::handleMouseInteraction(float x, float y, float force) {
    int meshX = static_cast<int>(x / mesh.getSpacing());
    int meshY = static_cast<int>(y / mesh.getSpacing());
    
    glm::vec3 forceVector(0.0f, 0.0f, force);
    mesh.applyForce(meshX, meshY, forceVector);
}

const Mesh& CPUSimulation::getMesh() const {
    return mesh;
}

double CPUSimulation::getLastFrameTime() const {
    return lastFrameTime;
}