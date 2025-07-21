#include "verlet.h"

void VerletIntegrator::integrate(Mesh& mesh, PhysicsModel& physics, float deltaTime) {
    int width = mesh.getWidth();
    int height = mesh.getHeight();
    float dt2 = deltaTime * deltaTime;
    
    //limpiar fuerzas y calcular nuevas fuerzas
    physics.clearForces(width * height);
    physics.calculateForces(mesh);
    
    //integracion de Verlet: x(t+dt) = 2*x(t) - x(t-dt) + a(t)*dt^2
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int vertexId = y * width + x;
            
            glm::vec3 currentPos = mesh.getVertex(x, y);
            glm::vec3 oldPos = mesh.getOldVertex(x, y);
            glm::vec3 force = physics.getForce(vertexId);
            
            //calcular aceleracion
            glm::vec3 acceleration = force / mass;
            
            //integracion de verlet
            glm::vec3 newPos = 2.0f * currentPos - oldPos + acceleration * dt2;
            
            //actualizar posiciones
            mesh.setOldVertex(x, y, currentPos);
            mesh.setVertex(x, y, newPos);
        }
    }
    
    //aplicar restricciones despues de la integracion
    physics.applyConstraints(mesh);
}

void VerletIntegrator::updatePreviousPositions(Mesh& mesh) {
    int width = mesh.getWidth();
    int height = mesh.getHeight();
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            mesh.setOldVertex(x, y, mesh.getVertex(x, y));
        }
    }
}