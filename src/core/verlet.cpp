#include "verlet.h"

void VerletIntegrator::integrate(Mesh& mesh, PhysicsModel& physics, float deltaTime) {
    int width = mesh.getWidth();
    int height = mesh.getHeight();
    float dt2 = deltaTime * deltaTime;
    
    // Limpiar fuerzas y calcular nuevas fuerzas
    physics.clearForces(width * height);
    physics.calculateForces(mesh);
    
    // Integración de Verlet: x(t+dt) = 2*x(t) - x(t-dt) + a(t)*dt^2
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int vertexId = y * width + x;
            
            glm::vec3 currentPos = mesh.getVertex(x, y);
            glm::vec3 oldPos = mesh.getOldVertex(x, y);
            glm::vec3 force = physics.getForce(vertexId);
            
            // Calcular aceleración
            glm::vec3 acceleration = force / mass;
            
            // Integración de Verlet
            glm::vec3 newPos = 2.0f * currentPos - oldPos + acceleration * dt2;
            
            // Actualizar posiciones
            mesh.setOldVertex(x, y, currentPos);
            mesh.setVertex(x, y, newPos);
        }
    }
    
    // Aplicar restricciones después de la integración
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