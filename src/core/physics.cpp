#include "physics.h"
#include <glm/glm.hpp>

void PhysicsModel::generateSprings(const Mesh& mesh) {
    springs.clear();
    int width = mesh.getWidth();
    int height = mesh.getHeight();
    float spacing = mesh.getSpacing();
    
    // Resortes estructurales (horizontal y vertical)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int currentVertex = y * width + x;
            
            // Horizontal
            if (x < width - 1) {
                int rightVertex = y * width + (x + 1);
                Spring spring = {currentVertex, rightVertex, spacing, 50.0f, STRUCTURAL};
                springs.push_back(spring);
            }
            
            // Vertical
            if (y < height - 1) {
                int bottomVertex = (y + 1) * width + x;
                Spring spring = {currentVertex, bottomVertex, spacing, 50.0f, STRUCTURAL};
                springs.push_back(spring);
            }
        }
    }
    
    // Resortes de corte (diagonales)
    for (int y = 0; y < height - 1; ++y) {
        for (int x = 0; x < width - 1; ++x) {
            int currentVertex = y * width + x;
            int diagVertex = (y + 1) * width + (x + 1);
            float diagLength = spacing * sqrt(2.0f);
            
            Spring spring1 = {currentVertex, diagVertex, diagLength, 25.0f, SHEAR};
            springs.push_back(spring1);
            
            int rightVertex = y * width + (x + 1);
            int bottomLeftVertex = (y + 1) * width + x;
            Spring spring2 = {rightVertex, bottomLeftVertex, diagLength, 25.0f, SHEAR};
            springs.push_back(spring2);
        }
    }
    
    // Resortes virtuales (para simular volumen)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int currentVertex = y * width + x;
            
            if (x < width - 2) {
                int farRightVertex = y * width + (x + 2);
                Spring spring = {currentVertex, farRightVertex, spacing * 2.0f, 15.0f, VIRTUAL};
                springs.push_back(spring);
            }
            
            if (y < height - 2) {
                int farBottomVertex = (y + 2) * width + x;
                Spring spring = {currentVertex, farBottomVertex, spacing * 2.0f, 15.0f, VIRTUAL};
                springs.push_back(spring);
            }
        }
    }
}

void PhysicsModel::calculateForces(const Mesh& mesh) {
    const auto& vertices = mesh.getVertices();
    
    for (const auto& spring : springs) {
        glm::vec3 pos1 = vertices[spring.vertex1];
        glm::vec3 pos2 = vertices[spring.vertex2];
        
        glm::vec3 diff = pos2 - pos1;
        float currentLength = glm::length(diff);
        
        if (currentLength > 0.0f) {
            float displacement = currentLength - spring.restLength;
            glm::vec3 direction = diff / currentLength;
            glm::vec3 springForce = direction * displacement * spring.stiffness;
            
            forces[spring.vertex1] += springForce;
            forces[spring.vertex2] -= springForce;
        }
    }
}

glm::vec3 PhysicsModel::getForce(int vertexId) const {
    if (vertexId < forces.size()) {
        return forces[vertexId];
    }
    return glm::vec3(0.0f);
}

void PhysicsModel::clearForces(int numVertices) {
    forces.clear();
    forces.resize(numVertices, glm::vec3(0.0f));
}

void PhysicsModel::applyConstraints(Mesh& mesh) {
    int width = mesh.getWidth();
    int height = mesh.getHeight();
    
    // Fijar bordes superior e inferior
    for (int x = 0; x < width; ++x) {
        // Borde superior
        glm::vec3 topPos = mesh.getVertex(x, 0);
        topPos.z = 0.0f;
        mesh.setVertex(x, 0, topPos);
        
        // Borde inferior
        glm::vec3 bottomPos = mesh.getVertex(x, height - 1);
        bottomPos.z = 0.0f;
        mesh.setVertex(x, height - 1, bottomPos);
    }
}