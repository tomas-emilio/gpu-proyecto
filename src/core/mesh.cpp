#include "mesh.h"

void Mesh::initialize(int w, int h, float s) {
    width = w;
    height = h;
    spacing = s;
    
    vertices.clear();
    oldVertices.clear();
    indices.clear();
    
    // Crear vértices
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float posX = x * spacing;
            float posY = y * spacing;
            glm::vec3 pos(posX, posY, 0.0f);
            vertices.push_back(pos);
            oldVertices.push_back(pos);
        }
    }
    
    // Crear índices para triángulos
    for (int y = 0; y < height - 1; ++y) {
        for (int x = 0; x < width - 1; ++x) {
            int topLeft = y * width + x;
            int topRight = y * width + (x + 1);
            int bottomLeft = (y + 1) * width + x;
            int bottomRight = (y + 1) * width + (x + 1);
            
            // Triángulo 1
            indices.push_back(topLeft);
            indices.push_back(bottomLeft);
            indices.push_back(topRight);
            
            // Triángulo 2
            indices.push_back(topRight);
            indices.push_back(bottomLeft);
            indices.push_back(bottomRight);
        }
    }
}

glm::vec3& Mesh::getVertex(int x, int y) {
    return vertices[y * width + x];
}

const glm::vec3& Mesh::getVertex(int x, int y) const {
    return vertices[y * width + x];
}

void Mesh::setVertex(int x, int y, const glm::vec3& pos) {
    vertices[y * width + x] = pos;
}

glm::vec3& Mesh::getOldVertex(int x, int y) {
    return oldVertices[y * width + x];
}

void Mesh::setOldVertex(int x, int y, const glm::vec3& pos) {
    oldVertices[y * width + x] = pos;
}

std::vector<unsigned int>& Mesh::getIndices() {
    return indices;
}

void Mesh::applyForce(int x, int y, const glm::vec3& force) {
    if (x >= 0 && x < width && y >= 0 && y < height) {
        vertices[y * width + x] += force;
    }
}