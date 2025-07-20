#pragma once
#include <vector>
#include <glm/glm.hpp>

class Mesh {
private:
    int width, height;
    float spacing;
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> oldVertices;
    std::vector<unsigned int> indices;

public:
    void initialize(int width, int height, float spacing);
    
    glm::vec3& getVertex(int x, int y);
    const glm::vec3& getVertex(int x, int y) const;
    void setVertex(int x, int y, const glm::vec3& pos);
    
    glm::vec3& getOldVertex(int x, int y);
    void setOldVertex(int x, int y, const glm::vec3& pos);
    
    std::vector<unsigned int>& getIndices();
    
    void applyForce(int x, int y, const glm::vec3& force);
    
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    float getSpacing() const { return spacing; }
    
    const std::vector<glm::vec3>& getVertices() const { return vertices; }
    std::vector<glm::vec3>& getVertices() { return vertices; }
};