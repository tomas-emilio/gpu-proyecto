#pragma once
#include <vector>
#include <glm/glm.hpp>
#include "mesh.h"
#include "tissue_params.h"

enum SpringType {
    STRUCTURAL,
    SHEAR,
    VIRTUAL
};

struct Spring {
    int vertex1, vertex2;
    float restLength;
    float stiffness;
    SpringType type;
};

class PhysicsModel {
private:
    std::vector<Spring> springs;
    std::vector<glm::vec3> forces;
    float damping;
    
public:
    PhysicsModel() : damping(0.98f) {}
    
    void generateSprings(const Mesh& mesh);
    void calculateForces(const Mesh& mesh);
    glm::vec3 getForce(int vertexId) const;
    void clearForces(int numVertices);
    void applyConstraints(Mesh& mesh);
    
    void setDamping(float d) { damping = d; }
    void updateParams(const TissueParams& params);
};