#pragma once
#include "../core/mesh.h"
#include <GL/glew.h>
#include <chrono>
#include <vector>
#include "../core/tissue_params.h"

struct ShaderSpring {
    int vertex1, vertex2;
    int type;
    int padding;
};

struct ShaderSpringData {
    float restLength;
    float stiffness;
    float padding1;
    float padding2;
};

class ShaderSimulation {
private:
    int width, height, numVertices;
    float spacing;
    
    //objetos oengl
    GLuint computeProgram;
    GLuint positionSSBO, oldPositionSSBO, forceSSBO;
    GLuint springSSBO, springDataSSBO;
    
    //host data
    std::vector<glm::vec4> hostPositions;
    std::vector<ShaderSpring> hostSprings;
    std::vector<ShaderSpringData> hostSpringData;
    int numSprings;
    
    //timing
    std::chrono::high_resolution_clock::time_point lastTime;
    double lastFrameTime;
    
    void generateSprings();
    void initializeBuffers();
    
public:
    ShaderSimulation();
    ~ShaderSimulation();
    
    void initialize(int meshWidth, int meshHeight);
    void update(float deltaTime);
    void handleMouseInteraction(float x, float y, float force);
    void getMesh(Mesh& mesh);
    
    double getLastFrameTime() const;
    void cleanup();
    void reset();
    void updateParams(const TissueParams& params);
};