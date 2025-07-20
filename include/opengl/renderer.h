#pragma once
#include "../core/mesh.h"
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class OpenGLRenderer {
private:
    GLuint shaderProgram;
    GLuint VAO, VBO, EBO;
    glm::mat4 view, projection;
    
    int windowWidth, windowHeight;
    
public:
    OpenGLRenderer();
    ~OpenGLRenderer();
    
    void initialize(int windowWidth, int windowHeight);
    void setupMeshBuffers(const Mesh& mesh);
    void render(const Mesh& mesh);
    void setupCamera(const glm::mat4& view, const glm::mat4& projection);
    void updateMeshData(const Mesh& mesh);
    
    unsigned int getVBO() const { return VBO; }
    void cleanup();
};