#include "renderer.h"
#include "utils.h"
#include <iostream>

OpenGLRenderer::OpenGLRenderer() 
    : shaderProgram(0), VAO(0), VBO(0), EBO(0), windowWidth(0), windowHeight(0) {
}

OpenGLRenderer::~OpenGLRenderer() {
    cleanup();
}

void OpenGLRenderer::initialize(int width, int height) {
    windowWidth = width;
    windowHeight = height;
    
    // Crear programa de shaders
    shaderProgram = OpenGLUtils::createShaderProgram("shaders/vertex.glsl", "shaders/fragment.glsl");
    if (shaderProgram == 0) {
        std::cerr << "Failed to create render shader program" << std::endl;
        return;
    }
    
    // Configurar proyección
    projection = glm::perspective(
        glm::radians(45.0f), 
        (float)width / (float)height, 
        0.1f, 100.0f
    );
    
    // Configurar OpenGL
    glEnable(GL_DEPTH_TEST);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); // Wireframe para ver la malla
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    
    std::cout << "Renderer initialized" << std::endl;
}

void OpenGLRenderer::setupMeshBuffers(const Mesh& mesh) {
    const auto& vertices = mesh.getVertices();
    const auto& indices = mesh.getIndices();

    // Generar buffers
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    
    // Configurar VAO
    glBindVertexArray(VAO);
    
    // VBO
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), 
                 vertices.data(), GL_DYNAMIC_DRAW);
    
    // EBO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), 
                 indices.data(), GL_STATIC_DRAW);
    
    // Atributos de vértice
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    glBindVertexArray(0);
}

void OpenGLRenderer::updateMeshData(const Mesh& mesh) {
    const auto& vertices = mesh.getVertices();
    
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(glm::vec3), vertices.data());
}


void OpenGLRenderer::render(const Mesh& mesh) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glUseProgram(shaderProgram);
    
    // Calcular posición central de la malla dinámicamente
    float meshCenterX = (mesh.getWidth() - 1) * mesh.getSpacing() * 0.5f;
    float meshCenterY = (mesh.getHeight() - 1) * mesh.getSpacing() * 0.5f;
    float meshSize = std::max(meshCenterX, meshCenterY) * 2.5f;
    
    // Configurar cámara centrada en la malla
    glm::vec3 center(meshCenterX, meshCenterY, 0.0f);
    glm::vec3 cameraPos = center + glm::vec3(0.0f, 0.0f, meshSize);
    
    view = glm::lookAt(cameraPos, center, glm::vec3(0.0f, 1.0f, 0.0f));
    
    // Calcular MVP matrix
    glm::mat4 model = glm::mat4(1.0f);
    glm::mat4 mvp = projection * view * model;
    
    // Enviar matriz al shader
    GLint mvpLocation = glGetUniformLocation(shaderProgram, "mvp");
    glUniformMatrix4fv(mvpLocation, 1, GL_FALSE, &mvp[0][0]);
    
    // Actualizar datos de vértices
    updateMeshData(mesh);
    
    // Dibujar
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, mesh.getIndices().size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void OpenGLRenderer::setupCamera(const glm::mat4& v, const glm::mat4& p) {
    view = v;
    projection = p;
}

void OpenGLRenderer::cleanup() {
    if (VAO) {
        glDeleteVertexArrays(1, &VAO);
        VAO = 0;
    }
    if (VBO) {
        glDeleteBuffers(1, &VBO);
        VBO = 0;
    }
    if (EBO) {
        glDeleteBuffers(1, &EBO);
        EBO = 0;
    }
    if (shaderProgram) {
        glDeleteProgram(shaderProgram);
        shaderProgram = 0;
    }
}