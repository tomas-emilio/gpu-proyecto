#include "shader_simulation.h"
#include "utils.h"
#include <iostream>
#include <cmath>

ShaderSimulation::ShaderSimulation() 
    : computeProgram(0), positionSSBO(0), oldPositionSSBO(0), forceSSBO(0),
      springSSBO(0), springDataSSBO(0), numSprings(0), lastFrameTime(0.0) {
}

ShaderSimulation::~ShaderSimulation() {
    cleanup();
}

void ShaderSimulation::initialize(int meshWidth, int meshHeight) {
    width = meshWidth;
    height = meshHeight;
    numVertices = width * height;
    spacing = 0.1f;
    
    // Cargar compute shader
    computeProgram = OpenGLUtils::createComputeShaderProgram("shaders/compute.glsl");
    if (computeProgram == 0) {
        std::cerr << "Failed to create compute shader program" << std::endl;
        return;
    }
    
    // Inicializar posiciones
    hostPositions.clear();
    hostPositions.resize(numVertices);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            hostPositions[idx] = glm::vec4(x * spacing, y * spacing, 0.0f, 1.0f);
        }
    }
    
    generateSprings();
    initializeBuffers();
    
    std::cout << "Shader Simulation initialized: " << width << "x" << height 
              << " (" << numVertices << " vertices, " << numSprings << " springs)" << std::endl;
}

void ShaderSimulation::generateSprings() {
    hostSprings.clear();
    hostSpringData.clear();
    
    // Resortes estructurales
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int currentVertex = y * width + x;
            
            // Horizontal
            if (x < width - 1) {
                int rightVertex = y * width + (x + 1);
                hostSprings.push_back({currentVertex, rightVertex, 0, 0}); // STRUCTURAL
                hostSpringData.push_back({spacing, 50.0f, 0.0f, 0.0f});
            }
            
            // Vertical
            if (y < height - 1) {
                int bottomVertex = (y + 1) * width + x;
                hostSprings.push_back({currentVertex, bottomVertex, 0, 0}); // STRUCTURAL
                hostSpringData.push_back({spacing, 50.0f, 0.0f, 0.0f});
            }
        }
    }
    
    // Resortes de corte (diagonales)
    for (int y = 0; y < height - 1; ++y) {
        for (int x = 0; x < width - 1; ++x) {
            int currentVertex = y * width + x;
            int diagVertex = (y + 1) * width + (x + 1);
            float diagLength = spacing * sqrtf(2.0f);
            
            hostSprings.push_back({currentVertex, diagVertex, 1, 0}); // SHEAR
            hostSpringData.push_back({diagLength, 25.0f, 0.0f, 0.0f});
            
            int rightVertex = y * width + (x + 1);
            int bottomLeftVertex = (y + 1) * width + x;
            hostSprings.push_back({rightVertex, bottomLeftVertex, 1, 0}); // SHEAR
            hostSpringData.push_back({diagLength, 25.0f, 0.0f, 0.0f});
        }
    }
    
    // Resortes virtuales
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int currentVertex = y * width + x;
            
            if (x < width - 2) {
                int farRightVertex = y * width + (x + 2);
                hostSprings.push_back({currentVertex, farRightVertex, 2, 0}); // VIRTUAL
                hostSpringData.push_back({spacing * 2.0f, 15.0f, 0.0f, 0.0f});
            }
            
            if (y < height - 2) {
                int farBottomVertex = (y + 2) * width + x;
                hostSprings.push_back({currentVertex, farBottomVertex, 2, 0}); // VIRTUAL
                hostSpringData.push_back({spacing * 2.0f, 15.0f, 0.0f, 0.0f});
            }
        }
    }
    
    numSprings = hostSprings.size();
}

void ShaderSimulation::initializeBuffers() {
    // Position buffer
    glGenBuffers(1, &positionSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, positionSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numVertices * sizeof(glm::vec4), 
                 hostPositions.data(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, positionSSBO);
    
    // Old position buffer
    glGenBuffers(1, &oldPositionSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, oldPositionSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numVertices * sizeof(glm::vec4), 
                 hostPositions.data(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, oldPositionSSBO);
    
    // Force buffer
    glGenBuffers(1, &forceSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, forceSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numVertices * sizeof(glm::vec4), 
                 nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, forceSSBO);
    
    // Spring buffer
    glGenBuffers(1, &springSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, springSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numSprings * sizeof(ShaderSpring), 
                 hostSprings.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, springSSBO);
    
    // Spring data buffer
    glGenBuffers(1, &springDataSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, springDataSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numSprings * sizeof(ShaderSpringData), 
                 hostSpringData.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, springDataSSBO);
    
    OpenGLUtils::checkGLError("Buffer initialization");
}

void ShaderSimulation::update(float deltaTime) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    glUseProgram(computeProgram);
    
    // Set uniforms
    glUniform1i(glGetUniformLocation(computeProgram, "numSprings"), numSprings);
    glUniform1i(glGetUniformLocation(computeProgram, "meshWidth"), width);
    glUniform1i(glGetUniformLocation(computeProgram, "meshHeight"), height);
    glUniform1f(glGetUniformLocation(computeProgram, "deltaTime"), deltaTime);
    glUniform1f(glGetUniformLocation(computeProgram, "mass"), 1.0f);
    
    // Calculate work groups
    GLuint groupsX = (width + 15) / 16;  // 16 is local_size_x
    GLuint groupsY = (height + 15) / 16; // 16 is local_size_y
    
    // Phase 0: Clear forces
    glUniform1i(glGetUniformLocation(computeProgram, "phase"), 0);
    glDispatchCompute(groupsX, groupsY, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    
    // Phase 1: Calculate forces
    glUniform1i(glGetUniformLocation(computeProgram, "phase"), 1);
    glDispatchCompute(groupsX, groupsY, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    
    // Phase 2: Integrate
    glUniform1i(glGetUniformLocation(computeProgram, "phase"), 2);
    glDispatchCompute(groupsX, groupsY, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    
    glFinish(); // Wait for GPU to complete
    
    auto endTime = std::chrono::high_resolution_clock::now();
    lastFrameTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    
    OpenGLUtils::checkGLError("Compute shader dispatch");
}

void ShaderSimulation::handleMouseInteraction(float x, float y, float force) {
    int meshX = static_cast<int>(x / spacing);
    int meshY = static_cast<int>(y / spacing);
    
    if (meshX >= 0 && meshX < width && meshY >= 0 && meshY < height) {
        int targetVertex = meshY * width + meshX;
        
        // Read current position, modify, and write back
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, positionSSBO);
        glm::vec4* positions = (glm::vec4*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_WRITE);
        if (positions) {
            positions[targetVertex].z += force;
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        }
    }
}

void ShaderSimulation::getMesh(Mesh& mesh) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, positionSSBO);
    glm::vec4* positions = (glm::vec4*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    
    if (positions) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                glm::vec4 pos = positions[idx];
                mesh.setVertex(x, y, glm::vec3(pos.x, pos.y, pos.z));
            }
        }
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }
}

double ShaderSimulation::getLastFrameTime() const {
    return lastFrameTime;
}

void ShaderSimulation::cleanup() {
    if (positionSSBO) {
        glDeleteBuffers(1, &positionSSBO);
        positionSSBO = 0;
    }
    if (oldPositionSSBO) {
        glDeleteBuffers(1, &oldPositionSSBO);
        oldPositionSSBO = 0;
    }
    if (forceSSBO) {
        glDeleteBuffers(1, &forceSSBO);
        forceSSBO = 0;
    }
    if (springSSBO) {
        glDeleteBuffers(1, &springSSBO);
        springSSBO = 0;
    }
    if (springDataSSBO) {
        glDeleteBuffers(1, &springDataSSBO);
        springDataSSBO = 0;
    }
    if (computeProgram) {
        glDeleteProgram(computeProgram);
        computeProgram = 0;
    }
}