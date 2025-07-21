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
    
    //cargar compute shader
    computeProgram = OpenGLUtils::createComputeShaderProgram("shaders/compute.glsl");
    if (computeProgram == 0) {
        std::cerr << "Failed to create compute shader program" << std::endl;
        return;
    }
    
    //inicializar posiciones
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

//modificar generateSprings en shader_simulation.cpp
void ShaderSimulation::generateSprings() {
    hostSprings.clear();
    hostSpringData.clear();
    
    extern TissueParams g_tissueParams;
    
    //resortes estructurales
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int currentVertex = y * width + x;
            
            //horizontal
            if (x < width - 1) {
                int rightVertex = y * width + (x + 1);
                hostSprings.push_back({currentVertex, rightVertex, 0, 0}); // STRUCTURAL
                hostSpringData.push_back({spacing, g_tissueParams.structuralStiffness, 0.0f, 0.0f});
            }
            
            //vertical
            if (y < height - 1) {
                int bottomVertex = (y + 1) * width + x;
                hostSprings.push_back({currentVertex, bottomVertex, 0, 0}); // STRUCTURAL
                hostSpringData.push_back({spacing, g_tissueParams.structuralStiffness, 0.0f, 0.0f});
            }
        }
    }
    
    //resortes de corte (diagonales)
    for (int y = 0; y < height - 1; ++y) {
        for (int x = 0; x < width - 1; ++x) {
            int currentVertex = y * width + x;
            int diagVertex = (y + 1) * width + (x + 1);
            float diagLength = spacing * sqrtf(2.0f);
            
            hostSprings.push_back({currentVertex, diagVertex, 1, 0}); // SHEAR
            hostSpringData.push_back({diagLength, g_tissueParams.shearStiffness, 0.0f, 0.0f});
            
            int rightVertex = y * width + (x + 1);
            int bottomLeftVertex = (y + 1) * width + x;
            hostSprings.push_back({rightVertex, bottomLeftVertex, 1, 0}); // SHEAR
            hostSpringData.push_back({diagLength, g_tissueParams.shearStiffness, 0.0f, 0.0f});
        }
    }
    
    //resortes virtuales
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int currentVertex = y * width + x;
            
            if (x < width - 2) {
                int farRightVertex = y * width + (x + 2);
                hostSprings.push_back({currentVertex, farRightVertex, 2, 0}); // VIRTUAL
                hostSpringData.push_back({spacing * 2.0f, g_tissueParams.virtualStiffness, 0.0f, 0.0f});
            }
            
            if (y < height - 2) {
                int farBottomVertex = (y + 2) * width + x;
                hostSprings.push_back({currentVertex, farBottomVertex, 2, 0}); // VIRTUAL
                hostSpringData.push_back({spacing * 2.0f, g_tissueParams.virtualStiffness, 0.0f, 0.0f});
            }
        }
    }
    
    numSprings = hostSprings.size();
}

void ShaderSimulation::initializeBuffers() {
    //position buffer
    glGenBuffers(1, &positionSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, positionSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numVertices * sizeof(glm::vec4), 
                 hostPositions.data(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, positionSSBO);
    
    //old position buffer
    glGenBuffers(1, &oldPositionSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, oldPositionSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numVertices * sizeof(glm::vec4), 
                 hostPositions.data(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, oldPositionSSBO);
    
    //force buffer
    glGenBuffers(1, &forceSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, forceSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numVertices * sizeof(glm::vec4), 
                 nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, forceSSBO);
    
    //spring buffer
    glGenBuffers(1, &springSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, springSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numSprings * sizeof(ShaderSpring), 
                 hostSprings.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, springSSBO);
    
    //spring data buffer
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
    OpenGLUtils::checkGLError("Use program");
    
    //verificar que el programa es válido
    GLint linked;
    glGetProgramiv(computeProgram, GL_LINK_STATUS, &linked);
    if (!linked) {
        std::cerr << "Compute shader program not linked!" << std::endl;
        return;
    }
    
    //set uniforms con verificación
    GLint loc;
    loc = glGetUniformLocation(computeProgram, "numSprings");
    if (loc >= 0) glUniform1i(loc, numSprings);
    
    loc = glGetUniformLocation(computeProgram, "meshWidth");
    if (loc >= 0) glUniform1i(loc, width);
    
    loc = glGetUniformLocation(computeProgram, "meshHeight");
    if (loc >= 0) glUniform1i(loc, height);
    
    loc = glGetUniformLocation(computeProgram, "deltaTime");
    if (loc >= 0) glUniform1f(loc, deltaTime);
    
    loc = glGetUniformLocation(computeProgram, "mass");
    if (loc >= 0) glUniform1f(loc, 1.0f);
    
    OpenGLUtils::checkGLError("Set uniforms");
    
    //verificar lmites de work groups
    GLint maxWorkGroupSize[3];
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &maxWorkGroupSize[0]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &maxWorkGroupSize[1]);
    
    GLuint groupsX = (width + 15) / 16;
    GLuint groupsY = (height + 15) / 16;
    
    //verificar limites
    if (groupsX > maxWorkGroupSize[0] || groupsY > maxWorkGroupSize[1]) {
        std::cerr << "Work group size too large!" << std::endl;
        return;
    }
    
    //fase 0: Clear forces
    loc = glGetUniformLocation(computeProgram, "phase");
    if (loc >= 0) glUniform1i(loc, 0);
    glDispatchCompute(groupsX, groupsY, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    OpenGLUtils::checkGLError("Phase 0");
    
    //fase 1: Calcular fuerzas
    if (loc >= 0) glUniform1i(loc, 1);
    glDispatchCompute(groupsX, groupsY, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    OpenGLUtils::checkGLError("Phase 1");
    
    //fase 2: Integrate
    if (loc >= 0) glUniform1i(loc, 2);
    glDispatchCompute(groupsX, groupsY, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    OpenGLUtils::checkGLError("Phase 2");
    
    glFinish();
    
    auto endTime = std::chrono::high_resolution_clock::now();
    lastFrameTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
}

void ShaderSimulation::handleMouseInteraction(float x, float y, float force) {
    int meshX = static_cast<int>(x / spacing);
    int meshY = static_cast<int>(y / spacing);
    
    if (meshX >= 0 && meshX < width && meshY >= 0 && meshY < height) {
        int targetVertex = meshY * width + meshX;
        
        //read current position, modify, and write back
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

void ShaderSimulation::reset() {
    //reinicializar posiciones
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            hostPositions[idx] = glm::vec4(x * spacing, y * spacing, 0.0f, 1.0f);
        }
    }
    
    //actualizar buffers
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, positionSSBO);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, numVertices * sizeof(glm::vec4), hostPositions.data());
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, oldPositionSSBO);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, numVertices * sizeof(glm::vec4), hostPositions.data());
}

void ShaderSimulation::updateParams(const TissueParams& params) {
    //regenerar resortes con nuevos parámetros
    generateSprings();
    
    //actualizar buffer de datos de resortes
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, springDataSSBO);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, numSprings * sizeof(ShaderSpringData), hostSpringData.data());
}