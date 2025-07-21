#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <memory>
#include <string>

#include "cpu/cpu_simulation.h"
#include "cuda/cuda_simulation.h"
#include "opengl/shader_simulation.h"
#include "opengl/renderer.h"

enum SimulationType {
    SIM_CPU,
    SIM_CUDA,
    SIM_SHADER
};

class Application {
private:
    GLFWwindow* window;
    OpenGLRenderer renderer;
    
    SimulationType simType;
    std::unique_ptr<CPUSimulation> cpuSim;
    std::unique_ptr<CUDASimulation> cudaSim;
    std::unique_ptr<ShaderSimulation> shaderSim;
    
    Mesh tempMesh; // Para obtener datos de GPU simulations
    
    int windowWidth, windowHeight;
    int meshWidth, meshHeight;
    bool mousePressed;
    double lastMouseX, lastMouseY;
    
    // Métricas de rendimiento
    double frameTimeSum;
    int frameCount;
    double lastTime;
    
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    
    void handleMouseInput(double xpos, double ypos);
    void updatePerformanceMetrics();
    void printPerformanceStats();
    void resetTissue();
    void updateTissueParams();
    void printControls();
    
public:
    Application();
    ~Application();
    
    bool initialize(const std::string& simTypeStr, int meshSize = 40);
    void run();
    void cleanup();
};