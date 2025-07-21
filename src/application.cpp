#include "application.h"
#include <iostream>
#include <iomanip>
#include "core/tissue_params.h"

TissueParams g_tissueParams;

Application::Application() 
    : window(nullptr), simType(SIM_CPU), mousePressed(false), 
      lastMouseX(0), lastMouseY(0), frameTimeSum(0), frameCount(0), lastTime(0),
      windowWidth(800), windowHeight(600), meshWidth(40), meshHeight(40) {
}

Application::~Application() {
    cleanup();
}

bool Application::initialize(const std::string& simTypeStr, int meshSize) {
    meshWidth = meshSize;
    meshHeight = meshSize;
    
    //determinar tipo de simulacion
    if (simTypeStr == "cpu") {
        simType = SIM_CPU;
    } else if (simTypeStr == "cuda") {
        simType = SIM_CUDA;
    } else if (simTypeStr == "shader") {
        simType = SIM_SHADER;
    } else {
        std::cerr << "Invalid simulation type: " << simTypeStr << std::endl;
        return false;
    }
    
    //inicializar GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }
    
    //configurar contexto opengl
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    //crear ventana
    std::string title = "Tissue Simulation - " + simTypeStr + " (" + 
                       std::to_string(meshWidth) + "x" + std::to_string(meshHeight) + ")";
    window = glfwCreateWindow(windowWidth, windowHeight, title.c_str(), nullptr, nullptr);
    
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // VSync
    
    //configurar callbacks
    glfwSetWindowUserPointer(window, this);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetKeyCallback(window, keyCallback);
    
    //inicializar glew
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return false;
    }
    
    //inicializar renderer
    renderer.initialize(windowWidth, windowHeight);
    
    //inicializar simulación según el tipo
    tempMesh.initialize(meshWidth, meshHeight, 0.1f);
    
    switch (simType) {
        case SIM_CPU:
            cpuSim = std::make_unique<CPUSimulation>();
            cpuSim->initialize(meshWidth, meshHeight);
            renderer.setupMeshBuffers(cpuSim->getMesh());
            break;
            
        case SIM_CUDA:
            cudaSim = std::make_unique<CUDASimulation>();
            cudaSim->initialize(meshWidth, meshHeight);
            renderer.setupMeshBuffers(tempMesh);
            break;
            
        case SIM_SHADER:
            shaderSim = std::make_unique<ShaderSimulation>();
            shaderSim->initialize(meshWidth, meshHeight);
            renderer.setupMeshBuffers(tempMesh);
            break;
    }
    
    lastTime = glfwGetTime();
    
    std::cout << "Application initialized successfully" << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "- Left click and drag to apply force" << std::endl;
    std::cout << "- ESC to quit" << std::endl;
    std::cout << "- P to print performance stats" << std::endl;
    
    return true;
}

void Application::run() {
    while (!glfwWindowShouldClose(window)) {
        double currentTime = glfwGetTime();
        float deltaTime = static_cast<float>(currentTime - lastTime);
        lastTime = currentTime;
        
        //limitar deltaTime para estabilidad
        if (deltaTime > 0.033f) deltaTime = 0.033f; //maximo 30 FPS
        
        //actualizar simulacionn
        switch (simType) {
            case SIM_CPU:
                cpuSim->update(deltaTime);
                renderer.render(cpuSim->getMesh());
                break;
                
            case SIM_CUDA:
                cudaSim->update(deltaTime);
                cudaSim->getMesh(tempMesh);
                renderer.render(tempMesh);
                break;
                
            case SIM_SHADER:
                shaderSim->update(deltaTime);
                shaderSim->getMesh(tempMesh);
                renderer.render(tempMesh);
                break;
        }
        
        updatePerformanceMetrics();
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

void Application::handleMouseInput(double xpos, double ypos) {
    if (!mousePressed) return;
    
    //convrtir coordenadas de pantalla a coordenadas del mundo
    float normalizedX = (float)xpos / windowWidth;
    float normalizedY = 1.0f - (float)ypos / windowHeight; //invertir Y
    
    float worldX = normalizedX * (meshWidth - 1) * 0.1f;
    float worldY = normalizedY * (meshHeight - 1) * 0.1f;
    
    //calcular fuerza basada en el movimiento del mouse
    float deltaX = static_cast<float>(xpos - lastMouseX);
    float deltaY = static_cast<float>(ypos - lastMouseY);
    float force = -(deltaX + deltaY) * 0.001f; //fuerza negativa para deformacion hacia adentro
    
    //aplicar fuerza segun el tipo de simulacion
    switch (simType) {
        case SIM_CPU:
            cpuSim->handleMouseInteraction(worldX, worldY, force);
            break;
            
        case SIM_CUDA:
            cudaSim->handleMouseInteraction(worldX, worldY, force);
            break;
            
        case SIM_SHADER:
            shaderSim->handleMouseInteraction(worldX, worldY, force);
            break;
    }
    
    lastMouseX = xpos;
    lastMouseY = ypos;
}

void Application::updatePerformanceMetrics() {
    double frameTime = 0.0;
    
    switch (simType) {
        case SIM_CPU:
            frameTime = cpuSim->getLastFrameTime();
            break;
        case SIM_CUDA:
            frameTime = cudaSim->getLastFrameTime();
            break;
        case SIM_SHADER:
            frameTime = shaderSim->getLastFrameTime();
            break;
    }
    
    frameTimeSum += frameTime;
    frameCount++;
}

void Application::printPerformanceStats() {
    if (frameCount > 0) {
        double avgFrameTime = frameTimeSum / frameCount;
        double avgFPS = 1000.0 / avgFrameTime;
        
        std::cout << "=== Performance Statistics ===" << std::endl;
        std::cout << "Simulation type: ";
        switch (simType) {
            case SIM_CPU: std::cout << "CPU"; break;
            case SIM_CUDA: std::cout << "CUDA"; break;
            case SIM_SHADER: std::cout << "Shader"; break;
        }
        std::cout << std::endl;
        std::cout << "Frames counted: " << frameCount << std::endl;
        std::cout << "Average frame time: " << std::fixed << std::setprecision(3) 
                  << avgFrameTime << " ms" << std::endl;
        std::cout << "Average FPS: " << std::fixed << std::setprecision(1) 
                  << avgFPS << std::endl;
        
        //reset counters
        frameTimeSum = 0.0;
        frameCount = 0;
    }
}

//callbacks estáticos
void Application::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            app->mousePressed = true;
            glfwGetCursorPos(window, &app->lastMouseX, &app->lastMouseY);
        } else if (action == GLFW_RELEASE) {
            app->mousePressed = false;
        }
    }
}

void Application::cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    app->handleMouseInput(xpos, ypos);
}

void Application::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    
    if (action == GLFW_PRESS) {
        bool paramsChanged = false;
        
        switch (key) {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                break;
                
            case GLFW_KEY_P:
                app->printPerformanceStats();
                break;
                
            //structural stiffness (Q/A)
            case GLFW_KEY_Q:
                g_tissueParams.increaseStructural();
                paramsChanged = true;
                break;
            case GLFW_KEY_A:
                g_tissueParams.decreaseStructural();
                paramsChanged = true;
                break;
                
            //shear stiffness (W/S)  
            case GLFW_KEY_W:
                g_tissueParams.increaseShear();
                paramsChanged = true;
                break;
            case GLFW_KEY_S:
                g_tissueParams.decreaseShear();
                paramsChanged = true;
                break;
                
            //virtual stiffness (E/D)
            case GLFW_KEY_E:
                g_tissueParams.increaseVirtual();
                paramsChanged = true;
                break;
            case GLFW_KEY_D:
                g_tissueParams.decreaseVirtual();
                paramsChanged = true;
                break;
                
            //damping (R/F)
            case GLFW_KEY_R:
                g_tissueParams.increaseDamping();
                paramsChanged = true;
                break;
            case GLFW_KEY_F:
                g_tissueParams.decreaseDamping();
                paramsChanged = true;
                break;
                
            //reset tissue (SPACE)
            case GLFW_KEY_SPACE:
                app->resetTissue();
                break;
                
            //reset parameters (BACKSPACE)
            case GLFW_KEY_BACKSPACE:
                g_tissueParams.reset();
                paramsChanged = true;
                break;
                
            //show help (H)
            case GLFW_KEY_H:
                app->printControls();
                break;
        }
        
        if (paramsChanged) {
            g_tissueParams.print();
            app->updateTissueParams();
        }
    }
}

void Application::cleanup() {
    renderer.cleanup();
    
    if (window) {
        glfwDestroyWindow(window);
        window = nullptr;
    }
    
    glfwTerminate();
}

void Application::resetTissue() {
    switch (simType) {
        case SIM_CPU:
            cpuSim->reset();
            break;
        case SIM_CUDA:
            cudaSim->reset();
            break;
        case SIM_SHADER:
            shaderSim->reset();
            break;
    }
    std::cout << "Tissue reset!" << std::endl;
}

void Application::updateTissueParams() {
    switch (simType) {
        case SIM_CPU:
            cpuSim->updateParams(g_tissueParams);
            break;
        case SIM_CUDA:
            cudaSim->updateParams(g_tissueParams);
            break;
        case SIM_SHADER:
            shaderSim->updateParams(g_tissueParams);
            break;
    }
}

void Application::printControls() {
    std::cout << "\n=== CONTROLS ===" << std::endl;
    std::cout << "Mouse: Left click + drag to apply force" << std::endl;
    std::cout << "Q/A: Increase/Decrease structural stiffness" << std::endl;
    std::cout << "W/S: Increase/Decrease shear stiffness" << std::endl;
    std::cout << "E/D: Increase/Decrease virtual stiffness" << std::endl;
    std::cout << "R/F: Increase/Decrease damping" << std::endl;
    std::cout << "SPACE: Reset tissue positions" << std::endl;
    std::cout << "BACKSPACE: Reset all parameters" << std::endl;
    std::cout << "P: Print performance stats" << std::endl;
    std::cout << "H: Show this help" << std::endl;
    std::cout << "ESC: Exit" << std::endl;
}