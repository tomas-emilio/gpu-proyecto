#include <iostream>
#include <thread>
#include <chrono>
#include "cpu/cpu_simulation.h"

void printMeshState(const Mesh& mesh, int frame) {
    int width = mesh.getWidth();
    int height = mesh.getHeight();
    
    std::cout << "Frame " << frame << " - Center vertex position: ";
    glm::vec3 centerPos = mesh.getVertex(width/2, height/2);
    std::cout << "(" << centerPos.x << ", " << centerPos.y << ", " << centerPos.z << ")" << std::endl;
}

int main() {
    std::cout << "=== Tissue Deformation Simulation - CPU Version ===" << std::endl;
    
    CPUSimulation simulation;
    
    // Configuración de la simulación
    int meshWidth = 20;
    int meshHeight = 20;
    float deltaTime = 0.016f; // ~60 FPS
    
    simulation.initialize(meshWidth, meshHeight);
    
    std::cout << "\nRunning simulation..." << std::endl;
    std::cout << "Press Ctrl+C to stop\n" << std::endl;
    
    int frame = 0;
    double totalTime = 0.0;
    int frameCount = 0;
    
    while (frame < 1000) { // Limitar a 1000 frames para la prueba
        // Simular interacción del usuario cada 100 frames
        if (frame % 100 == 50) {
            simulation.handleMouseInteraction(1.0f, 1.0f, -0.5f);
            std::cout << "Applied user force at frame " << frame << std::endl;
        }
        
        // Actualizar simulación
        simulation.update(deltaTime);
        
        // Recopilar métricas
        double frameTime = simulation.getLastFrameTime();
        totalTime += frameTime;
        frameCount++;
        
        // Mostrar estado cada 50 frames
        if (frame % 50 == 0) {
            printMeshState(simulation.getMesh(), frame);
            std::cout << "Frame time: " << frameTime << " ms" << std::endl;
        }
        
        // Simular frame rate
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
        frame++;
    }
    
    // Mostrar estadísticas finales
    std::cout << "\n=== Performance Statistics ===" << std::endl;
    std::cout << "Total frames: " << frameCount << std::endl;
    std::cout << "Average frame time: " << (totalTime / frameCount) << " ms" << std::endl;
    std::cout << "Average FPS: " << (1000.0 / (totalTime / frameCount)) << std::endl;
    
    return 0;
}