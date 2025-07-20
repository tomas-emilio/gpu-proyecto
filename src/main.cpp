#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include "cpu/cpu_simulation.h"
#include "cuda/cuda_simulation.h"

enum SimulationMode {
    CPU_MODE,
    CUDA_MODE,
    COMPARISON_MODE
};

void printMeshState(const Mesh& mesh, int frame, const std::string& mode) {
    int width = mesh.getWidth();
    int height = mesh.getHeight();
    
    std::cout << "[" << mode << "] Frame " << frame << " - Center vertex position: ";
    glm::vec3 centerPos = mesh.getVertex(width/2, height/2);
    std::cout << "(" << centerPos.x << ", " << centerPos.y << ", " << centerPos.z << ")" << std::endl;
}

void runSimulation(SimulationMode mode, int meshWidth, int meshHeight, int numFrames) {
    float deltaTime = 0.016f; // ~60 FPS
    
    if (mode == CPU_MODE || mode == COMPARISON_MODE) {
        std::cout << "\n=== CPU Simulation ===" << std::endl;
        
        CPUSimulation cpuSim;
        cpuSim.initialize(meshWidth, meshHeight);
        
        double cpuTotalTime = 0.0;
        int frameCount = 0;
        
        for (int frame = 0; frame < numFrames; ++frame) {
            // Simular interacción del usuario cada 100 frames
            if (frame % 100 == 50) {
                cpuSim.handleMouseInteraction(1.0f, 1.0f, -0.5f);
                std::cout << "[CPU] Applied user force at frame " << frame << std::endl;
            }
            
            // Actualizar simulación
            cpuSim.update(deltaTime);
            
            // Recopilar métricas
            double frameTime = cpuSim.getLastFrameTime();
            cpuTotalTime += frameTime;
            frameCount++;
            
            // Mostrar estado cada 100 frames
            if (frame % 100 == 0) {
                printMeshState(cpuSim.getMesh(), frame, "CPU");
                std::cout << "[CPU] Frame time: " << frameTime << " ms" << std::endl;
            }
        }
        
        // Mostrar estadísticas CPU
        std::cout << "\n=== CPU Performance Statistics ===" << std::endl;
        std::cout << "Total frames: " << frameCount << std::endl;
        std::cout << "Average frame time: " << (cpuTotalTime / frameCount) << " ms" << std::endl;
        std::cout << "Average FPS: " << (1000.0 / (cpuTotalTime / frameCount)) << std::endl;
        
        if (mode == CPU_MODE) return;
    }
    
    if (mode == CUDA_MODE || mode == COMPARISON_MODE) {
        std::cout << "\n=== CUDA Simulation ===" << std::endl;
        
        CUDASimulation cudaSim;
        cudaSim.initialize(meshWidth, meshHeight);
        
        // Crear malla temporal para obtener resultados
        Mesh tempMesh;
        tempMesh.initialize(meshWidth, meshHeight, 0.1f);
        
        double cudaTotalTime = 0.0;
        int frameCount = 0;
        
        for (int frame = 0; frame < numFrames; ++frame) {
            // Simular interacción del usuario cada 100 frames
            if (frame % 100 == 50) {
                cudaSim.handleMouseInteraction(1.0f, 1.0f, -0.5f);
                std::cout << "[CUDA] Applied user force at frame " << frame << std::endl;
            }
            
            // Actualizar simulación
            cudaSim.update(deltaTime);
            
            // Recopilar métricas
            double frameTime = cudaSim.getLastFrameTime();
            cudaTotalTime += frameTime;
            frameCount++;
            
            // Mostrar estado cada 100 frames
            if (frame % 100 == 0) {
                cudaSim.getMesh(tempMesh);
                printMeshState(tempMesh, frame, "CUDA");
                std::cout << "[CUDA] Frame time: " << frameTime << " ms" << std::endl;
            }
        }
        
        // Mostrar estadísticas CUDA
        std::cout << "\n=== CUDA Performance Statistics ===" << std::endl;
        std::cout << "Total frames: " << frameCount << std::endl;
        std::cout << "Average frame time: " << (cudaTotalTime / frameCount) << " ms" << std::endl;
        std::cout << "Average FPS: " << (1000.0 / (cudaTotalTime / frameCount)) << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "=== Tissue Deformation Simulation - CPU vs CUDA Comparison ===" << std::endl;
    
    // Configuración de la simulación
    int meshWidth = 40;
    int meshHeight = 40;
    int numFrames = 500;
    
    SimulationMode mode = COMPARISON_MODE;
    
    // Procesar argumentos de línea de comandos
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "cpu") {
            mode = CPU_MODE;
        } else if (arg == "cuda") {
            mode = CUDA_MODE;
        } else if (arg == "compare") {
            mode = COMPARISON_MODE;
        }
    }
    
    // Configurar tamaño de malla desde argumentos
    if (argc > 2) {
        meshWidth = std::atoi(argv[2]);
        meshHeight = meshWidth;
    }
    
    if (argc > 3) {
        numFrames = std::atoi(argv[3]);
    }
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "- Mesh size: " << meshWidth << "x" << meshHeight 
              << " (" << (meshWidth * meshHeight) << " vertices)" << std::endl;
    std::cout << "- Number of frames: " << numFrames << std::endl;
    std::cout << "- Mode: ";
    
    switch (mode) {
        case CPU_MODE:
            std::cout << "CPU only" << std::endl;
            break;
        case CUDA_MODE:
            std::cout << "CUDA only" << std::endl;
            break;
        case COMPARISON_MODE:
            std::cout << "CPU vs CUDA comparison" << std::endl;
            break;
    }
    
    std::cout << "\nUsage: " << argv[0] << " [cpu|cuda|compare] [mesh_size] [num_frames]" << std::endl;
    std::cout << "Running simulation...\n" << std::endl;
    
    runSimulation(mode, meshWidth, meshHeight, numFrames);
    
    return 0;
}