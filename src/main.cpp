#include <iostream>
#include <string>
#include "application.h"

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [cpu|cuda|shader] [mesh_size]" << std::endl;
    std::cout << "  cpu    - Run CPU simulation" << std::endl;
    std::cout << "  cuda   - Run CUDA simulation" << std::endl;
    std::cout << "  shader - Run Compute Shader simulation" << std::endl;
    std::cout << "  mesh_size - Size of the mesh (default: 40)" << std::endl;
    std::cout << std::endl;
    std::cout << "Interactive Controls:" << std::endl;
    std::cout << "  Left click and drag - Apply force to tissue" << std::endl;
    std::cout << "  P key - Print performance statistics" << std::endl;
    std::cout << "  ESC key - Exit application" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=== Interactive Tissue Deformation Simulation ===" << std::endl;
    
    // Valores por defecto
    std::string simType = "cpu";
    int meshSize = 40;
    
    // Procesar argumentos
    if (argc > 1) {
        simType = argv[1];
        if (simType != "cpu" && simType != "cuda" && simType != "shader") {
            std::cerr << "Error: Invalid simulation type '" << simType << "'" << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (argc > 2) {
        meshSize = std::atoi(argv[2]);
        if (meshSize < 10 || meshSize > 200) {
            std::cerr << "Error: Mesh size must be between 10 and 200" << std::endl;
            return 1;
        }
    }
    
    if (argc == 1 || (argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help"))) {
        printUsage(argv[0]);
        return 0;
    }
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Simulation type: " << simType << std::endl;
    std::cout << "  Mesh size: " << meshSize << "x" << meshSize 
              << " (" << (meshSize * meshSize) << " vertices)" << std::endl;
    std::cout << std::endl;
    
    // Crear y ejecutar aplicación
    Application app;
    
    if (!app.initialize(simType, meshSize)) {
        std::cerr << "Failed to initialize application" << std::endl;
        return 1;
    }
    
    app.run();
    
    return 0;
}