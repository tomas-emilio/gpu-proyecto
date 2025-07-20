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
}

void printInteractiveControls() {
    std::cout << "=== INTERACTIVE CONTROLS ===" << std::endl;
    std::cout << std::endl;
    std::cout << "Mouse Controls:" << std::endl;
    std::cout << "  Left click + drag - Apply force to deform tissue" << std::endl;
    std::cout << std::endl;
    std::cout << "Tissue Parameters (Real-time adjustment):" << std::endl;
    std::cout << "  Q / A - Increase / Decrease structural stiffness" << std::endl;
    std::cout << "  W / S - Increase / Decrease shear stiffness" << std::endl;
    std::cout << "  E / D - Increase / Decrease virtual stiffness" << std::endl;
    std::cout << "  R / F - Increase / Decrease damping factor" << std::endl;
    std::cout << std::endl;
    std::cout << "Simulation Controls:" << std::endl;
    std::cout << "  SPACE     - Reset tissue to initial state" << std::endl;
    std::cout << "  BACKSPACE - Reset all parameters to defaults" << std::endl;
    std::cout << "  P         - Print performance statistics" << std::endl;
    std::cout << "  H         - Show help (this menu)" << std::endl;
    std::cout << "  ESC       - Exit application" << std::endl;
    std::cout << std::endl;
    std::cout << "Note: Parameter changes are applied immediately" << std::endl;
    std::cout << "      and affect the physical behavior of the tissue." << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "==========================================================" << std::endl;
    std::cout << "    Interactive Tissue Deformation Simulation" << std::endl;
    std::cout << "    GPU-Accelerated Biological Tissue Modeling" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << std::endl;
    
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
        printInteractiveControls();
        return 0;
    }
    
    // Mostrar configuración
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Simulation Backend: " << simType;
    if (simType == "cpu") std::cout << " (Sequential CPU processing)";
    else if (simType == "cuda") std::cout << " (CUDA GPU parallel processing)";
    else if (simType == "shader") std::cout << " (OpenGL Compute Shaders)";
    std::cout << std::endl;
    
    std::cout << "  Mesh Resolution: " << meshSize << "x" << meshSize 
              << " (" << (meshSize * meshSize) << " vertices)" << std::endl;
    
    int totalSprings = (meshSize * (meshSize-1)) * 2 + // structural
                       (meshSize-1) * (meshSize-1) * 2 + // shear  
                       meshSize * (meshSize-2) * 2; // virtual
    std::cout << "  Spring Count: ~" << totalSprings << " springs" << std::endl;
    std::cout << std::endl;
    
    
    // Crear y ejecutar aplicación
    std::cout << "Initializing simulation..." << std::endl;
    Application app;
    
    if (!app.initialize(simType, meshSize)) {
        std::cerr << "Failed to initialize application" << std::endl;
        return 1;
    }
    
    std::cout << std::endl;
    printInteractiveControls();
    
    std::cout << "Starting interactive simulation..." << std::endl;
    std::cout << "Press 'H' during simulation for help" << std::endl;
    std::cout << "==========================================================" << std::endl;
    
    app.run();
    
    std::cout << std::endl;
    std::cout << "Simulation ended" << std::endl;
    
    return 0;
}