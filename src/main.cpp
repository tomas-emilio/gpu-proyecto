#include <iostream>
#include <string>
#include "application.h"

void printUsage(const char* programName) {
    std::cout << "Uso: " << programName << " [cpu|cuda|shader] [mesh_size]" << std::endl;
    std::cout << "  cpu    - Correr simulacion CPU" << std::endl;
    std::cout << "  cuda   - Correr simulacion CUDA" << std::endl;
    std::cout << "  shader - Correr simulacion con Shader" << std::endl;
    std::cout << "  mesh_size - Tamaño del mesh (default: 40)" << std::endl;
    std::cout << std::endl;
}

void printInteractiveControls() {
    std::cout << "=== CONTROLES ===" << std::endl;
    std::cout << std::endl;
    std::cout << "Mouse:" << std::endl;
    std::cout << "  Left click + drag - Aplicar fuerza para deformar el tejido" << std::endl;
    std::cout << std::endl;
    std::cout << "Parámetros del Tejido (ajuste en tiempo real):" << std::endl;
    std::cout << "  Q / A - Subir/Bajar rigidez estructural" << std::endl;
    std::cout << "  W / S - Subir/Bajar rigidez shear" << std::endl;
    std::cout << "  E / D - Subir/Bajar rigidez virtual" << std::endl;
    std::cout << "  R / F - Subir/Bajar factor de amortiguamiento" << std::endl;
    std::cout << std::endl;
    std::cout << "Controles:" << std::endl;
    std::cout << "  SPACE     - Resetear tejido al estado inicial" << std::endl;
    std::cout << "  BACKSPACE - Resetear todos los parametros por defecto" << std::endl;
    std::cout << "  P         - Print estadisticas de desempeño" << std::endl;
    std::cout << "  H         - help" << std::endl;
    std::cout << "  ESC       - Exit" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "==========================================================" << std::endl;
    std::cout << "    Simulación Interactiva" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << std::endl;
    
    //valores por defecto
    std::string simType = "cpu";
    int meshSize = 40;
    
    // Procesar argumentos
    if (argc > 1) {
        simType = argv[1];
        if (simType != "cpu" && simType != "cuda" && simType != "shader") {
            std::cerr << "Error: tipo de simulacion no valido '" << simType << "'" << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (argc > 2) {
        meshSize = std::atoi(argv[2]);
        if (meshSize < 10 || meshSize > 200) {
            std::cerr << "Error: Mesh size debe ser entre 10 y 200" << std::endl;
            return 1;
        }
    }
    
    if (argc == 1 || (argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help"))) {
        printUsage(argv[0]);
        printInteractiveControls();
        return 0;
    }
    
    // Mostrar configuración
    std::cout << "Configuracion:" << std::endl;
    std::cout << "  Simulacion Backend: " << simType;
    if (simType == "cpu") std::cout << " (Modelo secuencial en procesamiento)";
    else if (simType == "cuda") std::cout << " (Modelo cuda en procesamiento)";
    else if (simType == "shader") std::cout << " (OpenGL Compute Shaders en procesamiento)";
    std::cout << std::endl;
    
    std::cout << " Resolucion mesh: " << meshSize << "x" << meshSize 
              << " (" << (meshSize * meshSize) << " vertices)" << std::endl;
    
    int totalSprings = (meshSize * (meshSize-1)) * 2 + // structural
                       (meshSize-1) * (meshSize-1) * 2 + // shear  
                       meshSize * (meshSize-2) * 2; // virtual
    std::cout << "  Resortes: ~" << totalSprings << std::endl;
    std::cout << std::endl;
    
    
    // Crear y ejecutar aplicación
    std::cout << "Iniciando simulacion..." << std::endl;
    Application app;
    
    if (!app.initialize(simType, meshSize)) {
        std::cerr << "Simulación fallada" << std::endl;
        return 1;
    }
    
    std::cout << std::endl;
    printInteractiveControls();
    
    std::cout << "Empezando simulacion..." << std::endl;
    std::cout << "Presiona 'H' durante la simulacion para ayuda" << std::endl;
    std::cout << "==========================================================" << std::endl;
    
    app.run();
    
    std::cout << std::endl;
    std::cout << "Simulacion terminada" << std::endl;
    
    return 0;
}