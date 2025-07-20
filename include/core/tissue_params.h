#pragma once
#include <iostream>

struct TissueParams {
    float structuralStiffness = 50.0f;  // Rigidez resortes estructurales
    float shearStiffness = 25.0f;       // Rigidez resortes de corte
    float virtualStiffness = 15.0f;     // Rigidez resortes virtuales
    float damping = 0.98f;              // Factor de amortiguación
    float mass = 1.0f;                  // Masa de cada punto
    
    void increaseStructural() { structuralStiffness += 5.0f; }
    void decreaseStructural() { structuralStiffness = std::max(1.0f, structuralStiffness - 5.0f); }
    
    void increaseShear() { shearStiffness += 2.5f; }
    void decreaseShear() { shearStiffness = std::max(1.0f, shearStiffness - 2.5f); }
    
    void increaseVirtual() { virtualStiffness += 2.0f; }
    void decreaseVirtual() { virtualStiffness = std::max(1.0f, virtualStiffness - 2.0f); }
    
    void increaseDamping() { damping = std::min(0.99f, damping + 0.01f); }
    void decreaseDamping() { damping = std::max(0.90f, damping - 0.01f); }
    
    void reset() {
        structuralStiffness = 50.0f;
        shearStiffness = 25.0f;
        virtualStiffness = 15.0f;
        damping = 0.98f;
        mass = 1.0f;
    }
    
    void print() const {
        std::cout << "=== Tissue Parameters ===" << std::endl;
        std::cout << "Structural: " << structuralStiffness << std::endl;
        std::cout << "Shear: " << shearStiffness << std::endl;
        std::cout << "Virtual: " << virtualStiffness << std::endl;
        std::cout << "Damping: " << damping << std::endl;
        std::cout << "Mass: " << mass << std::endl;
    }
};

// Singleton global para parámetros
extern TissueParams g_tissueParams;