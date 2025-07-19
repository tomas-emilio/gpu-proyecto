class Application {
    // Inicializar ventana GLFW y contextos
    void initialize();
    
    // Loop principal de renderizado
    void run();
    
    // Cambiar entre modos (CPU/CUDA/Compute Shader)
    void switchSimulationMode(SimulationMode mode);
    
    // Manejar input del usuario
    void handleInput();
    
    // Medir y mostrar métricas de rendimiento
    void updatePerformanceMetrics();
    
    // Limpiar recursos
    void cleanup();
};