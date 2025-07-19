class CPUSimulation {
    // Inicializar simulación en CPU
    void initialize(int meshWidth, int meshHeight);
    
    // Actualizar un frame de simulación
    void update(float deltaTime);
    
    // Aplicar interacción del usuario
    void handleMouseInteraction(float x, float y, float force);
    
    // Obtener datos para rendering
    const Mesh& getMesh() const;
    
    // Medir tiempo de cálculo
    double getLastFrameTime() const;
};