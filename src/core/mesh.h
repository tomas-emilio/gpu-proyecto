class Mesh {
    // Inicializar malla con dimensiones width x height
    void initialize(int width, int height, float spacing);
    
    // Obtener/establecer posición de vértice
    glm::vec3& getVertex(int x, int y);
    void setVertex(int x, int y, const glm::vec3& pos);
    
    // Obtener índices para rendering
    std::vector<unsigned int>& getIndices();
    
    // Aplicar fuerza en posición específica
    void applyForce(int x, int y, const glm::vec3& force);
};