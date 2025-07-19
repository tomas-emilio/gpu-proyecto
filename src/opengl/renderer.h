class OpenGLRenderer {
    // Inicializar contexto OpenGL y buffers
    void initialize(int windowWidth, int windowHeight);
    
    // Crear VAO, VBO para la malla
    void setupMeshBuffers(const Mesh& mesh);
    
    // Renderizar frame
    void render(const Mesh& mesh);
    
    // Configurar cámara y matrices
    void setupCamera(const glm::mat4& view, const glm::mat4& projection);
    
    // Obtener VBO handle para interoperabilidad
    unsigned int getVBO() const;
};