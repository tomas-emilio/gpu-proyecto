struct Spring {
    int vertex1, vertex2;
    float restLength;
    float stiffness;
    SpringType type; // STRUCTURAL, SHEAR, VIRTUAL
};

class PhysicsModel {
    // Crear resortes estructurales, de corte y virtuales
    void generateSprings(const Mesh& mesh);
    
    // Calcular fuerzas de resortes
    glm::vec3 calculateSpringForces(int vertexId, const Mesh& mesh);
    
    // Aplicar restricciones de borde
    void applyConstraints(Mesh& mesh);
};