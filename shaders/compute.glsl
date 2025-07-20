#version 430

layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) buffer PositionBuffer {
    vec4 positions[];
};

layout(std430, binding = 1) buffer OldPositionBuffer {
    vec4 oldPositions[];
};

layout(std430, binding = 2) buffer ForceBuffer {
    vec4 forces[];
};

layout(std430, binding = 3) buffer SpringBuffer {
    ivec4 springs[]; // vertex1, vertex2, type, padding
};

layout(std430, binding = 4) buffer SpringDataBuffer {
    vec4 springData[]; // restLength, stiffness, padding, padding
};

uniform int numSprings;
uniform int meshWidth;
uniform int meshHeight;
uniform float deltaTime;
uniform float mass;
uniform int phase; // 0: clear forces, 1: calculate forces, 2: integrate

void main() {
    uint index = gl_GlobalInvocationID.y * meshWidth + gl_GlobalInvocationID.x;
    
    if (gl_GlobalInvocationID.x >= meshWidth || gl_GlobalInvocationID.y >= meshHeight) {
        return;
    }
    
    if (phase == 0) {
        // Limpiar fuerzas
        forces[index] = vec4(0.0, 0.0, 0.0, 0.0);
    }
    else if (phase == 1) {
        // Calcular fuerzas de resortes
        for (int i = 0; i < numSprings; ++i) {
            ivec4 spring = springs[i];
            vec4 springProps = springData[i];
            
            if (spring.x == int(index) || spring.y == int(index)) {
                vec3 pos1 = positions[spring.x].xyz;
                vec3 pos2 = positions[spring.y].xyz;
                
                vec3 diff = pos2 - pos1;
                float currentLength = length(diff);
                
                if (currentLength > 0.0) {
                    float displacement = currentLength - springProps.x; // restLength
                    vec3 direction = diff / currentLength;
                    vec3 springForce = direction * displacement * springProps.y; // stiffness
                    
                    if (spring.x == int(index)) {
                        atomicAdd(forces[index].x, springForce.x);
                        atomicAdd(forces[index].y, springForce.y);
                        atomicAdd(forces[index].z, springForce.z);
                    } else {
                        atomicAdd(forces[index].x, -springForce.x);
                        atomicAdd(forces[index].y, -springForce.y);
                        atomicAdd(forces[index].z, -springForce.z);
                    }
                }
            }
        }
    }
    else if (phase == 2) {
        // Integración de Verlet
        vec3 currentPos = positions[index].xyz;
        vec3 oldPos = oldPositions[index].xyz;
        vec3 force = forces[index].xyz;
        
        // Calcular aceleración
        vec3 acceleration = force / mass;
        
        float dt2 = deltaTime * deltaTime;
        
        // Integración de Verlet: x(t+dt) = 2*x(t) - x(t-dt) + a(t)*dt^2
        vec3 newPos = 2.0 * currentPos - oldPos + acceleration * dt2;
        
        // Actualizar posiciones
        oldPositions[index] = vec4(currentPos, 1.0);
        positions[index] = vec4(newPos, 1.0);
        
        // Aplicar restricciones (fijar bordes)
        if (gl_GlobalInvocationID.y == 0 || gl_GlobalInvocationID.y == meshHeight - 1) {
            positions[index].z = 0.0;
        }
    }
}