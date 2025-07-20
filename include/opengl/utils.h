#pragma once
#include <GL/glew.h>
#include <string>

class OpenGLUtils {
public:
    static std::string loadShaderSource(const std::string& filepath);
    static GLuint compileShader(const std::string& source, GLenum shaderType);
    static GLuint createShaderProgram(const std::string& vertexPath, const std::string& fragmentPath);
    static GLuint createComputeShaderProgram(const std::string& computePath);
    static void checkGLError(const std::string& operation);
};