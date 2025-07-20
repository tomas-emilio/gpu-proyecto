# Simulación Interactiva de Tejidos Biológicos

Simulación de deformación de tejidos con tres implementaciones: CPU, CUDA y OpenGL Compute Shaders.

## Instalación de Dependencias

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install build-essential cmake git
sudo apt install libgl1-mesa-dev libglfw3-dev libglew-dev libglm-dev
sudo apt install nvidia-driver-525 cuda-toolkit-12-0
```

## Compilación

```bash
mkdir build
cd build
cmake ..
make
```

## Ejecución

```bash
cd build
# Sintaxis
./src/simulacion [cpu|cuda|shader] [tamaño_malla]

# Ejemplos
./src/simulacion cpu 40
./src/simulacion cuda 80
./src/simulacion shader 60
```

## Controles

### Mouse
- **Click izquierdo + arrastrar**: Aplicar fuerza

### Teclado
- **Q/A**: Rigidez estructural +/-
- **W/S**: Rigidez de corte +/-
- **E/D**: Rigidez virtual +/-
- **R/F**: Amortiguación +/-
- **SPACE**: Reset tejido
- **BACKSPACE**: Reset parámetros
- **P**: Estadísticas
- **H**: Ayuda
- **ESC**: Salir