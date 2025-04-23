#include <iostream>
#include <cuda_runtime.h>

#define TILE_WIDTH 16  // Define el tamaño de los bloques de hilos

// Kernel de multiplicación de matrices
__global__ void matrixMultiply(float *A, float *B, float *C, int m, int n, int p) {
    // Índices de los hilos
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Índice de fila
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Índice de columna

    // Comprobamos si el hilo está dentro de los límites de la matriz resultante
    if (row < m && col < p) {
        float value = 0;
        // Sumamos los productos de la multiplicación de A y B
        for (int i = 0; i < n; ++i) {
            value += A[row * n + i] * B[i * p + col];
        }
        C[row * p + col] = value;
    }
}

int main() {
    int m = 512; // Número de filas en A y C
    int n = 512; // Número de columnas en A y filas en B
    int p = 512; // Número de columnas en B y C

    size_t sizeA = m * n * sizeof(float);
    size_t sizeB = n * p * sizeof(float);
    size_t sizeC = m * p * sizeof(float);

    // Reserva memoria en el host
    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);

    // Inicializar las matrices A y B con valores
    for (int i = 0; i < m * n; i++) {
        h_A[i] = 1.0f; // Puedes cambiar estos valores a lo que necesites
    }
    for (int i = 0; i < n * p; i++) {
        h_B[i] = 1.0f;
    }

    // Reserva memoria en el dispositivo
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    // Copia las matrices del host a la memoria del dispositivo
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Definir las dimensiones de los bloques y la rejilla
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((p + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH);

    // Llamar al kernel de multiplicación de matrices
    matrixMultiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, p);

    // Copia el resultado desde la memoria del dispositivo al host
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Libera la memoria del dispositivo
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Mostrar el resultado de la matriz C (opcional)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            std::cout << h_C[i * p + j] << " ";
        }
        std::cout << std::endl;
    }

    // Liberar la memoria del host
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
