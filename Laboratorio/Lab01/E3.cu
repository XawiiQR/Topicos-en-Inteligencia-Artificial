#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

int main() {
    int N = 1000000; // Cambiar tamaño según sea necesario (> 1000)
    size_t size = N * sizeof(float);

    // Host vectors
    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];

    // Inicializar vectores
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // Device vectors
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copiar datos al dispositivo
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Configurar ejecución en GPU
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Medir tiempo
    auto start = std::chrono::high_resolution_clock::now();

    // Ejecutar kernel en GPU
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    // Copiar resultado de vuelta al host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Mostrar tiempo
    std::cout << "Tiempo de ejecución en GPU: " << duration_ms.count() << " ms\n";

    // Liberar memoria
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
