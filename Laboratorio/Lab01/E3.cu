#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

const int N = 1000;  // Puedes probar con más, como 1000000
vector<float> A(N), B(N), C(N);

// Kernel CUDA
__global__ void vectorAddGPU(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// Inicializa A y B con valores aleatorios
void CrearVectores(vector<float>& A, vector<float>& B) {
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        A[i] = (rand() % 1000) / 10.0f;
        B[i] = (rand() % 1000) / 10.0f;
    }
}

// Ejecuta la suma en GPU y mide tiempo
float TiempoGPU(vector<float>& A, vector<float>& B, vector<float>& C) {
    float *d_A, *d_B, *d_C;
    size_t size = N * sizeof(float);

    // Reservar memoria en la GPU
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copiar datos desde host a GPU
    cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), size, cudaMemcpyHostToDevice);

    // Configurar kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    auto start = std::chrono::high_resolution_clock::now();

    // Ejecutar kernel
    vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;

    // Copiar resultado de vuelta al host
    cudaMemcpy(C.data(), d_C, size, cudaMemcpyDeviceToHost);

    // Liberar memoria en GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return duration.count(); // tiempo en milisegundos
}

int main() {
    CrearVectores(A, B);

    float time = TiempoGPU(A, B, C);

    for (int i = 0; i < 1000; i++) {
        cout << "A" << i << ": " << A[i]
             << " + B" << i << ": " << B[i]
             << " = C" << i << ": " << C[i] << endl;
    }

    cout << "El tiempo de ejecución en GPU fue: " << time << " ms" << endl;

    return 0;
}
