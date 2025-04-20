#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

const int N = 1000;  
vector<float> A(N), B(N), C(N);


__global__ void vectorAddGPU(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}


void CrearVectores(vector<float>& A, vector<float>& B) {
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        A[i] = (rand() % 1000) / 10.0f;
        B[i] = (rand() % 1000) / 10.0f;
    }
}


float TiempoGPU(vector<float>& A, vector<float>& B, vector<float>& C) {
    float *d_A, *d_B, *d_C;
    size_t size = N * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    auto start = std::chrono::high_resolution_clock::now();

    vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;

    cudaMemcpy(C.data(), d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return duration.count(); 
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
