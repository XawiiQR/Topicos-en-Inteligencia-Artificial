__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main() {
    
    int N = 1024;
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    
    vectorAdd<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N);

    
    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

   
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
