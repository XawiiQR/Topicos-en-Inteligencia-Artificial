#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


__global__ void suma_vectores(float *c, float *a, float *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  
    if (idx < N) {
        c[idx] = a[idx] + b[idx];  
    }
}

int main(void) {
    t
    float *a_h, *b_h, *c_h;
    
    float *a_d, *b_d, *c_d;
    int N=10;
    size_t size = N * sizeof(float);  

    
    a_h = (float *)malloc(size);
    b_h = (float *)malloc(size);
    c_h = (float *)malloc(size);

    
    for (int i = 0; i < N; i++) {
        a_h[i] = (float)i;  
        b_h[i] = (float)(i + 1);
    }

    printf("\nArreglo a:\n");
    for (int i = 0; i < N; i++) printf("%f ", a_h[i]);
    printf("\nArreglo b:\n");
    for (int i = 0; i < N; i++) printf("%f ", b_h[i]);

    
    cudaMalloc((void **) &a_d, size);
    cudaMalloc((void **) &b_d, size);
    cudaMalloc((void **) &c_d, size);

    
    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

    
    int block_size = 8;
    int n_blocks = (N + block_size - 1) / block_size;  
    suma_vectores<<<n_blocks, block_size>>>(c_d, a_d, b_d, N); 

    
    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

    
    printf("\nResultado c:\n");
    for (int i = 0; i < N; i++) printf("%f ", c_h[i]);
    printf("\n");

    
    free(a_h);
    free(b_h);
    free(c_h);

    
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    return 0;
}
