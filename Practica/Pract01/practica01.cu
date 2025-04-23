#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 24  
#define BLOCK_SIZE 16 

// Función Kernel que se ejecuta en el Device para la suma de vectores
__global__ void suma_vectores(float *c, float *a, float *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  
    if (idx < N) {
        c[idx] = a[idx] + b[idx];  
    }
}

// Función Kernel que se ejecuta en el Device para la multiplicación de matrices
__global__ void Multiplica_Matrices_GM(float *C, float *A, float *B, int nfil, int ncol) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int idy = blockIdx.y * blockDim.y + threadIdx.y;  
    int index = idy * ncol + idx;

    if (idy < nfil && idx < ncol) {
        float sum = 0.0f;
        for (int k = 0; k < ncol; k++) {
            sum += A[idy * ncol + k] * B[k * ncol + idx];
        }
        C[index] = sum;
    }
}

int div_up(int x, int y) {
    return (x + y - 1) / y;  
}

int main(void) {

    // Parte 1: Información sobre los dispositivos CUDA

    int noOfDevices;
    cudaGetDeviceCount(&noOfDevices);

    cudaDeviceProp prop;
    for (int i = 0; i < noOfDevices; i++) {
        cudaGetDeviceProperties(&prop, i);

        printf("Device Name:\t%s\n", prop.name);
        printf("Total global memory:\t%ld\n", prop.totalGlobalMem);
        printf("No. of SMs:\t%d\n", prop.multiProcessorCount);
        printf("Shared memory / SM:\t%ld\n", prop.sharedMemPerBlock);
        printf("Registers / SM:\t%d\n", prop.regsPerBlock);
    }

    // Parte 2: Suma de Vectores
    float *a_h, *b_h, *c_h;
    float *a_d, *b_d, *c_d;
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


    

    
    // Parte 3: Multiplicación de matrices
    int nfil = 5;  
    int ncol = 5; 
    int N_m = nfil * ncol;  

    size_t size_m = N_m * sizeof(float);  
    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;

    A_h = (float *)malloc(size_m);
    B_h = (float *)malloc(size_m);
    C_h = (float *)malloc(size_m);

    for (int i = 0; i < nfil; i++) {
        for (int j = 0; j < ncol; j++) {
            A_h[i * ncol + j] = 1.0f;
            B_h[i * ncol + j] = 2.0f;
        }
    }

    
    cudaMalloc((void **) &A_d, size_m);
    cudaMalloc((void **) &B_d, size_m);
    cudaMalloc((void **) &C_d, size_m);

    
    cudaMemcpy(A_d, A_h, size_m, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size_m, cudaMemcpyHostToDevice);

    
    dim3 block_size_m(BLOCK_SIZE, BLOCK_SIZE);
    dim3 n_blocks_m(div_up(ncol, BLOCK_SIZE), div_up(nfil, BLOCK_SIZE));
    Multiplica_Matrices_GM<<<n_blocks_m, block_size_m>>>(C_d, A_d, B_d, nfil, ncol);

    
    cudaMemcpy(C_h, C_d, size_m, cudaMemcpyDeviceToHost);

    
    printf("\nMatriz A:\n");
    for (int i = 0; i < nfil; i++) {
        for (int j = 0; j < ncol; j++) {
            printf("%2.2f ", A_h[i * ncol + j]);
        }
        printf("\n");
    }

    printf("\nMatriz B:\n");
    for (int i = 0; i < nfil; i++) {
        for (int j = 0; j < ncol; j++) {
            printf("%2.2f ", B_h[i * ncol + j]);
        }
        printf("\n");
    }

    printf("\nMatriz C (Resultado):\n");
    for (int i = 0; i < nfil; i++) {
        for (int j = 0; j < ncol; j++) {
            printf("%2.2f ", C_h[i * ncol + j]);
        }
        printf("\n");
    }

    
    free(A_h);
    free(B_h);
    free(C_h);

    
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;
}
