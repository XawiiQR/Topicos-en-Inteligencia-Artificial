#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16  // Tamaño del bloque (ajustable)

int div_up(int x, int y) {
    return (x + y - 1) / y;  // Redondear hacia arriba
}

// Función Kernel que se ejecuta en el Device
__global__ void Multiplica_Matrices_GM(float *C, float *A, float *B, int nfil, int ncol) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Índice de la fila
    int idy = blockIdx.y * blockDim.y + threadIdx.y;  // Índice de la columna
    int index = idy * ncol + idx;

    if (idy < nfil && idx < ncol) {
        float sum = 0.0f;
        for (int k = 0; k < ncol; k++) {
            sum += A[idy * ncol + k] * B[k * ncol + idx];
        }
        C[index] = sum;
    }
}

int main(void) {
    // Punteros a matrices en el Host
    float *A_h, *B_h, *C_h;
    // Punteros a matrices en el Device
    float *A_d, *B_d, *C_d;
    
    int nfil = 5;  // Número de filas
    int ncol = 5;  // Número de columnas
    int N = nfil * ncol;  // Número de elementos en la matriz

    size_t size = N * sizeof(float);  // Tamaño de los arreglos

    // GPU Time
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Pedimos memoria en el Host
    A_h = (float *)malloc(size);
    B_h = (float *)malloc(size);
    C_h = (float *)malloc(size);

    // Inicializamos las matrices A y B en el Host
    for (int i = 0; i < nfil; i++) {
        for (int j = 0; j < ncol; j++) {
            A_h[i * ncol + j] = 1.0f;  // Valor inicial de A
            B_h[i * ncol + j] = 2.0f;  // Valor inicial de B
        }
    }

    // Pedimos memoria en el Device
    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);

    // Pasamos las matrices A y B del Host al Device
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // Realizamos el cálculo en el Device
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 n_blocks(div_up(ncol, BLOCK_SIZE), div_up(nfil, BLOCK_SIZE));
    cudaEventRecord(start);
    Multiplica_Matrices_GM<<<n_blocks, block_size>>>(C_d, A_d, B_d, nfil, ncol);
    cudaEventRecord(stop);
    
    // Esperamos a que termine el kernel
    cudaEventSynchronize(stop);
    
    // Pasamos el resultado del Device al Host
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    // Mostrar el resultado
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

    printf("\nMatriz C:\n");
    for (int i = 0; i < nfil; i++) {
        for (int j = 0; j < ncol; j++) {
            printf("%2.2f ", C_h[i * ncol + j]);
        }
        printf("\n");
    }

    // Calculamos el tiempo de ejecución en la GPU
    cudaEventElapsedTime(&time, start, stop);
    printf("Tiempo de ejecución: %3.1f ms\n", time);

    // Liberamos la memoria del Host
    free(A_h);
    free(B_h);
    free(C_h);

    // Liberamos la memoria del Device
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;
}
