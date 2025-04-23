#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

  // Número de elementos en los arreglos (puedes probar 1000000)

// Función Kernel que se ejecuta en el Device
__global__ void suma_vectores(float *c, float *a, float *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Índice del hilo
    if (idx < N) {
        c[idx] = a[idx] + b[idx];  // Suma de los elementos de los vectores
    }
}

int main(void) {
    // Punteros a arreglos en el Host
    float *a_h, *b_h, *c_h;
    // Punteros a arreglos en el Device
    float *a_d, *b_d, *c_d;
    int N=10000;
    size_t size = N * sizeof(float);  // Tamaño de los arreglos

    // Pedimos memoria en el Host
    a_h = (float *)malloc(size);
    b_h = (float *)malloc(size);
    c_h = (float *)malloc(size);

    // Inicializamos los arreglos a y b en el Host
    for (int i = 0; i < N; i++) {
        a_h[i] = (float)i;  // Asignamos valores
        b_h[i] = (float)(i + 1);
    }

    printf("\nArreglo a:\n");
    for (int i = 0; i < N; i++) printf("%f ", a_h[i]);
    printf("\nArreglo b:\n");
    for (int i = 0; i < N; i++) printf("%f ", b_h[i]);

    // Pedimos memoria en el Device
    cudaMalloc((void **) &a_d, size);
    cudaMalloc((void **) &b_d, size);
    cudaMalloc((void **) &c_d, size);

    // Pasamos los arreglos a y b del Host al Device
    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

    // Realizamos el cálculo en el Device
    int block_size = 8;
    int n_blocks = (N + block_size - 1) / block_size;  // Calculo de los bloques
    suma_vectores<<<n_blocks, block_size>>>(c_d, a_d, b_d, N);  // Llamada al kernel

    // Pasamos el resultado del Device al Host
    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

    // Mostrar el resultado
    printf("\nResultado c:\n");
    for (int i = 0; i < N; i++) printf("%f ", c_h[i]);
    printf("\n");

    // Liberamos la memoria del Host
    free(a_h);
    free(b_h);
    free(c_h);

    // Liberamos la memoria del Device
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    return 0;
}
