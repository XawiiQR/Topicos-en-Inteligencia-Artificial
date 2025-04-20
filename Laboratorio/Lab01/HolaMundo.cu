#include<stdio.h>
__global__ void hello(void)
{
    printf("GPU: Hola Mundo!\n");
}
int main(int argc,char **argv)
{
    
    hello<<<1,10>>>();
    cudaDeviceReset();
    return 0;
}