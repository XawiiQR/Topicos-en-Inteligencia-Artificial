#include<stdio.h>
__global__ void hello(void)
{
    printf("GPU: Hello Javi!\n");
}
int main(int argc,char **argv)
{
    printf("CPU: Hello!\n");
    hello<<<1,10>>>();
    cudaDeviceReset();
    return 0;
}