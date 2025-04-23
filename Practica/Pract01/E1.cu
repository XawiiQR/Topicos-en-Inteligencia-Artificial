#include <stdio.h>

int main()
{
    int noOfDevices;
    /* get no. of device */
    cudaGetDeviceCount (&noOfDevices);

    cudaDeviceProp prop;
    for (int i = 0; i < noOfDevices; i++)
    {
        /* get device properties */
        cudaGetDeviceProperties (&prop, i);

        printf("Device Name:\t%s\n", prop.name);
        printf("Total global memory:\t%ld\n", prop.totalGlobalMem);
        printf("No. of SMs:\t%d\n", prop.multiProcessorCount);
        printf("Shared memory / SM:\t%ld\n", prop.sharedMemPerBlock);
        printf("Registers / SM:\t%d\n", prop.regsPerBlock);
    }

    return 1;
}
