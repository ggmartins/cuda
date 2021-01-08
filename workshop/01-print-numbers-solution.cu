#include <stdio.h>

__global__ void printNumber(int number)
{
  printf("%d\n", number);
}

int main()
{
  cudaStream_t stream[5];       // CUDA streams are of type `cudaStream_t`.
  for(int i = 0; i < 5; i++)
     cudaStreamCreate(&stream[i]); 
  for (int i = 0; i < 5; ++i)
  {
    printNumber<<<1, 1, 0, stream[i]>>>(i);
  }
  cudaDeviceSynchronize();
  for (int i = 0; i < 5; ++i)
     cudaStreamDestroy(stream[i]);
}

