#include <stdio.h>


int main()
{
  int *host_a, *device_a;        // Define host-specific and device-specific arrays.
  cudaMalloc(&device_a, size);   // `device_a` is immediately available on the GPU.
  cudaMallocHost(&host_a, size); // `host_a` is immediately available on CPU, and is page-locked, or pinned.

  initializeOnHost(host_a, N);   // No CPU page faulting since memory is already allocated on the host.

  // `cudaMemcpy` takes the destination, source, size, and a CUDA-provided variable for the direction of the copy.
  cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);

  kernel<<<blocks, threads, 0, someStream>>>(device_a, N);

  // `cudaMemcpy` can also copy data from device to host.
  cudaMemcpy(host_a, device_a, size, cudaMemcpyDeviceToHost);

  verifyOnHost(host_a, N);

  cudaFree(device_a);
  cudaFreeHost(host_a); // Free pinned memory like this.
}
