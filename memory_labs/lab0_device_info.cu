#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static void check(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(e));
    std::exit(1);
  }
}

static int getAttr(cudaDeviceAttr attr, int dev) {
  int v = 0;
  check(cudaDeviceGetAttribute(&v, attr, dev), "cudaDeviceGetAttribute");
  return v;
}

int main() {
  int count = 0;
  check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  if (count == 0) {
    printf("No CUDA devices found.\n");
    return 0;
  }

  int dev = 0;
  cudaDeviceProp p{};
  check(cudaGetDeviceProperties(&p, dev), "cudaGetDeviceProperties");

  // Attributes (CUDA 13-safe for removed/deprecated cudaDeviceProp fields)
  const int memClockKHz = getAttr(cudaDevAttrMemoryClockRate, dev);              // kHz
  const int memBusWidthBits = getAttr(cudaDevAttrGlobalMemoryBusWidth, dev);    // bits
  const int l2Bytes = getAttr(cudaDevAttrL2CacheSize, dev);                     // bytes
  const int smCount = getAttr(cudaDevAttrMultiProcessorCount, dev);
  const int warpSize = getAttr(cudaDevAttrWarpSize, dev);
  const int maxThreadsPerBlock = getAttr(cudaDevAttrMaxThreadsPerBlock, dev);

  printf("Device %d: %s\n", dev, p.name);
  printf("Compute Capability: %d.%d\n", p.major, p.minor);

  printf("Global Memory (from cudaDeviceProp): %.2f GB\n",
         (double)p.totalGlobalMem / (1024.0 * 1024 * 1024));

  printf("L2 Cache (attr): %.2f MB\n", (double)l2Bytes / (1024.0 * 1024));
  printf("Memory Clock (attr): %.0f MHz\n", memClockKHz / 1000.0);
  printf("Memory Bus Width (attr): %d bits\n", memBusWidthBits);

  printf("SM count: %d\n", smCount);
  printf("Warp size: %d\n", warpSize);
  printf("Max threads per block: %d\n", maxThreadsPerBlock);

  // Very rough peak estimate (assumes “DDR x2”; actual GDDR effective rate differs)
  double memClockHz = (double)memClockKHz * 1000.0;
  double busBytes = (double)memBusWidthBits / 8.0;
  double roughGBs = memClockHz * busBytes * 2.0 / 1e9;
  printf("Rough bandwidth hint (DDR x2 assumption): %.1f GB/s\n", roughGBs);

  return 0;
}
