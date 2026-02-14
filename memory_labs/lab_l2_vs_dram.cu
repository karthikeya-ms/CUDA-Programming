#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static void check(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(e));
    std::exit(1);
  }
}

// Repeatedly read x[] and accumulate into out[] to prevent optimization.
// Uses grid-stride to cover the array.
__global__ void repeated_read(const float* __restrict__ x,
                              float* __restrict__ out,
                              int n, int repeats) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  float acc = 0.0f;

  // repeats passes over the same working set
  for (int r = 0; r < repeats; r++) {
    for (int i = tid; i < n; i += stride) {
      acc += x[i];
    }
    // stop compiler from being too clever across repeats
    __syncthreads();
  }

  if (tid < n) out[tid] = acc;
}

static void run_case(const char* name, size_t bytes, int repeats) {
  int dev = 0;
  int l2Bytes = 0;
  check(cudaDeviceGetAttribute(&l2Bytes, cudaDevAttrL2CacheSize, dev),
        "cudaDeviceGetAttribute(L2)");

  int n = (int)(bytes / sizeof(float));
  float *d_x = nullptr, *d_out = nullptr;
  check(cudaMalloc(&d_x, bytes), "cudaMalloc d_x");
  check(cudaMalloc(&d_out, bytes), "cudaMalloc d_out");

  // initialize x (doesn't matter what values are)
  check(cudaMemset(d_x, 1, bytes), "cudaMemset d_x");

  // choose a reasonable launch
  int block = 256;
  int smCount = 0;
  check(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, dev),
        "cudaDeviceGetAttribute(SM count)");
  int grid = smCount * 8; // enough warps to hide latency

  // warm-up kernel (and also fills caches a bit)
  repeated_read<<<grid, block>>>(d_x, d_out, n, 1);
  check(cudaGetLastError(), "warmup launch");
  check(cudaDeviceSynchronize(), "warmup sync");

  cudaEvent_t s, e;
  check(cudaEventCreate(&s), "event create");
  check(cudaEventCreate(&e), "event create");

  check(cudaEventRecord(s), "event record s");
  repeated_read<<<grid, block>>>(d_x, d_out, n, repeats);
  check(cudaGetLastError(), "timed launch");
  check(cudaEventRecord(e), "event record e");
  check(cudaEventSynchronize(e), "event sync e");

  float ms = 0.0f;
  check(cudaEventElapsedTime(&ms, s, e), "event elapsed");

  double totalBytesRead = (double)bytes * (double)repeats;
  double GBs = (totalBytesRead / 1e9) / (ms / 1000.0);

  printf("%s\n", name);
  printf("  Working set: %.2f MB  (L2 is %.2f MB)\n",
         bytes / (1024.0 * 1024.0), l2Bytes / (1024.0 * 1024.0));
  printf("  Repeats: %d\n", repeats);
  printf("  Time: %.3f ms\n", ms);
  printf("  Effective read bandwidth: %.1f GB/s\n\n", GBs);

  check(cudaEventDestroy(s), "event destroy");
  check(cudaEventDestroy(e), "event destroy");
  check(cudaFree(d_x), "cudaFree d_x");
  check(cudaFree(d_out), "cudaFree d_out");
}

int main() {
  // Your L2 is 32MB. We pick:
  //  - 24MB (likely fits in L2 reasonably well after first pass)
  //  - 256MB (definitely does not fit; DRAM dominated)
  const int repeats = 50;

  run_case("CASE A (expected L2-heavy after first pass)", 24ull * 1024 * 1024, repeats);
  run_case("CASE B (expected DRAM-dominated)",           256ull * 1024 * 1024, repeats);

  return 0;
}
