#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {
template <typename scalar_t>
__device__ __forceinline__ scalar_t gelu(scalar_t z) {
  return z * normcdff(z);
}

template <typename scalar_t>
__global__ void fff_cuda_forward_kernel(
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> x,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> in_weight,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> in_bias,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> out_weight,
    const unsigned int width,
    const unsigned int depth,
    const unsigned int n_nodes
  ) {
  extern __shared__ char cache_raw[]; // Raw shared memory space
  scalar_t* cache = reinterpret_cast<scalar_t*>(cache_raw); // Cast to desired type when used


  // compute which row of inputs we're dealing with
  const int cache_index = threadIdx.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch = blockIdx.y;

  int current_node = 0;
  for (int current_depth = 0; current_depth <= depth; ++current_depth) {

    // compute 1024 accumulations (one for each thread)
    scalar_t temp = 0;
    int vid = tid;
    while (vid < width) {
      temp += x[batch][vid] * in_weight[current_node][vid];
      vid += blockDim.x;
    }
    cache[cache_index] = temp;
    __syncthreads();

    // reduce the accumulations into a single value
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
      if (cache_index < i) {
        cache[cache_index] += cache[cache_index + i];
      }
      __syncthreads();
    }
    
    //bias and activation
    if (cache_index == 0) {
      cache[0] += in_bias[current_node];
      cache[1] = gelu(cache[0]);
    }
    __syncthreads();

    // compute the output contribution due to the current node
    vid = tid;
    while(vid < width) {
      output[batch][vid] += cache[1] * out_weight[current_node][vid];
      vid += blockDim.x;
    }

    // decide where to move to
    current_node = (current_node<<1) + 1 + (cache[0] > 0 ? 1 : 0);
  }
}
} // namespace

torch::Tensor fff_cuda_forward(
	torch::Tensor x,
	torch::Tensor in_weight,
	torch::Tensor in_bias,
	torch::Tensor out_weight,
	const unsigned int width,
	const unsigned int depth,
	const unsigned int parallel_size,
	const unsigned int n_nodes
) {

  auto output = torch::empty(
    {x.size(0), width},
    torch::TensorOptions()
      .dtype(torch::kFloat32)
      .device(x.device())
  );

  const int batch_size = x.size(0);
  //blockx adds extra level of parallelism combining accumulations across blocks
  //each thread does N/(threads*blockx) accumulations
  //after this there will be blockx accumulations left to do
  const int blockx = 1; //tuneable 
  const int blocky = batch_size;

  const int threads = 1024;
  const dim3 blocks = dim3(blockx, blocky);
  const int shared_size = threads * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES(in_weight.type(), "fff_forward_cuda", ([&] {
    fff_cuda_forward_kernel<scalar_t><<<blocks, threads, shared_size>>>(
        output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        x.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        in_weight.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        in_bias.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
        out_weight.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        width,
        depth,
        n_nodes
    );
  }));

  cudaError_t err;
  err = cudaGetLastError();
  if (cudaSuccess != err) {
      fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
  }

  cudaError_t cudaStatus;
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
  }

  return output;
}
