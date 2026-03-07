#include "init_tensor_cuda.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cstdio>

// ==================== 内部工具 ====================

namespace {

// 内部管理的默认流（懒初始化）
static cudaStream_t g_default_stream = nullptr;
static bool         g_stream_initialized = false;

cudaStream_t get_default_stream() {
    if (!g_stream_initialized) {
        cudaStreamCreate(&g_default_stream);
        g_stream_initialized = true;
    }
    return g_default_stream;
}

void cleanup_default_stream() {
    if (g_stream_initialized && g_default_stream) {
        cudaStreamDestroy(g_default_stream);
        g_default_stream = nullptr;
        g_stream_initialized = false;
    }
}

// 自动注册清理函数
struct StreamCleanup {
    ~StreamCleanup() { cleanup_default_stream(); }
};
static StreamCleanup g_cleanup;

// __device__ __forceinline__ float curand_uniform_range(curandState* state, float min, float max) {
//     return min + curand_uniform(state) * (max - min);
// 简单的 Xorshift32 算法，速度极快
__device__ __forceinline__ uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// 将 uint32 转换为 [0, 1) 之间的 float
__device__ __forceinline__ float uint2float_uniform(uint32_t x) {
    // 使用 24 位精度，避免除法，使用乘法加速
    return (float)(x >> 8) * 5.96046448e-8f; 
}
// 对应原来的 curand_uniform_range
__device__ __forceinline__ float xorshift_uniform_range(uint32_t* state, float min, float max) {
    return min + uint2float_uniform(xorshift32(state)) * (max - min);
}


// ==================== CUDA Kernels ====================

__global__ void kernel_init_f32(
    float* __restrict__ data,
    size_t n,
    float min,
    float max,
    unsigned long long seed_offset)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    // 初始化状态：确保不为 0 (xorshift 状态为 0 会卡死)
    uint32_t state = (uint32_t)(seed_offset + idx + 1); 

    for (size_t i = idx; i < n; i += stride) {
        data[i] = xorshift_uniform_range(&state, min, max);
    }
}

__global__ void kernel_init_f16(
    half* __restrict__ data,
    size_t n,
    float min,
    float max,
    unsigned long long seed_offset)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    uint32_t state = (uint32_t)(seed_offset + idx + 1);

    for (size_t i = idx; i < n; i += stride) {
        float val = xorshift_uniform_range(&state, min, max);
        data[i] = __float2half(val);
    }
}

// kernel_init_f16_vec 也做类似修改，注意 state 初始化逻辑保持一致
__global__ void kernel_init_f16_vec(
    half* __restrict__ data,
    size_t n,
    float min,
    float max,
    unsigned long long seed_offset)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    size_t n_pairs = n / 2;
    
    // 这里为了保持随机性独立，state 种子稍微错开
    uint32_t state = (uint32_t)(seed_offset + idx * 2 + 1); 

    for (size_t i = idx; i < n_pairs; i += stride) {
        float v0 = xorshift_uniform_range(&state, min, max);
        float v1 = xorshift_uniform_range(&state, min, max);
        reinterpret_cast<half2*>(data)[i] = __halves2half2(__float2half(v0), __float2half(v1));
    }
    if (n % 2 == 1 && idx == 0) {
        float val = xorshift_uniform_range(&state, min, max);
        data[n - 1] = __float2half(val);
    }
}

// ==================== Launch 配置 ====================

void get_launch_config(size_t n, dim3* grid, dim3* block) {
    constexpr size_t BLOCK_SIZE = 256;
    block->x = BLOCK_SIZE;
    block->y = block->z = 1;
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    size_t max_blocks = prop.maxGridSize[0];
    size_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    grid->x = (blocks < max_blocks) ? blocks : max_blocks;
    grid->y = grid->z = 1;
}

// ==================== 核心初始化函数 ====================

cudaError_t init_tensor_cuda_internal(
    ggml_tensor*     tensor,
    float            min,
    float            max,
    cudaStream_t     stream,
    unsigned long long seed) 
{
    if (!tensor || !tensor->data) {
        return cudaErrorInvalidValue;
    }
    cudaPointerAttributes attrs;
    cudaError_t ptr_err = cudaPointerGetAttributes(&attrs, tensor->data);
    if (ptr_err != cudaSuccess || attrs.type == cudaMemoryTypeHost) {
        fprintf(stderr, "[init_tensor_cuda] Error: Tensor data is not on CUDA device!\n");
        return cudaErrorInvalidValue;
    }
    size_t nels = ggml_nelements(tensor);
    if (nels == 0) return cudaSuccess;
    
    dim3 grid, block;
    get_launch_config(nels, &grid, &block);
    
    // 使用默认种子
    if (seed == 0) seed = 123456789ULL;
    
    switch (tensor->type) {
        case GGML_TYPE_F32: {
            float* data = static_cast<float*>(tensor->data);
            kernel_init_f32<<<grid, block, 0, stream>>>(data, nels, min, max, seed);
            break;
        }
        case GGML_TYPE_F16: {
            half* data = static_cast<half*>(tensor->data);
            if (nels % 2 == 0 && (reinterpret_cast<uintptr_t>(data) % 4 == 0)) {
                kernel_init_f16_vec<<<grid, block, 0, stream>>>(data, nels, min, max, seed);
            } else {
                kernel_init_f16<<<grid, block, 0, stream>>>(data, nels, min, max, seed);
            }
            break;
        }
        default:
            fprintf(stderr, "[init_tensor_cuda] Unsupported type: %d (only F32 and F16 supported)\n", tensor->type);
            return cudaErrorInvalidValue;
    }
    
    cudaError_t err = cudaGetLastError();
    return err;
}

} // anonymous namespace

// ==================== 对外 API 实现 ====================

int init_tensor_cuda_simple(
    ggml_tensor* tensor,
    float        min,
    float        max,
    unsigned long long seed) 
{
    cudaStream_t stream = get_default_stream();
    cudaError_t err = init_tensor_cuda_internal(tensor, min, max, stream, seed);
    
    if (err == cudaSuccess) {
        cudaStreamSynchronize(stream);
        return 0;
    }
    
    return -1;
}

cudaError_t init_tensors_cuda_simple(
    ggml_tensor** tensors,
    size_t        n_tensors,
    float         min,
    float         max,
    unsigned long long seed) 
{
    if (!tensors || n_tensors == 0) return cudaErrorInvalidValue;
    
    cudaStream_t stream = get_default_stream();
    
    for (size_t i = 0; i < n_tensors; ++i) {
        unsigned long long tensor_seed = seed + i * 100003ULL;
        if (tensor_seed == 0) tensor_seed = 123456789ULL + i;
        
        cudaError_t err = init_tensor_cuda_internal(tensors[i], min, max, stream, tensor_seed);
        if (err != cudaSuccess) return err;
    }
    
    cudaStreamSynchronize(stream);
    return cudaSuccess;
}

cudaError_t init_tensor_cuda_async(
    ggml_tensor*  tensor,
    float         min,
    float         max,
    cudaStream_t* stream,
    unsigned long long seed) 
{
    cudaStream_t s = get_default_stream();
    cudaError_t err = init_tensor_cuda_internal(tensor, min, max, s, seed);
    
    if (stream) {
        *stream = s;
    }
    
    return err;
}

cudaError_t init_tensor_cuda_sync(void) {
    if (g_stream_initialized && g_default_stream) {
        return cudaStreamSynchronize(g_default_stream);
    }
    return cudaSuccess;
}

bool init_tensor_cuda_support_type(ggml_type type){
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16;
}