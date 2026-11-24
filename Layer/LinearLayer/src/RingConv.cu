#include "RingConvGPU.h"
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>

namespace LinearLayer {
namespace {
void Check(cudaError_t err) {
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}
struct Geometry {
    int size, ci, co, kernel, stride, padding, side, depth, spatial;
};
__global__ void ExtractPatches(Geometry g, const uint32_t *input, uint32_t *patches) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x, k = blockIdx.y;
    if (p >= g.spatial) return;
    const int y = (p / g.side) * g.stride + (k / g.kernel) % g.kernel - g.padding;
    const int x = (p % g.side) * g.stride + k % g.kernel - g.padding;
    const int c = k / (g.kernel * g.kernel);
    patches[size_t(k) * g.spatial + p] = (unsigned(y) < g.size && unsigned(x) < g.size)
        ? input[(size_t(c) * g.size + y) * g.size + x] : 0;
}
constexpr int tile = 16;
__global__ void Multiply(Geometry g, const uint32_t *weights, const uint32_t *patches, uint32_t *output) {
    __shared__ uint32_t a[tile][tile], b[tile][tile];
    const int x = threadIdx.x, y = threadIdx.y;
    const int c = blockIdx.y * tile + y, p = blockIdx.x * tile + x;
    uint32_t sum = 0;
    for (int k = 0; k < g.depth; k += tile) {
        a[y][x] = c < g.co && k+x < g.depth ? weights[size_t(c) * g.depth + k+x] : 0;
        b[y][x] = p < g.spatial && k+y < g.depth ? patches[size_t(k+y) * g.spatial + p] : 0;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < tile; ++i) sum += a[y][i] * b[i][x];
        __syncthreads();
    }
    if (c < g.co && p < g.spatial) output[size_t(c) * g.spatial + p] = sum;
}
} // namespace

struct RingConvGPU {
    Geometry g;
    uint32_t *input = nullptr, *patches = nullptr, *weights = nullptr, *output = nullptr;
    ~RingConvGPU() { cudaFree(input); cudaFree(patches); cudaFree(weights); cudaFree(output); }
};
std::shared_ptr<RingConvGPU> MakeRingConvGPU(size_t size, size_t ci, size_t co, size_t kernel,
                                             size_t stride, size_t padding, size_t side,
                                             const uint32_t *weights) {
    const size_t depth = ci * kernel * kernel, spatial = side * side;
    if (depth > 65535 || co > 65535 * tile || spatial > std::numeric_limits<int>::max() - 256 ||
        size > std::numeric_limits<int>::max() || stride > std::numeric_limits<int>::max() ||
        padding > std::numeric_limits<int>::max())
        throw std::invalid_argument("Ring convolution geometry exceeds CUDA grid limits");
    auto state = std::make_shared<RingConvGPU>();
    state->g = {int(size),int(ci),int(co),int(kernel),int(stride),int(padding),int(side),int(depth),int(spatial)};
    Check(cudaMalloc(&state->input, ci * size * size * sizeof(uint32_t)));
    Check(cudaMalloc(&state->weights, co * depth * sizeof(uint32_t)));
    Check(cudaMalloc(&state->patches, depth * spatial * sizeof(uint32_t)));
    Check(cudaMalloc(&state->output, co * spatial * sizeof(uint32_t)));
    Check(cudaMemcpy(state->weights, weights, co * depth * sizeof(uint32_t), cudaMemcpyHostToDevice));
    return state;
}
void RunRingConvGPU(RingConvGPU &state, const uint32_t *input, uint32_t *output) {
    const auto &g = state.g;
    const auto stream = cudaStreamPerThread;
    Check(cudaMemcpyAsync(state.input, input, size_t(g.ci) * g.size * g.size * sizeof(uint32_t),
                          cudaMemcpyHostToDevice, stream));
    ExtractPatches<<<dim3((g.spatial+255)/256, g.depth),256,0,stream>>>(g, state.input, state.patches);
    Check(cudaGetLastError());
    Multiply<<<dim3((g.spatial+tile-1)/tile, (g.co+tile-1)/tile),dim3(tile,tile),0,stream>>>(
        g, state.weights, state.patches, state.output);
    Check(cudaGetLastError());
    Check(cudaMemcpyAsync(output, state.output, size_t(g.co) * g.spatial * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost, stream));
    Check(cudaStreamSynchronize(stream));
}
} // namespace LinearLayer
