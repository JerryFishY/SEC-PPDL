#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>

namespace LinearLayer {
struct RingConvGPU;
std::shared_ptr<RingConvGPU> MakeRingConvGPU(size_t size, size_t ci, size_t co, size_t kernel,
                                             size_t stride, size_t padding, size_t output_size,
                                             const uint32_t *weights, bool depthwise);
void RunRingConvGPU(RingConvGPU &conv, const uint32_t *input, uint32_t *output);
}
