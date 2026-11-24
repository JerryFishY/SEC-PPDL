#include <LinearLayer/ConvOptiEncode.h>
#include <Utils/ParallelFor.h>
#ifdef USE_HE_GPU
#include "RingConvGPU.h"
#endif

namespace LinearLayer {
void Conv2DOptiEncode::AddRingProduct(const Tensor<uint64_t> &delta, Tensor<uint64_t> &output) {
    const auto &l = layout_;
    const size_t spatial = l.output_height * l.output_width;
    const size_t depth = l.input_channels * l.kernel * l.kernel;
    const uint64_t mask = HE->plain_mod - 1;
    std::vector<uint32_t> input(delta.data().begin(), delta.data().end());
    std::vector<uint32_t> sums(output.size());
#ifdef USE_HE_GPU
    if (ring_gpu_) {
        RunRingConvGPU(*ring_gpu_, input.data(), sums.data());
    } else
#endif
    {
        // Each patch column corresponds to an output position. Unsigned
        // 32-bit products and accumulation preserve every low ring bit.
        std::vector<uint32_t> patches(depth * spatial);
        Utils::ParallelFor(depth, options_.threads, [&](size_t k, size_t) {
            const size_t c = k / (l.kernel * l.kernel);
            const size_t ky = k / l.kernel % l.kernel, kx = k % l.kernel;
            for (size_t y = 0; y < l.output_height; ++y) {
                const auto iy = y * l.stride + ky - l.padding;
                if (iy >= l.height) continue;
                for (size_t x = 0; x < l.output_width; ++x) {
                    const auto ix = x * l.stride + kx - l.padding;
                    if (ix < l.width)
                        patches[k * spatial + y * l.output_width + x] = input[(c * l.height + iy) * l.width + ix];
                }
            }
        });
        Utils::ParallelFor(l.output_channels, options_.threads, [&](size_t c, size_t) {
            auto *sum = sums.data() + c * spatial;
            for (size_t k = 0; k < depth; ++k) {
                const uint32_t w = ring_weights_[c * depth + k];
                if (!w) continue;
                const auto *row = patches.data() + k * spatial;
                for (size_t p = 0; p < spatial; ++p) sum[p] += w * row[p];
            }
        });
    }
    for (size_t i = 0; i < sums.size(); ++i) output(i) = (output(i) + sums[i]) & mask;
}
} // namespace LinearLayer
