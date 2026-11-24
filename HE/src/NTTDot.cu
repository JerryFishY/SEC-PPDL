#include <HE/unified/NTTDot.h>
#include <phantom/uintmodmath.cuh>
#include <stdexcept>

namespace HE::unified {
namespace {
__global__ void SumNTTProducts(NTTProductBatch batch, const DModulus *moduli,
                               size_t degree, uint64_t *output, bool accumulate) {
    const size_t coefficient = blockIdx.x * blockDim.x + threadIdx.x;
    if (coefficient >= degree) return;
    const size_t plain_index = blockIdx.y * degree + coefficient;
    const size_t cipher_index = blockIdx.z * gridDim.y * degree + plain_index;
    const auto &modulus = moduli[blockIdx.y];
    uint64_t sum = accumulate ? output[cipher_index] : 0;
    for (unsigned term = 0; term < batch.count; ++term) {
        const auto product = phantom::arith::multiply_and_barrett_reduce_uint64(
            batch.ciphertexts[term][cipher_index], batch.plaintexts[term][plain_index],
            modulus.value(), modulus.const_ratio());
        sum = phantom::arith::add_uint64_uint64_mod(sum, product, modulus.value());
    }
    output[cipher_index] = sum;
}
} // namespace

void AccumulateNTTProducts(const NTTProductBatch &batch, const DModulus *moduli,
                           size_t degree, size_t limbs, size_t components,
                           uint64_t *output, bool accumulate) {
    constexpr unsigned threads = 128;
    const dim3 blocks((degree + threads - 1) / threads, limbs, components);
    SumNTTProducts<<<blocks, threads, 0, cudaStreamPerThread>>>(batch, moduli, degree, output, accumulate);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
} // namespace HE::unified
