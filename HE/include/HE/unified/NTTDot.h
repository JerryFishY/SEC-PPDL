#pragma once
#include <cstdint>
#include <cstddef>

class DModulus;

namespace HE::unified {
// Kernel arguments own the pointer list, so they need no asynchronous host staging.
struct NTTProductBatch {
    static constexpr unsigned capacity = 32;
    const std::uint64_t *ciphertexts[capacity]{};
    const std::uint64_t *plaintexts[capacity]{};
    unsigned count = 0;
};

// All operands use the same BFV chain level and NTT representation.
// Layout is [ciphertext component][RNS limb][coefficient].
void AccumulateNTTProducts(const NTTProductBatch &batch, const DModulus *moduli,
                           std::size_t degree, std::size_t limbs, std::size_t components,
                           std::uint64_t *output, bool accumulate);
} // namespace HE::unified
