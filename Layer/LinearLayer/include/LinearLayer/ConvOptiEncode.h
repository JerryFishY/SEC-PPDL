#pragma once
#include <LinearLayer/Conv.h>
#include <LinearOperator/OptiEncode.h>
#include <memory>
#include <unordered_map>

namespace LinearLayer {
struct RingConvGPU;
struct OptiEncodeOptions {
    size_t threads = 1;
    size_t channels_per_block = 0;
    size_t tile_output_size = 0;
    size_t baby_steps = 0;
    bool add_bias = true;
    size_t output_mod_switches = 1;
    // Predivide the reconstructed input ciphertext in R_q before any Galois
    // operation (BumbleBee ZeroGap normalization). Weights remain unscaled.
    bool ciphertext_inverse_normalization = false;
    bool cuda_graph = false;
    bool compact_output = false; // Send only the masked output coefficients needed by the client.
    bool depthwise = false; // Multiplier one, compact [channels, 1, kernel, kernel] weights.
};
struct OptiEncodeStats {
    size_t input_ciphertexts = 0, output_ciphertexts = 0;
    size_t weight_plaintexts = 0, nonzero_weight_plaintexts = 0;
    double offline_seconds = 0, compute_seconds = 0;
};

// BFV coefficient encoding on the host. Input/output are additive field shares.
// Bias is a server-owned field residue and is included once by default.
// Instances may be reused sequentially; simultaneous calls on one instance/IO
// channel are not supported. Worker threads never read/write the network.
class Conv2DOptiEncode : public Conv2D {
public:
    Conv2DOptiEncode(uint64_t feature_size, uint64_t stride, uint64_t padding,
                    const Tensor<uint64_t> &weight, const Tensor<uint64_t> &bias,
                    HE::HEEvaluator *he, OptiEncodeOptions options = {});
    Conv2DOptiEncode(uint64_t feature_size, uint64_t in_channels, uint64_t out_channels,
                    uint64_t kernel_size, uint64_t stride, HE::HEEvaluator *he,
                    OptiEncodeOptions options = {});
    // Two-party, input-independent preprocessing for exactly one subsequent call.
    // Requires a power-of-two plaintext modulus and one active BFV data prime.
    void PrepareInput(bool include_product = false);
    Tensor<uint64_t> operator()(Tensor<uint64_t> &input) override;
    const Operator::OptiEncodeLayout &Layout() const { return layout_; }
    const OptiEncodeStats &Stats() const { return stats_; }
    // The small union covers all automatically selected layouts for this N.
    static std::vector<uint32_t> GaloisElements(size_t degree);
private:
    struct PreparedInput {
        Tensor<uint64_t> mask, product_share;
        bool include_product = false;
        Tensor<UnifiedCiphertext> ciphertexts;
    };
    std::unique_ptr<PreparedInput> prepared_;
    Tensor<uint64_t> OpenMaskedInput(const Tensor<uint64_t> &input, const PreparedInput &state);
    void Normalize(seal::Ciphertext &ct, uint64_t inverse) const;
    uint64_t InputInverse() const;
    Tensor<UnifiedCiphertext> ApplyPreparedInput(Tensor<uint64_t> &correction, PreparedInput &state);
    void AddRingProduct(const Tensor<uint64_t> &correction, Tensor<uint64_t> &output);
    std::vector<uint32_t> ring_weights_;
#ifdef USE_HE_GPU
    std::shared_ptr<RingConvGPU> ring_gpu_;
#endif
    void ValidateCompactTransport() const;
    Tensor<uint64_t> ExtractMaskedOutput(Tensor<UnifiedCiphertext> &evaluated);
    void AddBias(Tensor<uint64_t> &output) const;
    OptiEncodeOptions options_;
    Operator::OptiEncodeLayout layout_;
    OptiEncodeStats stats_;
    std::vector<uint8_t> nonzero_weights_;
    std::unordered_map<uint32_t, int> rotation_steps_;
    Tensor<UnifiedPlaintext> PackWeight() override;
    Tensor<uint64_t> PackActivation(Tensor<uint64_t> &input) override;
    Tensor<UnifiedCiphertext> HECompute(const Tensor<UnifiedPlaintext> &weights,
                                       Tensor<UnifiedCiphertext> &input) override;
#ifdef USE_HE_GPU
    struct GraphState;
    std::shared_ptr<GraphState> graph_;
    void PrepareGraph();
    void EvaluateGPU(const Tensor<UnifiedPlaintext> &weights, const Tensor<UnifiedCiphertext> &inputs,
                     Tensor<UnifiedCiphertext> &output);
    Tensor<UnifiedCiphertext> HEComputeGPU(const Tensor<UnifiedPlaintext> &weights,
                                         Tensor<UnifiedCiphertext> &input);
    void ApplyAutomorphismGPU(UnifiedCiphertext &ciphertext, uint32_t element) const;
#endif
    Tensor<uint64_t> DepackResult(Tensor<uint64_t> &output) override;
    void ApplyAutomorphism(seal::Evaluator &evaluator, seal::Ciphertext &ciphertext,
                           uint32_t element) const;
};
} // namespace LinearLayer
