#include <LinearLayer/ConvOptiEncode.h>
#include <Utils/ParallelFor.h>
#include <algorithm>
#include <chrono>
#include <set>
#include <limits>
#include <stdexcept>

namespace LinearLayer {
namespace {
using Clock = std::chrono::steady_clock;
const Tensor<uint64_t> &ValidateWeights(size_t size, size_t stride, size_t padding,
    const Tensor<uint64_t> &weights, const Tensor<uint64_t> &bias, HE::HEEvaluator *he,
    const OptiEncodeOptions &options) {
    if (!he || !he->context || !he->galoisKeys || !he->publicKeys)
        throw std::invalid_argument("OptiEncode requires an initialized HE context and exchanged keys");
    const auto &shape = weights.shape();
    if (!size || !stride || shape.size() != 4 || !shape[0] || !shape[1] || !shape[2] || shape[2] != shape[3] ||
        (options.depthwise && shape[1] != 1) || bias.shape() != std::vector<size_t>{shape[0]} ||
        padding > (std::numeric_limits<size_t>::max() - size) / 2 ||
        size + 2 * padding < shape[2])
        throw std::invalid_argument("Invalid OptiEncode convolution dimensions or bias shape");
    if ((!options.ciphertext_inverse_normalization && !(he->plain_mod & 1)) ||
        he->plain_mod > (uint64_t(1) << 61))
        throw std::invalid_argument("Even plaintext moduli require ciphertext inverse normalization");
    return weights;
}
Tensor<uint64_t> DefaultWeights(size_t ci, size_t co, size_t k, HE::HEEvaluator *he, bool depthwise) {
    if (!he || !ci || !co || !k || (depthwise && ci != co)) throw std::invalid_argument("Invalid OptiEncode convolution parameters");
    Tensor<uint64_t> weights({co, depthwise ? 1 : ci, k, k});
    if (he->server) weights.randomize(16);
    return weights;
}
std::vector<std::unique_ptr<seal::Evaluator>> Evaluators(const seal::SEALContext &context, size_t count) {
    std::vector<std::unique_ptr<seal::Evaluator>> result;
    for (size_t i = 0; i < count; ++i) result.emplace_back(std::make_unique<seal::Evaluator>(context));
    return result;
}
}
Conv2DOptiEncode::Conv2DOptiEncode(uint64_t size, uint64_t stride, uint64_t padding,
    const Tensor<uint64_t> &weights, const Tensor<uint64_t> &bias, HE::HEEvaluator *he,
    OptiEncodeOptions options)
    : Conv2D(size, stride, padding, ValidateWeights(size, stride, padding, weights, bias, he, options), bias, he), options_(options),
      layout_(he->polyModulusDegree, size, size, options.depthwise ? out_channels : in_channels, out_channels, kernel_size,
              stride, padding, options.channels_per_block, options.tile_output_size, options.depthwise) {
    in_channels = layout_.input_channels;
    if (options_.compact_output) ValidateCompactTransport();
    if (!options_.threads) throw std::invalid_argument("OptiEncode threads must be positive");
    if (options_.baby_steps) {
        if (layout_.channels % options_.baby_steps) throw std::invalid_argument("Baby steps must divide the channel block");
        layout_.baby_steps = options_.baby_steps;
        layout_.giant_steps = layout_.channels / options_.baby_steps;
    }
    stats_.input_ciphertexts = layout_.InputCiphertexts();
    stats_.output_ciphertexts = layout_.OutputCiphertexts();
    uint32_t element = 1;
    for (size_t step = 0; step < he->polyModulusDegree / 2; ++step) {
        rotation_steps_[element] = static_cast<int>(step);
        element = (uint64_t(element) * 3) % (2 * he->polyModulusDegree);
    }
    if (he->server) {
        const auto start = Clock::now();
        weight_pt = PackWeight();
#ifdef USE_HE_GPU
        if (options_.cuda_graph && HE->Backend() == DEVICE) PrepareGraph();
#endif
        stats_.offline_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    }
}
Conv2DOptiEncode::Conv2DOptiEncode(uint64_t size, uint64_t ci, uint64_t co, uint64_t k,
    uint64_t stride, HE::HEEvaluator *he, OptiEncodeOptions options)
    : Conv2DOptiEncode(size, stride, (k - 1) / 2, DefaultWeights(ci, co, k, he, options.depthwise),
                       Tensor<uint64_t>({co}), he, options) {}
std::vector<uint32_t> Conv2DOptiEncode::GaloisElements(size_t n) {
    std::set<uint32_t> elements;
    // All powers are inexpensive to store at the small supported block sizes.
    // Including them also supports an explicitly requested baby-step split.
    for (size_t c = 2; c <= (2 * n) / c; c *= 2) {
        const uint64_t base = 1 + 2 * n / c;
        for (size_t b = 1; b < c; b *= 2)
            elements.insert(static_cast<uint32_t>(Operator::OptiEncodePowerMod(base, b, 2 * n)));
    }
    return {elements.begin(), elements.end()};
}
Tensor<UnifiedPlaintext> Conv2DOptiEncode::PackWeight() {
    const size_t count = layout_.WeightInputBlocks() * layout_.output_blocks * layout_.channels;
    Tensor<UnifiedPlaintext> packed({layout_.WeightInputBlocks(), layout_.output_blocks, layout_.channels}, HOST);
    nonzero_weights_.resize(count, 0);
    auto evaluators = Evaluators(HE->context->hcontext(), options_.threads);
    Utils::ParallelFor(count, options_.threads, [&](size_t index, size_t worker) {
        const size_t slot = index / (layout_.output_blocks * layout_.channels);
        const size_t output_block = index / layout_.channels % layout_.output_blocks;
        const size_t step = index % layout_.channels;
        const auto coefficients = layout_.PackWeight(
            weight, layout_.SourceBlock(slot, output_block), output_block, step,
            HE->plain_mod, !options_.ciphertext_inverse_normalization);
        if (std::none_of(coefficients.begin(), coefficients.end(), [](uint64_t v) { return v != 0; })) return;
        nonzero_weights_[index] = 1;
        auto &plain = packed(index).hplain();
        plain.resize(layout_.degree);
        std::copy(coefficients.begin(), coefficients.end(), plain.data());
        evaluators[worker]->transform_to_ntt_inplace(plain, HE->context->hcontext().first_parms_id());
    });
    stats_.weight_plaintexts = count;
    stats_.nonzero_weight_plaintexts = std::count(nonzero_weights_.begin(), nonzero_weights_.end(), uint8_t(1));
    if (HE->Backend() == DEVICE) {
        for (size_t index = 0; index < count; ++index)
            if (nonzero_weights_[index]) packed(index).to_device(*HE->context);
        HE->evaluator->sync();
    }
    return packed;
}
void Conv2DOptiEncode::ApplyAutomorphism(seal::Evaluator &evaluator, seal::Ciphertext &ciphertext,
                                        uint32_t element) const {
    if (element == 1) return;
    const auto &keys = HE->galoisKeys->hgalois();
    if (keys.has_key(element)) {
        evaluator.apply_galois_inplace(ciphertext, element, keys);
    } else {
        // Preserve compatibility with HEEvaluator's existing default key set.
        const auto step = rotation_steps_.find(element);
        if (step == rotation_steps_.end()) throw std::invalid_argument("Missing OptiEncode Galois key");
        evaluator.rotate_rows_inplace(ciphertext, step->second, keys);
    }
}
Tensor<uint64_t> Conv2DOptiEncode::PackActivation(Tensor<uint64_t> &input) {
    return layout_.PackActivation(input, HE->plain_mod, options_.threads);
}
Tensor<UnifiedCiphertext> Conv2DOptiEncode::HECompute(const Tensor<UnifiedPlaintext> &weights,
                                                   Tensor<UnifiedCiphertext> &inputs) {
#ifdef USE_HE_GPU
    if (HE->Backend() == DEVICE) return HEComputeGPU(weights, inputs);
#endif
    Tensor<UnifiedCiphertext> output({layout_.TileCount(), layout_.output_blocks}, HOST);
    if (!HE->server) return output;
    const auto start = Clock::now();
    auto evaluators = Evaluators(HE->context->hcontext(), options_.threads);
    std::vector<std::unique_ptr<seal::Encryptor>> encryptors;
    for (size_t i = 0; i < options_.threads; ++i)
        encryptors.emplace_back(std::make_unique<seal::Encryptor>(HE->context->hcontext(), *HE->publicKeys));
    const uint32_t giant_element = static_cast<uint32_t>(Operator::OptiEncodePowerMod(
        layout_.generator, layout_.baby_steps, 2 * layout_.degree));
    const size_t B = layout_.baby_steps, G = layout_.giant_steps;
    auto prepare_babies = [&](size_t tile, size_t i, seal::Evaluator &evaluator, seal::Ciphertext *babies) {
        auto current = inputs({tile, i}).hcipher();
        for (size_t a = 0; a < B; ++a) {
            if (a) ApplyAutomorphism(evaluator, current, layout_.generator);
            if (a + 1 == B) babies[a] = std::move(current);
            else babies[a] = current;
            evaluator.transform_to_ntt_inplace(babies[a]);
        }
    };
    auto partial = [&](size_t o, size_t giant, const seal::Ciphertext *babies,
                       seal::Evaluator &evaluator, seal::Ciphertext &sum) {
        bool have_sum = false;
        seal::Ciphertext product;
        for (size_t slot = 0; slot < layout_.WeightInputBlocks(); ++slot) {
            const size_t first = (slot * layout_.output_blocks + o) * layout_.channels + giant * B;
            for (size_t a = 0; a < B; ++a) {
                if (!nonzero_weights_[first + a]) continue;
                evaluator.multiply_plain(babies[slot * B + a], weights(first + a).hplain(),
                                         have_sum ? product : sum);
                if (have_sum) evaluator.add_inplace(sum, product);
                have_sum = true;
            }
        }
        if (have_sum) evaluator.transform_from_ntt_inplace(sum);
        return have_sum;
    };
    auto reduce_giants = [&](size_t tile, size_t o, size_t worker, auto get_partial) {
        auto &evaluator = *evaluators[worker];
        seal::Ciphertext result;
        bool have_result = false;
        for (size_t giant = G; giant-- > 0;) {
            if (have_result) ApplyAutomorphism(evaluator, result, giant_element);
            seal::Ciphertext sum;
            if (get_partial(giant, sum)) {
                if (have_result) evaluator.add_inplace(result, sum);
                else { result = std::move(sum); have_result = true; }
            }
        }
        if (!have_result) encryptors[worker]->encrypt_zero(result);
        for (size_t level = 0; level < options_.output_mod_switches; ++level) {
            if (!HE->context->hcontext().get_context_data(result.parms_id())->next_context_data()) break;
            evaluator.mod_switch_to_next_inplace(result);
        }
        output({tile, o}) = UnifiedCiphertext(std::move(result));
    };
    if (layout_.depthwise && output.size() >= options_.threads) {
        // Tile/block pairs are independent. Keep only a worker's B babies and its running sum.
        Utils::ParallelFor(output.size(), options_.threads, [&](size_t task, size_t worker) {
            const size_t tile = task / layout_.output_blocks, o = task % layout_.output_blocks;
            auto &evaluator = *evaluators[worker];
            std::vector<seal::Ciphertext> babies(B);
            prepare_babies(tile, o, evaluator, babies.data());
            reduce_giants(tile, o, worker, [&](size_t giant, seal::Ciphertext &sum) {
                return partial(o, giant, babies.data(), evaluator, sum);
            });
        });
    } else for (size_t tile = 0; tile < layout_.TileCount(); ++tile) {
        // Share dense babies across outputs, or parallelize giants when blocks alone cannot fill the CPU.
        std::vector<seal::Ciphertext> babies(layout_.input_blocks * B);
        Utils::ParallelFor(layout_.input_blocks, options_.threads, [&](size_t i, size_t worker) {
            prepare_babies(tile, i, *evaluators[worker], babies.data() + i * B);
        });
        std::vector<seal::Ciphertext> sums(layout_.output_blocks * G);
        std::vector<uint8_t> nonzero(sums.size());
        Utils::ParallelFor(sums.size(), options_.threads, [&](size_t task, size_t worker) {
            const size_t o = task / G;
            nonzero[task] = partial(o, task % G, babies.data() + layout_.SourceBlock(0, o) * B,
                                    *evaluators[worker], sums[task]);
        });
        Utils::ParallelFor(layout_.output_blocks, options_.threads, [&](size_t o, size_t worker) {
            reduce_giants(tile, o, worker, [&](size_t giant, seal::Ciphertext &sum) {
                const size_t index = o * G + giant;
                if (nonzero[index]) sum = std::move(sums[index]);
                return nonzero[index];
            });
        });
    }
    stats_.compute_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return output;
}
Tensor<uint64_t> Conv2DOptiEncode::DepackResult(Tensor<uint64_t> &output) {
    auto result = layout_.DepackResult(output);
    AddBias(result);
    return result;
}
void Conv2DOptiEncode::AddBias(Tensor<uint64_t> &result) const {
    if (options_.add_bias && HE->server) {
        const size_t plane = layout_.output_height * layout_.output_width;
        for (size_t i = 0; i < result.size(); ++i)
            result(i) = (result(i) + bias(i / plane) % HE->plain_mod) % HE->plain_mod;
    }
}
Tensor<uint64_t> Conv2DOptiEncode::operator()(Tensor<uint64_t> &input) {
    auto state = std::move(prepared_); // consume even if an operation throws
    Tensor<uint64_t> output;
    if (state && state->include_product) {
        auto correction = OpenMaskedInput(input, *state);
        output = std::move(state->product_share);
        if (HE->server) AddRingProduct(correction, output);
    } else {
        Tensor<UnifiedCiphertext> encrypted;
        if (state) {
            auto correction = OpenMaskedInput(input, *state);
            encrypted = ApplyPreparedInput(correction, *state);
        } else {
            encrypted = Operator::SSToHE_coeff(PackActivation(input), HE,
                options_.ciphertext_inverse_normalization ? layout_.channels : 1);
        }
        auto evaluated = HECompute(weight_pt, encrypted);
        if (state || options_.compact_output) {
            output = ExtractMaskedOutput(evaluated);
        } else {
            auto shared = Operator::HEToSS_coeff(evaluated, HE);
            output = DepackResult(shared);
        }
    }
    if (state || options_.compact_output) AddBias(output);
    output.bitwidth = input.bitwidth;
    output.scale = input.scale + weight.scale;
    return output;
}
} // namespace LinearLayer
