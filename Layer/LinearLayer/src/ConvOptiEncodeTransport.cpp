#include <LinearLayer/ConvOptiEncode.h>
#include <Utils/ParallelFor.h>
#include <Utils/prg.h>
#include <algorithm>
#ifdef USE_HE_GPU
#include "RingConvGPU.h"
#endif

namespace LinearLayer {
namespace {
// All message sizes follow from the public layer geometry. Each ciphertext
// starts on a word boundary, so parallel writers never share a word.
struct PackedWords {
    unsigned bits;
    uint64_t mask;
    explicit PackedWords(unsigned width) : bits(width), mask((uint64_t(1) << width) - 1) {}
    size_t Words(size_t count) const { return (count * bits + 63) / 64; }
    void Put(uint64_t *dst, size_t i, uint64_t value) const {
        const size_t bit = i * bits, word = bit / 64;
        const unsigned shift = bit % 64;
        dst[word] |= value << shift;
        if (shift + bits > 64) dst[word + 1] |= value >> (64 - shift);
    }
    uint64_t Get(const uint64_t *src, size_t i) const {
        const size_t bit = i * bits, word = bit / 64;
        const unsigned shift = bit % 64;
        uint64_t value = src[word] >> shift;
        if (shift + bits > 64) value |= src[word + 1] << (64 - shift);
        return value & mask;
    }
};
void Transfer(HE::HEEvaluator &he, std::vector<uint64_t> &words, bool sender) {
    constexpr size_t chunk = 1 << 17;
    for (size_t i = 0; i < words.size(); i += chunk) {
        const auto bytes = std::min(chunk, words.size() - i) * sizeof(uint64_t);
        if (sender) he.IO->send_data(words.data() + i, bytes);
        else he.IO->recv_data(words.data() + i, bytes);
    }
}
struct OutputTile {
    size_t row, col, channel, rows, cols, channels;

    OutputTile(const Operator::OptiEncodeLayout &l, size_t index) {
        const size_t tile = index / l.output_blocks;
        row = tile / l.tile_columns * l.tile_height;
        col = tile % l.tile_columns * l.tile_width;
        channel = index % l.output_blocks * l.channels;
        rows = std::min(l.tile_height, l.output_height - row);
        cols = std::min(l.tile_width, l.output_width - col);
        channels = std::min(l.channels, l.output_channels - channel);
    }
    size_t Size() const { return rows * cols * channels; }
    template<class F> void Visit(const Operator::OptiEncodeLayout &l, F f) const {
        size_t i = 0;
        for (size_t c = 0; c < channels; ++c)
            for (size_t y = 0; y < rows; ++y)
                for (size_t x = 0; x < cols; ++x)
                    f(i++, l.OutputIndex(y, x, c),
                      ((channel + c) * l.output_height + row + y) * l.output_width + col + x);
    }
};
} // namespace

void Conv2DOptiEncode::ValidateCompactTransport() const {
    const auto &context = HE->context->hcontext();
    if ((HE->plain_mod & (HE->plain_mod - 1)) ||
        context.first_context_data()->parms().coeff_modulus().size() != 1)
        throw std::invalid_argument("Compact outputs require a binary plaintext ring and one BFV data prime");
}
void Conv2DOptiEncode::PrepareInput(bool include_product) {
    if (prepared_) throw std::logic_error("Previous prepared input is still unused");
    ValidateCompactTransport();
    if (include_product && HE->plain_mod > (uint64_t(1) << 32))
        throw std::invalid_argument("Prepared ring products support up to 32 plaintext bits");
    auto state = std::make_unique<PreparedInput>();
    state->include_product = include_product;
    const auto &l = layout_;
    Tensor<UnifiedCiphertext> ct({l.TileCount(), l.input_blocks}, HOST);
    if (HE->server) {
        HE->ReceiveEncVec(ct);

    } else {
        state->mask = Tensor<uint64_t>({l.input_channels, l.height, l.width});
        Utils::RandomFieldElements(state->mask.data().data(), state->mask.size(), HE->plain_mod);
        auto packed = PackActivation(state->mask);
        Utils::ParallelFor(ct.size(), options_.threads, [&](size_t i, size_t) {
            seal::Plaintext plain(l.degree);
            std::copy_n(packed.data().data() + i * l.degree, l.degree, plain.data());
            HE->encryptor->encrypt(plain, ct(i).hcipher());
        });
        HE->SendEncVec(ct);
    }
    if (include_product) {
        if (HE->server) {
            const uint64_t inverse = InputInverse();
            Utils::ParallelFor(ct.size(), options_.threads, [&](size_t i, size_t) {
                Normalize(ct(i).hcipher(), inverse);
            });
            if (HE->Backend() == DEVICE) {
                for (auto &cipher : ct.data()) cipher.to_device(*HE->context);
                HE->evaluator->sync();
            }
            if (ring_weights_.empty()) {
                std::vector<uint32_t> weights(weight.data().begin(), weight.data().end());
#ifdef USE_HE_GPU
                if (HE->Backend() == DEVICE)
                    ring_gpu_ = MakeRingConvGPU(l.height, l.input_channels, l.output_channels, l.kernel,
                                                l.stride, l.padding, l.output_height, weights.data(), l.depthwise);
#endif
                ring_weights_ = std::move(weights);
            }
        }
        auto evaluated = HECompute(weight_pt, ct);
        state->product_share = ExtractMaskedOutput(evaluated);
    } else if (HE->server) {
        state->ciphertexts = std::move(ct);
    }
    prepared_ = std::move(state);
}

Tensor<uint64_t> Conv2DOptiEncode::OpenMaskedInput(const Tensor<uint64_t> &input,
                                                   const PreparedInput &state) {
    const auto &l = layout_;
    if (input.shape() != std::vector<size_t>{l.input_channels, l.height, l.width})
        throw std::invalid_argument("OptiEncode input shape mismatch");
    const PackedWords field(__builtin_ctzll(HE->plain_mod));
    std::vector<uint64_t> wire(field.Words(input.size()));
    if (!HE->server) {
        for (size_t i = 0; i < input.size(); ++i)
            field.Put(wire.data(), i, (input(i) - state.mask(i)) & field.mask);
        Transfer(*HE, wire, true);
        return {};
    }
    Transfer(*HE, wire, false);
    Tensor<uint64_t> correction(input.shape());
    for (size_t i = 0; i < input.size(); ++i)
        correction(i) = (input(i) + field.Get(wire.data(), i)) & field.mask;
    return correction;
}

uint64_t Conv2DOptiEncode::InputInverse() const {
    const auto &q = HE->context->hcontext().first_context_data()->parms().coeff_modulus().front();
    uint64_t inverse = 1;
    if (options_.ciphertext_inverse_normalization &&
        !seal::util::try_invert_uint_mod(layout_.channels, q, inverse))
        throw std::logic_error("Channel block is not invertible");
    return inverse;
}
void Conv2DOptiEncode::Normalize(seal::Ciphertext &ct, uint64_t inverse) const {
    if (inverse == 1) return;
    const auto &q = HE->context->hcontext().first_context_data()->parms().coeff_modulus().front();
    for (size_t k = 0; k < ct.size(); ++k)
        seal::util::multiply_poly_scalar_coeffmod(ct.data(k), layout_.degree, inverse, q, ct.data(k));
}
Tensor<UnifiedCiphertext> Conv2DOptiEncode::ApplyPreparedInput(Tensor<uint64_t> &correction,
                                                               PreparedInput &state) {
    const auto &l = layout_;
    if (!HE->server) return Tensor<UnifiedCiphertext>({l.TileCount(), l.input_blocks}, HOST);
    const auto packed = PackActivation(correction);
    auto ct = std::move(state.ciphertexts);
    const auto &context = HE->context->hcontext();
    const auto inverse = InputInverse();
    seal::Evaluator evaluator(context);
    Utils::ParallelFor(ct.size(), options_.threads, [&](size_t i, size_t) {
        seal::Plaintext plain(l.degree);
        std::copy_n(packed.data().data() + i * l.degree, l.degree, plain.data());
        auto &cipher = ct(i).hcipher();
        evaluator.add_plain_inplace(cipher, plain);
        Normalize(cipher, inverse);
    });
    if (HE->Backend() == DEVICE) {
        for (auto &cipher : ct.data()) cipher.to_device(*HE->context);
        HE->evaluator->sync();
    }
    return ct;
}

Tensor<uint64_t> Conv2DOptiEncode::ExtractMaskedOutput(Tensor<UnifiedCiphertext> &ct) {
    const auto &l = layout_;
    const auto &context = HE->context->hcontext();
    const auto &prime = context.first_context_data()->parms().coeff_modulus().front();
    const auto q = prime.value(), t = HE->plain_mod;
    const unsigned shift = __builtin_ctzll(t);
    const PackedWords residue(prime.bit_count());
    Tensor<uint64_t> out({l.output_channels, l.output_height, l.output_width});
    std::vector<size_t> offsets(ct.size() + 1);
    for (size_t i = 0; i < ct.size(); ++i)
        offsets[i + 1] = offsets[i] + residue.Words(l.degree + OutputTile(l, i).Size());
    std::vector<uint64_t> wire(offsets.back());
    if (HE->server) {
        Utils::RandomFieldElements(out.data().data(), out.size(), t);
        if (HE->Backend() == DEVICE)
            for (auto &cipher : ct.data()) cipher.to_host(*HE->context);
        Utils::ParallelFor(ct.size(), options_.threads, [&](size_t i, size_t) {
            const auto &cipher = ct(i).hcipher();
            if (cipher.size() != 2 || cipher.is_ntt_form() || cipher.parms_id() != context.first_parms_id())
                throw std::logic_error("Unexpected BFV output representation");
            auto *dst = wire.data() + offsets[i];
            for (size_t k = 0; k < l.degree; ++k) residue.Put(dst, k, cipher.data(1)[k]);
            OutputTile(l, i).Visit(l, [&](size_t j, size_t slot, size_t pos) {
                // SEAL BFV add_plain for t=2^k and one prime: round(q*m/t).
                const auto neg = (-out(pos)) & (t - 1);
                const uint64_t lift = (__uint128_t(q) * neg + t / 2) >> shift;
                const uint64_t value = cipher.data(0)[slot] + lift;
                residue.Put(dst, l.degree + j, value >= q ? value - q : value);
            });
        });
        Transfer(*HE, wire, true);
    } else {
        Transfer(*HE, wire, false);
        Utils::ParallelFor(ct.size(), options_.threads, [&](size_t i, size_t) {
            const auto *src = wire.data() + offsets[i];
            const auto read = [&](size_t k) {
                const auto v = residue.Get(src, k);
                if (v >= q) throw std::runtime_error("Invalid BFV residue");
                return v;
            };
            // In coefficient form Dec(c)[j] only uses c0[j] and (c1*s)[j].
            // Omitted c0 coefficients have no effect on any extracted position.
            seal::Ciphertext cipher;
            cipher.resize(context, context.first_parms_id(), 2);
            std::fill_n(cipher.data(0), l.degree, uint64_t(0));
            for (size_t k = 0; k < l.degree; ++k) cipher.data(1)[k] = read(k);
            OutputTile tile(l, i);
            tile.Visit(l, [&](size_t j, size_t slot, size_t) { cipher.data(0)[slot] = read(l.degree + j); });
            seal::Plaintext plain;
            HE->decryptor->decrypt(cipher, plain);
            tile.Visit(l, [&](size_t, size_t slot, size_t pos) {
                out(pos) = slot < plain.coeff_count() ? plain[slot] : 0;
            });
        });
    }
    return out;
}
} // namespace LinearLayer
