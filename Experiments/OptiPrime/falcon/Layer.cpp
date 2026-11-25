// Standalone ICCAD 2023 Falcon packing reproduction for public test vectors.
#include "Layer.h"
#include <LinearOperator/Conversion.h>
#include <Utils/ParallelFor.h>
#include <Utils/prg.h>

namespace FalconExperiment {
namespace {
using Datatype::Tensor;
using HE::HEEvaluator;
using HE::unified::UnifiedCiphertext;
constexpr size_t t = 1 << 20;

size_t Up(size_t a, size_t b) { return (a + b - 1) / b; }

struct Group {
    std::vector<size_t> lanes, offsets;
};
std::vector<Group> Groups(size_t cx, size_t cw) {
    std::vector<Group> result;
    for (size_t i = 0; i < cx; ++i) {
        if (i % cw == 0)
            result.emplace_back();
        auto &group = result.back();
        size_t lane = i % 2 ? i / 2 : cx - 1 - i / 2;
        size_t offset = group.lanes.empty()
                            ? 0
                            : group.offsets.back() + 1 + std::max(group.lanes.back(), cx - 1 - lane);
        group.lanes.push_back(lane);
        group.offsets.push_back(offset);
    }
    return result;
}

// The same per-ciphertext word alignment and residue width as OptiEncode compact output.
void Compact(uint64_t *wire, HEEvaluator &he, seal::Ciphertext &ct, const std::vector<size_t> &slots,
             const std::vector<size_t> &positions, Tensor<uint64_t> &shares) {
    const auto &context = he.context->hcontext();
    const size_t n = he.polyModulusDegree;
    const auto &prime = context.first_context_data()->parms().coeff_modulus().front();
    const unsigned bits = prime.bit_count();

    if (he.server) {
        seal::Plaintext mask(n);
        for (size_t j = 0; j < slots.size(); ++j)
            mask[slots[j]] = (-shares(positions[j])) & (t - 1);
        seal::Evaluator(context).add_plain_inplace(ct, mask);
        for (size_t j = 0; j < n + slots.size(); ++j) {
            const uint64_t v = j < n ? ct.data(1)[j] : ct.data(0)[slots[j - n]];
            const size_t word = j * bits / 64, off = j * bits % 64;
            wire[word] |= v << off;
            if (off + bits > 64)
                wire[word + 1] |= v >> (64 - off);
        }

    } else {

        ct.resize(context, context.first_parms_id(), 2);
        std::fill_n(ct.data(0), n, uint64_t(0));
        for (size_t j = 0; j < n + slots.size(); ++j) {
            const size_t word = j * bits / 64, off = j * bits % 64;
            uint64_t v = wire[word] >> off;
            if (off + bits > 64)
                v |= wire[word + 1] << (64 - off);
            (j < n ? ct.data(1)[j] : ct.data(0)[slots[j - n]]) = v & ((uint64_t(1) << bits) - 1);
        }
        seal::Plaintext plain;
        he.decryptor->decrypt(ct, plain);
        for (size_t j = 0; j < slots.size(); ++j)
            shares(positions[j]) = slots[j] < plain.coeff_count() ? plain[slots[j]] : 0;
    }
}

} // namespace
Layer::Layer(HEEvaluator &he, Shape s, const Tensor<uint64_t> &w, size_t threads)
    : he_(he), shape_(s), threads_(threads) {
    if (!s.stride)
        throw std::invalid_argument("Zero Falcon stride");
    const size_t n = he.polyModulusDegree, side = Up(s.h, s.stride);
    if (!s.rows || !s.cols || !s.cx || !s.cw || !s.channels || s.rows > side || s.cols > side ||
        s.cx % s.cw || he.Backend() != Datatype::HOST || he.plain_mod != t ||
        he.context->hcontext().first_context_data()->parms().coeff_modulus().size() != 1 ||
        w.shape() != std::vector<size_t>{s.channels, 1, 3, 3})
        throw std::invalid_argument("Invalid Falcon layer parameters");
    const size_t pw = (s.cols - 1) * s.stride + 3, ph = (s.rows - 1) * s.stride + 3;
    const size_t plane = pw * ph, origin = 2 * pw + 2, blocks = Up(s.channels, s.cx);
    const size_t tc = Up(side, s.cols), tiles = Up(side, s.rows) * tc;
    if (plane * s.cx > n)
        throw std::invalid_argument("Input exceeds Falcon capacity");
    inputs_ = tiles * blocks;
    const auto &ctx = he.context->hcontext();
    const unsigned bits = ctx.first_context_data()->parms().coeff_modulus()[0].bit_count();
    const auto groups = Groups(s.cx, s.cw);
    seal::Evaluator evaluator(ctx);
    for (size_t block = 0; block < blocks; ++block)
        for (const auto &g : groups) {
            seal::Plaintext weight(n);
            bool active = false;
            for (size_t j = 0; j < g.lanes.size(); ++j) {
                size_t c = block * s.cx + g.lanes[j];
                if (c >= s.channels)
                    continue;
                active = true;
                if (g.offsets[j] * plane + origin >= n)
                    throw std::invalid_argument("Weight exceeds degree");
                if (he.server)
                    for (size_t r = 0; r < 3; ++r)
                        for (size_t col = 0; col < 3; ++col)
                            weight[g.offsets[j] * plane + origin - r * pw - col] =
                                w(c * 9 + r * 3 + col);
            }
            if (!active)
                continue;
            size_t wi = weights_.size();
            if (he.server && !weight.is_zero()) {
                evaluator.transform_to_ntt_inplace(weight, ctx.first_parms_id());
                weights_.push_back(std::move(weight));
            } else
                weights_.emplace_back();
            for (size_t tile = 0; tile < tiles; ++tile) {
                Task task{tile * blocks + block, wi, words_, {}, {}};
                for (size_t j = 0; j < g.lanes.size(); ++j) {
                    size_t c = block * s.cx + g.lanes[j];
                    if (c >= s.channels)
                        continue;
                    for (size_t r = 0; r < s.rows; ++r)
                        for (size_t col = 0; col < s.cols; ++col) {
                            size_t y = tile / tc * s.rows + r, z = tile % tc * s.cols + col;
                            if (y >= side || z >= side)
                                continue;
                            size_t slot =
                                (g.lanes[j] + g.offsets[j]) * plane + origin + (r * pw + col) * s.stride;
                            if (slot >= n)
                                throw std::invalid_argument("Falcon output exceeds degree");
                            task.slots.push_back(slot);
                            task.positions.push_back((c * side + y) * side + z);
                        }
                }
                words_ += ((n + task.slots.size()) * bits + 63) / 64;
                tasks_.push_back(std::move(task));
            }
        }
}

Tensor<uint64_t> Layer::Forward(const Tensor<uint64_t> &x) const {
    const auto &s = shape_;
    const size_t n = he_.polyModulusDegree, side = Up(s.h, s.stride);
    const size_t ph = (s.rows - 1) * s.stride + 3, pw = (s.cols - 1) * s.stride + 3;
    const size_t plane = ph * pw, blocks = Up(s.channels, s.cx), tc = Up(side, s.cols);
    Tensor<uint64_t> packed({inputs_, n}), out({s.channels, side, side});
    Utils::ParallelFor(inputs_, threads_, [&](size_t input, size_t) {
        const size_t tile = input / blocks, first = (input % blocks) * s.cx;
        for (size_t c = first; c < std::min(s.channels, first + s.cx); ++c)
            for (size_t r = 0; r < ph; ++r)
                for (size_t col = 0; col < pw; ++col) {
                    size_t y = (tile / tc * s.rows) * s.stride + r - 1,
                           z = (tile % tc * s.cols) * s.stride + col - 1;
                    if (y < s.h && z < s.h)
                        packed(input * n + (c - first) * plane + r * pw + col) =
                            x((c * s.h + y) * s.h + z);
                }
    });
    auto ct = Operator::SSToHE_coeff(packed, &he_);
    std::vector<uint64_t> wire(words_);
    auto transfer = [&] {
        for (size_t i = 0; i < words_; i += 1 << 17) {
            size_t len = 8 * std::min<size_t>(1 << 17, words_ - i);
            if (he_.server)
                he_.IO->send_data(wire.data() + i, len);
            else
                he_.IO->recv_data(wire.data() + i, len);
        }
    };
    const auto &ctx = he_.context->hcontext();
    if (he_.server) {
        Utils::RandomFieldElements(out.data().data(), out.size(), t);
        Utils::ParallelFor(inputs_, threads_, [&](size_t i, size_t) {
            seal::Evaluator(ctx).transform_to_ntt_inplace(ct(i).hcipher());
        });
    } else
        transfer();
    Utils::ParallelFor(tasks_.size(), threads_, [&](size_t i, size_t) {
        const auto &task = tasks_[i];
        seal::Ciphertext result;
        if (he_.server) {
            seal::Evaluator evaluator(ctx);
            if (weights_[task.weight].is_zero())
                he_.encryptor->encrypt_zero(result);
            else {
                evaluator.multiply_plain(ct(task.input).hcipher(), weights_[task.weight], result);
                evaluator.transform_from_ntt_inplace(result);
            }
        }
        Compact(wire.data() + task.wire_offset, he_, result, task.slots, task.positions, out);
    });
    if (he_.server)
        transfer();
    return out;
}
} // namespace FalconExperiment
