#include <LinearLayer/ConvOptiEncode.h>
#ifdef USE_HE_GPU
#include <phantom/evaluate.cuh>
#include <HE/unified/NTTDot.h>
#include <chrono>

namespace LinearLayer {
void Conv2DOptiEncode::ApplyAutomorphismGPU(UnifiedCiphertext &ciphertext, uint32_t element) const {
    if (element == 1) return;
    if (HE->galoisKeys->hgalois().has_key(element)) {
        phantom::apply_galois_inplace(HE->context->dcontext(), ciphertext.dcipher(), element,
                                     HE->galoisKeys->dgalois());
    } else {
        const auto step = rotation_steps_.find(element);
        if (step == rotation_steps_.end()) throw std::invalid_argument("Missing OptiEncode GPU Galois key");
        HE->evaluator->rotate_rows_inplace(ciphertext, step->second, *HE->galoisKeys);
    }
}

namespace {
void CheckCUDA(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void CopyDevice(const UnifiedCiphertext &from, UnifiedCiphertext &to) {
    const auto &src = from.dcipher();
    auto &dst = to.dcipher();
    if (src.chain_index() != dst.chain_index() || src.size() != dst.size() ||
        src.poly_modulus_degree() != dst.poly_modulus_degree() ||
        src.is_ntt_form() != dst.is_ntt_form())
        throw std::logic_error("OptiEncode CUDA graph ciphertext shape mismatch");
    const auto bytes = src.size() * src.coeff_modulus_size() * src.poly_modulus_degree() * sizeof(uint64_t);
    CheckCUDA(cudaMemcpyAsync(dst.data(), src.data(), bytes, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
}
} // namespace

struct Conv2DOptiEncode::GraphState {
    Tensor<UnifiedCiphertext> inputs, outputs;
    cudaGraphExec_t executable = nullptr;
    ~GraphState() { if (executable) cudaGraphExecDestroy(executable); }
};

void Conv2DOptiEncode::PrepareGraph() {
    // A wholly zero output block needs a fresh encryption on each invocation.
    for (size_t o = 0; o < layout_.output_blocks; ++o) {
        bool nonzero = false;
        for (size_t i = 0; i < layout_.WeightInputBlocks(); ++i)
            for (size_t k = 0; k < layout_.channels; ++k)
                nonzero |= nonzero_weights_[(i * layout_.output_blocks + o) * layout_.channels + k] != 0;
        if (!nonzero) return;
    }
    auto state = std::make_shared<GraphState>();
    state->inputs = Tensor<UnifiedCiphertext>({layout_.TileCount(), layout_.input_blocks},
                                              HE->GenerateZeroCiphertext(DEVICE));
    state->outputs = Tensor<UnifiedCiphertext>({layout_.TileCount(), layout_.output_blocks}, DEVICE);
    EvaluateGPU(weight_pt, state->inputs, state->outputs);
    HE->evaluator->sync();

    Tensor<UnifiedCiphertext> temporary(state->outputs.shape(), DEVICE);
    cudaGraph_t captured = nullptr;
    CheckCUDA(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    try {
        EvaluateGPU(weight_pt, state->inputs, temporary);
        for (size_t i = 0; i < temporary.size(); ++i) {
            CopyDevice(temporary(i), state->outputs(i));
            temporary(i) = UnifiedCiphertext(DEVICE);
        }
    } catch (...) {
        cudaStreamEndCapture(cudaStreamPerThread, &captured);
        if (captured) cudaGraphDestroy(captured);
        throw;
    }
    CheckCUDA(cudaStreamEndCapture(cudaStreamPerThread, &captured));
    const auto status = cudaGraphInstantiate(&state->executable, captured, 0);
    cudaGraphDestroy(captured);
    CheckCUDA(status);
    CheckCUDA(cudaGraphUpload(state->executable, cudaStreamPerThread));
    CheckCUDA(cudaGraphLaunch(state->executable, cudaStreamPerThread));
    CheckCUDA(cudaStreamSynchronize(cudaStreamPerThread));
    graph_ = std::move(state);
}

Tensor<UnifiedCiphertext> Conv2DOptiEncode::HEComputeGPU(const Tensor<UnifiedPlaintext> &weights,
                                                         Tensor<UnifiedCiphertext> &inputs) {
    Tensor<UnifiedCiphertext> output({layout_.TileCount(), layout_.output_blocks}, HE->server ? DEVICE : HOST);
    if (!HE->server) return output;
    const auto start = std::chrono::steady_clock::now();
    if (graph_) {
        for (size_t i = 0; i < inputs.size(); ++i) CopyDevice(inputs(i), graph_->inputs(i));
        CheckCUDA(cudaGraphLaunch(graph_->executable, cudaStreamPerThread));
        output = graph_->outputs;
    } else {
        EvaluateGPU(weights, inputs, output);
    }
    HE->evaluator->sync();
    stats_.compute_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return output;
}

void Conv2DOptiEncode::EvaluateGPU(const Tensor<UnifiedPlaintext> &weights,
                                   const Tensor<UnifiedCiphertext> &inputs,
                                   Tensor<UnifiedCiphertext> &output) {
    auto &evaluator = *HE->evaluator;
    const uint32_t giant_element = static_cast<uint32_t>(Operator::OptiEncodePowerMod(
        layout_.generator, layout_.baby_steps, 2 * layout_.degree));
    const size_t slots = layout_.WeightInputBlocks();
    for (size_t tile = 0; tile < layout_.TileCount(); ++tile) {
        std::vector<UnifiedCiphertext> babies(slots * layout_.baby_steps);
        auto prepare_babies = [&](size_t output_block) {
            for (size_t slot = 0; slot < slots; ++slot) {
                auto current = inputs({tile, layout_.SourceBlock(slot, output_block)});
                for (size_t a = 0; a < layout_.baby_steps; ++a) {
                    if (a) ApplyAutomorphismGPU(current, layout_.generator);
                    auto &baby = babies[slot * layout_.baby_steps + a];
                    if (a + 1 == layout_.baby_steps) baby = std::move(current);
                    else baby = current;
                    evaluator.transform_to_ntt_inplace(baby);
                }
            }
        };
        for (size_t o = 0; o < layout_.output_blocks; ++o) {
            // Dense outputs share all babies. Each depthwise output consumes one input block.
            if (layout_.depthwise || o == 0) prepare_babies(o);
            UnifiedCiphertext result;
            bool have_result = false;
            for (size_t giant = layout_.giant_steps; giant-- > 0;) {
                if (have_result) ApplyAutomorphismGPU(result, giant_element);
                UnifiedCiphertext subtotal;
                HE::unified::NTTProductBatch batch;
                bool have_subtotal = false;
                auto flush = [&] {
                    auto &ct = subtotal.dcipher();
                    HE::unified::AccumulateNTTProducts(
                        batch, HE->context->dcontext().gpu_rns_tables().modulus(),
                        ct.poly_modulus_degree(), ct.coeff_modulus_size(), ct.size(),
                        ct.data(), have_subtotal);
                    batch.count = 0;
                    have_subtotal = true;
                };
                for (size_t i = 0; i < layout_.WeightInputBlocks(); ++i) {
                    for (size_t a = 0; a < layout_.baby_steps; ++a) {
                        const size_t index = (i * layout_.output_blocks + o) * layout_.channels
                                             + giant * layout_.baby_steps + a;
                        if (!nonzero_weights_[index]) continue;
                        const auto &baby = babies[i * layout_.baby_steps + a];
                        if (!have_subtotal && !batch.count) subtotal = baby;
                        batch.ciphertexts[batch.count] = baby.dcipher().data();
                        batch.plaintexts[batch.count] = weights(index).dplain().data();
                        if (++batch.count == batch.capacity) flush();
                    }
                }
                if (batch.count) flush();
                if (have_subtotal) {
                    evaluator.transform_from_ntt_inplace(subtotal);
                    if (have_result) evaluator.add_inplace(result, subtotal);
                    else { result = std::move(subtotal); have_result = true; }
                }
            }
            if (!have_result) result = HE->GenerateZeroCiphertext(DEVICE);
            for (size_t level = 0; level < options_.output_mod_switches; ++level) {
                const size_t limbs = result.coeff_modulus_size();
                // SEAL's valid chain can stop before the smallest Phantom level.
                auto data = HE->context->hcontext().first_context_data();
                while (data && data->parms().coeff_modulus().size() != limbs) data = data->next_context_data();
                if (!data || !data->next_context_data()) break;
                evaluator.mod_switch_to_next_inplace(result);
            }
            output({tile, o}) = std::move(result);
        }
    }
}
} // namespace LinearLayer
#endif
