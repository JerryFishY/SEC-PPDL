#pragma once
#include <LinearLayer/ConvOptiEncode.h>

namespace FalconExperiment {
struct Shape {
    size_t h, channels, stride, rows, cols, cx, cw;
};

// CPU depthwise reference, using the same compact wire format as the OptiEncode study.
class Layer {
  public:
    Layer(HE::HEEvaluator &he, Shape shape, const Datatype::Tensor<uint64_t> &weights, size_t threads);
    Datatype::Tensor<uint64_t> Forward(const Datatype::Tensor<uint64_t> &input) const;
    size_t InputCount() const { return inputs_; }
    size_t OutputCount() const { return tasks_.size(); }

  private:
    struct Task {
        size_t input, weight, wire_offset;
        std::vector<size_t> slots, positions;
    };
    HE::HEEvaluator &he_;
    Shape shape_;
    size_t threads_, inputs_, words_ = 0;
    std::vector<seal::Plaintext> weights_;
    std::vector<Task> tasks_;
};
} // namespace FalconExperiment
