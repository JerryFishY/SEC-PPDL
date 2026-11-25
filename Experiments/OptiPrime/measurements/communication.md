# Communication in encrypted CPU inference

[CPU and network setup](README.md#cpu-and-network-setup)

One matched ImageNetV2 image per model, including secure argmax. Both parties used 16 CPU threads and fresh Ferret OT preprocessing. HE was online in both configurations. Decimal GB sum application bytes sent by the two parties. The table excludes one-time setup and opening outputs for verification.

Cheetah uses torchvision V1 weights at Q8 fixed point (eight fractional bits), a 50-bit plaintext prime, a 60-bit work ring and exact known-MSB truncation after nonnegative activations. Signed branches retain signed truncation. OptiPrime uses W4A8, a W8 classifier, OptiEncode and approximate nonlinearities. VGG-19 uses a 26-bit work ring; other models use 20 bits and compact online HE output. For MobileNetV2, Cheetah implements depthwise convolution using diagonal weights in its standard dense convolution.

| Model | Cheetah online (GB) | OT (GB) | Total (GB) | OptiPrime online (GB) | OT (GB) | Total (GB) | Total reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet-18 | 6.884 | 0.071 | 6.955 | 0.291 | 0.018 | 0.308 | 95.57% |
| ResNet-34 | 9.846 | 0.089 | 9.934 | 0.404 | 0.027 | 0.431 | 95.66% |
| ResNet-50 | 131.609 | 0.239 | 131.848 | 1.089 | 0.044 | 1.134 | 99.14% |
| VGG-16 | 15.604 | 0.248 | 15.852 | 2.006 | 0.044 | 2.051 | 87.06% |
| VGG-19 | 17.303 | 0.266 | 17.569 | 3.697 | 0.062 | 3.759 | 78.60% |
| MobileNetV2 | 26.360 | 0.159 | 26.519 | 1.021 | 0.044 | 1.065 | 95.98% |
