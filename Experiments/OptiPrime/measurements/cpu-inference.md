# Encrypted CPU inference on SEC-PPDL

Both parties used 16 CPU workers. Timings include online inference through secure argmax and fresh per-image Ferret OT preprocessing. HE runs online on both sides. Model setup, key exchange, OT planning and opening test outputs for verification are excluded. Traffic sums application bytes sent by both parties in decimal GB.

Cheetah uses torchvision ImageNet-1K V1 weights at Q8 fixed point (eight fractional bits), a 50-bit HE plaintext prime and a 60-bit MPC work ring. ReLU precedes exact known-MSB truncation where the result is nonnegative. Signed projections use signed truncation. For MobileNetV2, the standard Cheetah convolution evaluates a diagonal kernel for depthwise layers. OptiPrime uses W4A8 (W8 final classifier), optimized nonlinear operators and OptiEncode. VGG-19 uses its 26-bit default work ring; the other networks use 20 bits and compact online HE outputs. RN18 uses the released AIMET weights; the other quantized models are exported from torchvision V1.

Linux tc shaped both directions of private Docker loopback on an AMD EPYC 9J14. LAN: 3,072 Mbit/s and 0.3 ms RTT. WAN: 352 Mbit/s and 40 ms RTT. The ResNet-18 means cover the same 16 ImageNetV2 photos in each configuration. The other models each have one LAN timing observation per configuration.

| Model | Network | Method | Images | Online (s) | Per-image OT (s) | End-to-end (s) | Online (GB) | OT (GB) | Total (GB) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet-18 | LAN | Cheetah | 16 | 33.25 | 12.06 | 45.31 | 6.884 | 0.054 | 6.938 |
| ResNet-18 | LAN | OptiPrime | 16 | 3.83 | 2.94 | 6.77 | 0.291 | 0.009 | 0.299 |
| ResNet-18 | WAN | Cheetah | 16 | 218.65 | 16.76 | 235.41 | 6.884 | 0.054 | 6.938 |
| ResNet-18 | WAN | OptiPrime | 16 | 36.44 | 3.36 | 39.79 | 0.291 | 0.009 | 0.299 |
| ResNet-34 | LAN | Cheetah | 1 | 49.02 | 27.32 | 76.34 | 9.846 | 0.089 | 9.934 |
| ResNet-34 | LAN | OptiPrime | 1 | 5.95 | 7.35 | 13.29 | 0.404 | 0.027 | 0.431 |
| ResNet-50 | LAN | Cheetah | 1 | 559.55 | 35.74 | 595.30 | 131.609 | 0.239 | 131.848 |
| ResNet-50 | LAN | OptiPrime | 1 | 13.01 | 12.50 | 25.51 | 1.089 | 0.044 | 1.134 |
| VGG-16 | LAN | Cheetah | 1 | 85.33 | 57.33 | 142.66 | 15.604 | 0.248 | 15.852 |
| VGG-16 | LAN | OptiPrime | 1 | 15.74 | 8.42 | 24.16 | 2.006 | 0.044 | 2.051 |
| VGG-19 | LAN | Cheetah | 1 | 94.58 | 45.53 | 140.10 | 17.303 | 0.266 | 17.569 |
| VGG-19 | LAN | OptiPrime | 1 | 26.02 | 18.49 | 44.51 | 3.697 | 0.062 | 3.759 |
| MobileNetV2 | LAN | Cheetah | 1 | 116.14 | 36.07 | 152.21 | 26.360 | 0.159 | 26.519 |
| MobileNetV2 | LAN | OptiPrime | 1 | 12.39 | 7.91 | 20.30 | 1.021 | 0.044 | 1.065 |

[Per-photo timings, byte counts, predictions and network probes](cpu-inference.json)

## Communication reduction

Relative to the Cheetah configuration above, per LAN inference.

| Model | Online reduction | Reduction including OT preprocessing |
|---|---:|---:|
| ResNet-18 | 95.78% | 95.68% |
| ResNet-34 | 95.89% | 95.66% |
| ResNet-50 | 99.17% | 99.14% |
| VGG-16 | 87.14% | 87.06% |
| VGG-19 | 78.63% | 78.60% |
| MobileNetV2 | 96.13% | 95.98% |
