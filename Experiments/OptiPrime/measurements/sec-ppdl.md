# SEC-PPDL measurements

These measurements cover two-party W4A8 ResNet-18 inference through secure argmax on a 224 × 224 input. Convolutions use 4-bit weights, activations use 8 bits, and the final classifier uses 8-bit weights. The paper evaluates its own CPU and HE accelerator configurations. [Paper evaluation](https://arxiv.org/html/2609.16898v1#S5) · [SEC-PPDL experiment guide](https://github.com/JerryFishY/SEC-PPDL/blob/b986671362e8e6e244ca80a81a0ba6fbccf1c976/Experiments/OptiPrime/README.md)

## CPU comparison on LAN and WAN

Both parties ran on one AMD EPYC 9J14 host with 16 worker threads per party. Linux tc (HTB and netem) shaped both directions inside Docker. LAN used 3,072 Mbit/s per direction and 0.3 ms RTT. WAN used 352 Mbit/s and 40 ms RTT. Each configuration processed the same single ImageNetV2 image once using the same quantized model. The numbers are individual observations.

The SEC-PPDL Cheetah baseline uses its 60-bit work ring and online IKNP OT. The two OptiEncode configurations use a 20-bit work ring and preprocessed Ferret OT. The comparison includes these protocol and quantization choices as well as the encoding.

| Network | CPU configuration | Online linear (s) | Other online (s) | Online total (s) | Online + per-image preprocessing (s) | Online traffic (MB) | Preprocessing traffic (MB) |
|---|---|---:|---:|---:|---:|---:|---:|
| LAN | Cheetah baseline | 42.31 | 35.32 | 77.62 | 77.62 | 22,099.64 | 0.00 |
| LAN | OptiEncode exact | 2.35 | 3.98 | 6.33 | 15.17 | 516.34 | 26.56 |
| LAN | OptiEncode approximate | 2.32 | 2.32 | 4.64 | 8.72 | 357.58 | 17.70 |
| WAN | Cheetah baseline | 280.90 | 284.28 | 565.18 | 565.18 | 22,099.64 | 0.00 |
| WAN | OptiEncode exact | 10.53 | 44.63 | 55.16 | 63.94 | 516.34 | 26.56 |
| WAN | OptiEncode approximate | 10.71 | 26.05 | 36.77 | 42.65 | 357.58 | 17.70 |

Linear time covers convolutions, the final classifier and their linear processing. Other online time includes activations, pooling, residuals, secure argmax and remaining protocol work. The per-image totals exclude one-time initialization, model loading and key exchange. Traffic sums application bytes sent by both parties using decimal MB.

## Compact ciphertexts and offline HE on LAN

This separate series used the same LAN settings and 16 threads per party. The GPU was an NVIDIA RTX PRO 6000 Blackwell with a cached CUDA Graph. Each session processed three real images with batch size one. The means include the first image and exclude graph construction. Every configuration used preprocessed Ferret OT. The last row also moves the masked HE product into fresh per-image preprocessing.

| Backend and quantization | HE product | Sessions × images | Online linear (s) | Online total (s) | Preprocessing (s) | Online + preprocessing (s) | Online traffic (MB) | Preprocessing traffic (MB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CPU approximate | Online | 1 × 3 | 1.578 | 4.043 | 2.946 | 6.990 | 290.56 | 11.81 |
| GPU exact | Online | 1 × 3 | 1.493 | 5.328 | 5.561 | 10.888 | 449.32 | 20.66 |
| GPU approximate | Online | 2 × 3 | 1.459 | 3.740 | 3.038 | 6.778 | 290.56 | 11.81 |
| GPU approximate | Offline | 1 × 3 | 0.062 | 2.246 | 4.005 | 6.252 | 179.49 | 128.32 |

## ImageNetV2 accuracy

Both protocols processed the same 3,000 distinct images from ImageNetV2 matched-frequency, covering 970 classes. We combined two non-overlapping samples of 1,500 by counting hits across images. Exact inference was checked against the integer reference per image. Approximate truncation uses randomized rounding.

| Model and execution | Top-1 | Top-5 |
|---|---:|---:|
| Original FP32 | 57.00% (1710/3000) | 79.67% (2390/3000) |
| AIMET weights, floating-point activations | 56.43% (1693/3000) | 79.63% (2389/3000) |
| Exact integer plaintext | 54.00% (1620/3000) | 76.70% (2301/3000) |
| Approximate integer plaintext simulation | 54.70% (1641/3000) | 77.50% (2325/3000) |
| Exact two-party inference | 54.00% (1620/3000) | 76.70% (2301/3000) |
| Approximate two-party inference | 54.60% (1638/3000) | 77.43% (2323/3000) |

The performance tables summarize archived runs. Their records do not identify an exact source commit. The experiment guide links to the maintained implementation.
