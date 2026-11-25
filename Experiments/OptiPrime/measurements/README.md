# OptiPrime measurements on SEC-PPDL

These measurements accompany the [OptiPrime project demo and communication chart](https://jerryfishy.github.io/OptiPrime/). The [MICRO paper](https://arxiv.org/html/2609.16898v1#S5) describes the experiments reported in the publication.

## CPU and network setup

Both parties ran on one AMD EPYC 9J14 host with 16 CPU worker threads per party. Linux `tc` shaped the Docker loopback connection in both directions. LAN used 3,072 Mbit/s per direction and 0.3 ms RTT. WAN used 352 Mbit/s per direction and 40 ms RTT.

For the demo and chart, both methods ran HE online and prepared fresh Ferret OT correlations for each image. End-to-end latency includes that preparation and online inference through secure argmax. Timing excludes model setup, key exchange, OT planning and opening the output for verification. Communication is the sum of application bytes sent by both parties, in decimal GB.

| Measurement | Sample and records |
|---|---|
| ResNet-18 demo and CPU latency | The same 16 ImageNetV2 images per method and network for ResNet-18. Other models have one LAN observation per method. [CPU results](cpu-inference.md) · [Per-image JSON](cpu-inference.json) |
| Project-page communication chart | One matched ImageNetV2 image per model and method. The ResNet-18 bars use this single-image run. The CPU latency table reports the 16-image mean. [Communication results](communication.md) |
| Earlier ResNet-18 experiments | Separate configurations for CPU/GPU inference, offline HE and accuracy on 3,000 images. The report specifies the OT and HE settings for each series. [Archived measurements](sec-ppdl.md) |
