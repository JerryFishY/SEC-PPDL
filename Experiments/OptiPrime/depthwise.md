# Packed depthwise convolution

This extension supports a distinct spatial kernel for each input channel and a channel multiplier of one. It uses the input tiles, halo padding, compact output transport and BFV backends of OptiEncode.

## Derivation

Work in the negacyclic ring with $X^N=-1$. Let $C$ be a power of two with $C^2\mid 2N$. Write one packed input and its filters as

$$
A=\sum_{j=0}^{C-1}X^j A_j,\qquad W=\sum_{j=0}^{C-1}X^j W_j,
$$

where the exponents of $A_j,W_j$ are multiples of $C$. For a patch of width $p$, a tap at $(r,s)$ has exponent $-C(rp+s)$ in $W_j$. This implements the cross-correlation used by PyTorch. Unused lanes and virtual input padding contain zeros.

Set $g=1+2N/C$, $\sigma(X)=X^g$ and $T(f)=\sum_{k=0}^{C-1}\sigma^k(f)$. On a monomial $X^d$, the sum is $C X^d$ when $C\mid d$. Otherwise its terms cancel in pairs because $X^N=-1$ and $g^k=1+k(2N/C)\pmod{2N}$. Thus $T(X^{-j}A)=C A_j$ and $T(X^{-j}W)=C W_j$. Multiplying the two projections and shifting back by $X^j$ gives $C^2 X^j A_j W_j$.

The server can construct each $W_j$ directly from its local kernel. Distributing the remaining input trace gives the equivalent identity

$$
C Y=\sum_{k=0}^{C-1}\sigma^k(A)D_k,\qquad
D_k=\sum_{j=0}^{C-1} X^{j(1-g^k)} W_j,\qquad
Y=\sum_{j=0}^{C-1} X^j A_j W_j .
$$

The server encodes the $D_k$ offline and uses the existing baby-step/giant-step evaluator. Each output block consumes only its matching input block. There are $C\lceil C_{\rm in}/C\rceil$ candidate weight plaintexts per layer. Spatial tiles reuse them. For odd plaintext modulus $t$, packing can include $C^{-1}\bmod t$. With $t=2^b$, the operator uses the existing ciphertext-modulus normalization before the Galois operations. At stride $s$, output position $(u,v)$ of lane $j$ is read at $C s(up+v)+j$ within its tile.

## Preparing weights for BSGS

Choose $B\mid C$, let $G=C/B$, and split $k=bB+a$ with $0\le a<B$. In a general BSGS evaluation the inner plaintext for this term is $P_{b,a}=\sigma^{-bB}(D_{bB+a})$. Here $g^k=1+k(2N/C)\pmod{2N}$ and $C\mid 2N/C$. Every exponent in $D_k$ is therefore divisible by $C$ and is fixed by $\sigma$. This gives $P_{b,a}=D_{bB+a}$ for every split.

For each output block, allocate one zero polynomial per $k$. For each valid local channel $j$ and kernel tap $(r,s)$, add $w_{j,r,s}$ at the signed negacyclic exponent

$$
e=-C(rp+s)-(2N/C)\,j\,(bB+a)\pmod{2N}.
$$

Use coefficient $e\bmod N$ and negate the contribution when the reduced exponent is at least $N$. Add collisions modulo $t$. At odd $t$ the plaintext-normalized path scales each contribution by $C^{-1}\bmod t$. The ciphertext-normalized path uses unscaled weights and normalizes the reconstructed input in $R_q$ before any key switch. Transform each nonzero prepared plaintext to NTT form once. Changing $B$ changes only the grouping of the same $C$ plaintexts.

Compute $U_a=\sigma^a(A)$ once per tile and input block, and accumulate

$$
S_b=\sum_{a=0}^{B-1} U_a D_{bB+a},\qquad
C Y=\sum_{b=0}^{G-1}\sigma^{bB}(S_b)
$$

before accounting for the chosen normalization. A descending Horner loop uses only the giant-step key for $\sigma^B$. With all diagonals nonzero, the operation count per tile and block is $B-1$ baby automorphisms, $G-1$ giant automorphisms and $C$ ciphertext–plaintext products. The default power-of-two split minimizes $B+G$. For $C=64$, $B=G=8$ gives 14 automorphisms and 64 products. On GPU each depthwise output uses a scratch array of $B$ baby ciphertexts. On CPU independent tile/block pairs run in parallel when enough are available, with an alternative schedule that parallelizes giant sums for small layers.

## Calling the operator

Supply a weight tensor shaped `[channels, 1, kernel, kernel]` and one bias per channel. Initialize both parties with the same options and public shape.

```cpp
LinearLayer::OptiEncodeOptions options;
options.depthwise = true;
options.threads = 16;
options.ciphertext_inverse_normalization = true;  // For a power-of-two plaintext modulus.
LinearLayer::Conv2DOptiEncode conv(side, stride, padding, weights, bias, he, options);
auto result = conv(input_shares);
```

The HE context selects CPU or GPU execution. Optional `PrepareInput(true)` prepares a fresh masked product for one inference. Its online correction uses only the matching channel on either backend. Run `python3 Experiments/OptiPrime/verify_depthwise.py` with NumPy to check both polynomial formulas against direct spatial convolution. The existing two-party `test_optiencode_conv` executable also exercises depthwise kernels.
