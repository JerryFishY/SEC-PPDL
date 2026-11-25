"""Check the trace derivation without inverting C in the plaintext ring."""
import numpy as np


def mono(a, exponent):
    n = len(a)
    shifted = np.roll(a, exponent % n)
    shifted[:exponent % n] *= -1
    return -shifted if (exponent // n) % 2 else shifted


def sigma(a, element):
    n = len(a)
    powers = np.arange(n) * element
    out = np.empty_like(a)
    out[powers % n] = a * (1 - 2 * ((powers // n) % 2))
    return out


def multiply(a, b):
    n = len(a)
    product = np.convolve(a, b)
    out = product[:n].copy()
    out[:-1] -= product[n:]
    return out


def verify(n, c, total, stride, pad, seed):
    rng = np.random.default_rng(seed)
    h, w, kernel = 5, 7, 3
    x = rng.integers(-8, 9, (total, h, w))
    weights = rng.integers(-7, 8, (total, kernel, kernel))
    padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel, kernel), axis=(1, 2))
    ref = np.einsum("chwrs,crs->chw", windows[:, ::stride, ::stride], weights)
    patch_width = padded.shape[2]
    assert c * padded.shape[1] * patch_width <= n
    elements = [pow(1 + 2 * n // c, k, 2 * n) for k in range(c)]

    def trace(poly):
        return sum(sigma(poly, g) for g in elements)

    one = np.zeros(n, dtype=np.int64)
    one[0] = 1
    for first in range(0, total, c):
        a = np.zeros(n, dtype=np.int64)
        weight = np.zeros_like(a)
        lanes = []
        for j in range(min(c, total - first)):
            pixels = padded[first + j].ravel()
            a[j:j + c * len(pixels):c] = pixels
            taps = np.zeros_like(a)
            for r, s in np.ndindex(kernel, kernel):
                taps += mono(one * weights[first + j, r, s],
                             -c * (r * patch_width + s))
            lanes.append(taps)
            weight += mono(taps, j)
        explicit = np.zeros_like(a)
        compact = np.zeros_like(a)
        for j, taps in enumerate(lanes):
            isolated = trace(mono(weight, -j))
            np.testing.assert_array_equal(isolated, c * taps)
            explicit += mono(multiply(trace(mono(a, -j)), isolated), j)
        diagonals = []
        for g in elements:
            diagonal = sum(mono(taps, j * (1 - g)) for j, taps in enumerate(lanes))
            diagonals.append(diagonal)
            compact += multiply(sigma(a, g), diagonal)
        np.testing.assert_array_equal(explicit, c * compact)
        # Derive the weights independently from tap coordinates and the closed-form phase.
        direct = [np.zeros_like(a) for _ in elements]
        for k, diagonal in enumerate(direct):
            for j in range(len(lanes)):
                for r, s in np.ndindex(kernel, kernel):
                    e = -c * (r * patch_width + s) - (2 * n // c) * j * k
                    diagonal[e % n] += (-1 if (e // n) % 2 else 1) * weights[first + j, r, s]
            np.testing.assert_array_equal(diagonal, diagonals[k])
            np.testing.assert_array_equal(sigma(diagonal, elements[1 % c]), diagonal)
        for power in range(c.bit_length()):
            baby = 1 << power
            babies = [sigma(a, g) for g in elements[:baby]]
            accum = np.zeros_like(a)
            for start in reversed(range(0, c, baby)):
                accum = sigma(accum, elements[baby % c])
                for offset, rotated in enumerate(babies):
                    accum += multiply(rotated, direct[start + offset])
            np.testing.assert_array_equal(accum, compact)
        for j in range(len(lanes)):
            for y, z in np.ndindex(ref.shape[1:]):
                index = c * stride * (y * patch_width + z) + j
                assert compact[index] == c * ref[first + j, y, z]


if __name__ == "__main__":
    for seed in range(3):
        for c in (1, 2, 4, 8):
            for stride in (1, 2):
                for pad in (0, 1):
                    verify(1024, c, c + 1, stride, pad, seed)
    print("96 cases passed: trace, direct weights, all BSGS splits, and spatial cross-correlation")
