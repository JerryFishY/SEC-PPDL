"""Generate public Falcon plans for a depthwise-only, matched-transport benchmark."""
import argparse
from functools import lru_cache
from math import isqrt
from pathlib import Path


def ceildiv(a, b):
    return (a + b - 1) // b


@lru_cache(None)
def channel_span(cx, cw):
    # Symmetric high/low order of Fig. 10, divided into weight polynomials.
    # Register both the last extracted coefficient and the lower boundary for
    # negacyclic wrap; this also handles odd and singleton groups.
    order = [cx - 1 - i // 2 if i % 2 == 0 else i // 2 for i in range(cx)]
    span = cx
    for start in range(0, cx, cw):
        lanes = order[start:start + cw]
        offset = 0
        outputs = [lanes[0]]
        for prev, lane in zip(lanes, lanes[1:]):
            offset += 1 + max(prev, cx - 1 - lane)
            outputs.append(offset + lane)
        span = max(span, 1 + max(outputs), cx + offset - min(outputs))
    return span


@lru_cache(None)
def channel_plans(channels, capacity, wire):
    best = None
    for cx in range(1, min(channels, capacity) + 1):
        for cw in range(1, min(cx, (2 * capacity + 2) // (cx + 2)) + 1):
            if cx % cw or channel_span(cx, cw) > capacity:
                continue
            inputs = ceildiv(channels, cx)
            # A tail block keeps the same lane ordering. Count nonempty weight groups.
            tail = channels % cx
            count = (channels // cx) * (cx // cw)
            if tail:
                order = [cx-1-i//2 if i % 2 == 0 else i//2 for i in range(cx)]
                count += sum(min(order[i:i+cw]) < tail for i in range(0, cx, cw))
            score = (128 * inputs + 50 * count if wire else inputs + count, inputs, count)
            if best is None or score < best[0]:
                best = score, cx, cw
    return best


def falcon_plan(n, h, channels, stride, wire):
    side = ceildiv(h, stride)
    best = None
    for rows in range(1, side + 1):
        for cols in range(rows, side + 1):
            area = ((rows - 1) * stride + 3) * ((cols - 1) * stride + 3)
            if area > n:
                continue
            score, cx, cw = channel_plans(channels, n // area, wire)
            tiles = ceildiv(side, rows) * ceildiv(side, cols)
            cost = (tiles * score[0], tiles, -rows * cols)
            if best is None or cost < best[0]:
                best = cost, (rows, cols, cx, cw, tiles * score[1], tiles * score[2])
    return best[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph', type=Path)
    parser.add_argument('degree', type=int, choices=(4096, 8192))
    parser.add_argument('output', type=Path)
    parser.add_argument('--match-spatial', action='store_true',
                        help='Use the current OptiEncode output tile on both sides at N=8192')
    args = parser.parse_args()
    layers = [tuple(map(int, (v[2], v[3], v[6]))) for line in args.graph.read_text().splitlines()
              if (v := line.split()) and v[0] == 'dwconv6']
    if len(layers) != 17:
        raise ValueError("Expected 17 MobileNetV2 depthwise layers")
    if args.degree == 4096:
        layers = [(28,192,1), (14,384,1), (14,576,1), (7,960,1)]
    with args.output.open('w') as out:
        for h, channels, stride in layers:
            # At N=4096 reproduce the paper's channel-only search over full feature maps.
            # At N=8192 additionally search rectangular spatial tiles and matched wire bytes.
            if args.degree == 4096:
                side = ceildiv(h, stride)
                _, cx, cw = channel_plans(channels, args.degree // (h + 2)**2, False)
                plan = (side, side, cx, cw)
            elif args.match_spatial:
                side = min(ceildiv(h, stride), (isqrt(args.degree) - 3) // stride + 1)
                patch = (side - 1) * stride + 3
                _, cx, cw = channel_plans(channels, args.degree // (patch * patch), True)
                plan = (side, side, cx, cw)
            else:
                plan = falcon_plan(args.degree, h, channels, stride, True)[:4]
            print(h, channels, stride, *plan, file=out)


if __name__ == '__main__':
    main()
