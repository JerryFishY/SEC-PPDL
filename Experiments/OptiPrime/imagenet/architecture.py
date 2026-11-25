"""Public ImageNet topology shared by calibration, clear inference and the MPC bundle."""

from collections import Counter
from dataclasses import dataclass


@dataclass
class Step:
    op: str
    src: int
    name: str = ""
    shape: tuple = ()
    signed: bool = False
    skip: int = 0
    projection: bool = False
    depthwise: bool = False
    relu6: bool = False


BLOCKS = {
    "resnet18": (2, 2, 2, 2),
    "resnet34": (3, 4, 6, 3),
    "resnet50": (3, 4, 6, 3),
    "vgg16": (2, 2, 3, 3, 3),
    "vgg19": (2, 2, 4, 4, 4),
}


MODELS = (*BLOCKS, "mobilenet_v2")


def topology(arch):
    if arch not in MODELS:
        raise ValueError(f"Unsupported model {arch}")
    steps, shapes = [], [(3, 224)]

    def append(step, shape):
        steps.append(step)
        shapes.append(shape)
        return len(steps)

    def conv(src, name, co, kernel, stride=1, pad=0, signed=False, depthwise=False, relu6=False):
        ci, h = shapes[src]
        s = (h, ci, co, kernel, stride, pad)
        return append(
            Step("conv", src, name, s, signed, depthwise=depthwise, relu6=relu6), (co, (h + 2 * pad - kernel) // stride + 1)
        )

    if arch == "mobilenet_v2":
        x = conv(0, "features.0.0", 32, 3, 2, 1, relu6=True)
        index = 1
        for expansion, channels, repeats, stride in (
            (1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2),
            (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1),
        ):
            for block in range(repeats):
                skip = x
                ci = shapes[x][0]
                prefix = f"features.{index}.conv"
                index += 1
                step_stride = stride if block == 0 else 1
                dw = int(expansion != 1)
                hidden = ci * expansion
                if expansion != 1:
                    x = conv(x, prefix + ".0.0", hidden, 1, relu6=True)
                x = conv(x, f"{prefix}.{dw}.0", hidden, 3, step_stride, 1,
                         depthwise=True, relu6=True)
                x = conv(x, f"{prefix}.{dw + 1}", channels, 1, signed=True)
                if step_stride == 1 and ci == channels:
                    x = append(Step("add", x, prefix + ".add", signed=True, skip=skip), shapes[x])
        x = conv(x, "features.18.0", 1280, 1, relu6=True)
        x = append(Step("mean", x), (1280, 1))
        conv(x, "classifier.1", 1000, 1)
    elif arch.startswith("vgg"):
        x = index = 0
        for repeats, channels in zip(BLOCKS[arch], (64, 128, 256, 512, 512)):
            for _ in range(repeats):
                x = conv(x, f"features.{index}", channels, 3, pad=1)
                index += 2
            x = append(Step("pool2", x), (channels, shapes[x][1] // 2))
            index += 1
        x = append(Step("flatten", x), (512 * 7 * 7, 1))
        for i, channels in [(0, 4096), (3, 4096), (6, 1000)]:
            x = conv(x, f"classifier.{i}", channels, 1)
    else:
        x = conv(0, "conv1", 64, 7, 2, 3)
        x = append(Step("pool3", x), (64, 56))
        bottleneck = arch == "resnet50"
        for i, n in enumerate(BLOCKS[arch]):
            width = 64 << i
            output = width * (4 if bottleneck else 1)
            for j in range(n):
                name = f"layer{i+1}.{j}"
                skip = x
                stride = 2 if i and j == 0 else 1
                project = shapes[x][0] != output or stride != 1
                if bottleneck:
                    x = conv(x, name + ".conv1", width, 1)
                    x = conv(x, name + ".conv2", width, 3, stride, 1)
                    x = conv(x, name + ".conv3", output, 1, signed=True)
                else:
                    x = conv(x, name + ".conv1", width, 3, stride, 1)
                    x = conv(x, name + ".conv2", output, 3, pad=1, signed=True)
                if project:
                    skip = conv(skip, name + ".downsample.0", output, 1, stride, signed=True)
                x = append(Step("add", x, name + ".add", skip=skip, projection=project), shapes[x])
        x = append(Step("mean", x), (shapes[x][0], 1))
        conv(x, "fc", 1000, 1)
    return steps


def forward(steps, image, conv, residual, mean, observe_layout=None):
    import torch.nn.functional as F

    values = {0: image}
    uses = Counter(s.src for s in steps)
    uses.update(s.skip for s in steps if s.op == "add")

    def take(i):
        uses[i] -= 1
        return values.pop(i) if uses[i] == 0 else values[i]

    for i, s in enumerate(steps, 1):
        x = take(s.src)
        if s.op == "conv":
            x = conv(s, x)
        elif s.op == "add":
            x = residual(s, x, take(s.skip))
        elif s.op == "pool3":
            x = F.max_pool2d(x, 3, 2, 1)
        elif s.op == "pool2":
            x = F.max_pool2d(x, 2, 2)
        elif s.op == "flatten":
            x = x.flatten(1)[:, :, None, None]
        elif s.op == "mean":
            x = mean(x)
        if observe_layout is not None and s.op not in ("conv", "add"):
            observe_layout(f"{i}/{s.op}", x)
        values[i] = x
    return x.flatten(1)
