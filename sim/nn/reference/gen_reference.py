#!/usr/bin/env python3
"""Generate PyTorch reference values for the Rust network simulator.

Writes tests/data/*.ref, which the Rust tests read. The files are committed,
so `cargo test` runs without PyTorch; rerun this script only to change the
cases. Seeds are fixed, so the output is reproducible for a given PyTorch.

File format (little-endian), read by src/refdata.rs:
    b"CSREF001", u32 tensor count, then per tensor:
    u32 name length, name (UTF-8), u32 ndim, u32 dims[ndim], f32 data[prod(dims)]
Integer data (class labels) is stored as f32; it is exact for small values.

Layout convention: a linear layer computes x @ W + b with W of shape (in, out),
so nn.Linear weights are stored transposed.
"""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent.parent / "tests" / "data"


def write_ref(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    with path.open("wb") as f:
        f.write(b"CSREF001")
        f.write(struct.pack("<I", len(tensors)))
        for name, t in tensors.items():
            t = t.detach().to(torch.float32).contiguous()
            raw = name.encode()
            f.write(struct.pack("<I", len(raw)) + raw)
            f.write(struct.pack("<I", t.dim()) + struct.pack(f"<{t.dim()}I", *t.shape))
            values = t.view(-1).tolist()
            f.write(struct.pack(f"<{len(values)}f", *values))
    print(f"{path.name}: {len(tensors)} tensors")


def leaf(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, requires_grad=True)


def ops_cases() -> dict[str, torch.Tensor]:
    """Each op: inputs, an upstream gradient G, the output and the input gradients."""
    torch.manual_seed(1)
    out: dict[str, torch.Tensor] = {}

    a, b = leaf(4, 5), leaf(5, 3)
    y = a @ b
    g = torch.randn_like(y)
    y.backward(g)
    out |= {"matmul.a": a, "matmul.b": b, "matmul.g": g, "matmul.y": y, "matmul.da": a.grad, "matmul.db": b.grad}

    x, bias = leaf(4, 3), leaf(3)
    y = x + bias  # bias broadcast over rows
    g = torch.randn_like(y)
    y.backward(g)
    out |= {"add_bias.x": x, "add_bias.b": bias, "add_bias.g": g, "add_bias.y": y,
            "add_bias.dx": x.grad, "add_bias.db": bias.grad}

    x = torch.randn(6, 7)
    x[0, :3] = 0.0  # ReLU at exactly zero: PyTorch's gradient there is 0
    x.requires_grad_(True)
    y = F.relu(x)
    g = torch.randn_like(y)
    y.backward(g)
    out |= {"relu.x": x, "relu.g": g, "relu.y": y, "relu.dx": x.grad}

    logits = leaf(8, 5)
    targets = torch.randint(0, 5, (8,))
    loss = F.cross_entropy(logits, targets)  # mean over the batch
    loss.backward()
    out |= {"cross_entropy.logits": logits, "cross_entropy.targets": targets.float(),
            "cross_entropy.loss": loss.reshape(1), "cross_entropy.dlogits": logits.grad}
    return out


def spiral(n_per_class: int, classes: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """The classic 2-D spiral dataset: not linearly separable, so the hidden layer matters."""
    xs, ys = [], []
    for c in range(classes):
        r = torch.linspace(0.0, 1.0, n_per_class)
        theta = torch.linspace(c * 4.0, (c + 1) * 4.0, n_per_class) + 0.2 * torch.randn(n_per_class, generator=gen)
        xs.append(torch.stack([r * torch.sin(theta), r * torch.cos(theta)], dim=1))
        ys.append(torch.full((n_per_class,), c))
    return torch.cat(xs), torch.cat(ys)


def mlp_case() -> dict[str, torch.Tensor]:
    """A 2-16-3 MLP trained with full-batch SGD: initial weights, every loss, final weights."""
    gen = torch.Generator().manual_seed(2)
    torch.manual_seed(2)
    x, y = spiral(32, 3, gen)
    l1, l2 = torch.nn.Linear(2, 16), torch.nn.Linear(16, 3)
    init = {"mlp.w1": l1.weight.T.clone(), "mlp.b1": l1.bias.clone(),
            "mlp.w2": l2.weight.T.clone(), "mlp.b2": l2.bias.clone()}
    opt = torch.optim.SGD(list(l1.parameters()) + list(l2.parameters()), lr=0.5)
    steps, losses = 100, []
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(l2(F.relu(l1(x))), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert math.isfinite(losses[-1]) and losses[-1] < losses[0], losses
    final = {"mlp.final.w1": l1.weight.T, "mlp.final.b1": l1.bias,
             "mlp.final.w2": l2.weight.T, "mlp.final.b2": l2.bias}
    return {"mlp.x": x, "mlp.y": y.float(), "mlp.lr": torch.tensor([0.5]), **init,
            "mlp.losses": torch.tensor(losses), **final}


def conv_cases() -> dict[str, torch.Tensor]:
    """conv2d, max_pool2d and flatten, forward and backward. Tensors are NCHW;
    conv weights are (C_out, C_in, kH, kW), as in PyTorch."""
    torch.manual_seed(3)
    out: dict[str, torch.Tensor] = {}

    # name: (N, C_in, H, W, C_out, kH, kW, stride, padding)
    convs = {"conv_s1p0": (2, 3, 6, 7, 4, 3, 2, 1, 0), "conv_s2p1": (2, 2, 7, 7, 3, 3, 3, 2, 1)}
    for name, (n, cin, h, w, cout, kh, kw, stride, pad) in convs.items():
        x, wt, b = leaf(n, cin, h, w), leaf(cout, cin, kh, kw), leaf(cout)
        y = F.conv2d(x, wt, b, stride=stride, padding=pad)
        g = torch.randn_like(y)
        y.backward(g)
        out |= {f"{name}.x": x, f"{name}.w": wt, f"{name}.b": b, f"{name}.g": g, f"{name}.y": y,
                f"{name}.dx": x.grad, f"{name}.dw": wt.grad, f"{name}.db": b.grad,
                f"{name}.params": torch.tensor([stride, pad], dtype=torch.float32)}

    # Ties: PyTorch sends a window's gradient to the first maximum in row-major order.
    x = torch.randn(2, 3, 6, 6)
    x[0, 0, 0:2, 0:2] = 0.0          # a window of equal values, as after ReLU
    x[0, 1, 2, 2] = x[0, 1, 2, 3] = x[0, 1, 3, 2] = 5.0  # three equal maxima
    x.requires_grad_(True)
    y = F.max_pool2d(x, kernel_size=2, stride=2)
    g = torch.randn_like(y)
    y.backward(g)
    out |= {"maxpool.x": x, "maxpool.g": g, "maxpool.y": y, "maxpool.dx": x.grad}

    x = leaf(2, 3, 2, 2)
    y = x.flatten(1)
    g = torch.randn_like(y)
    y.backward(g)
    out |= {"flatten.x": x, "flatten.g": g, "flatten.y": y, "flatten.dx": x.grad}
    return out


def bars(n_per_class: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """8x8 images of a horizontal bar, a vertical bar or a diagonal, at random
    positions, plus noise: telling them apart needs spatial filters."""
    xs, ys = [], []
    for c in range(3):
        for _ in range(n_per_class):
            img = 0.3 * torch.randn(1, 8, 8, generator=gen)
            k = int(torch.randint(1, 7, (1,), generator=gen))
            if c == 0:
                img[0, k, 1:7] += 1.0
            elif c == 1:
                img[0, 1:7, k] += 1.0
            else:
                idx = torch.arange(1, 7)
                img[0, idx, idx] += 1.0
            xs.append(img)
            ys.append(c)
    return torch.stack(xs), torch.tensor(ys)


def cnn_case() -> dict[str, torch.Tensor]:
    """conv(1->4, 3x3, pad 1) -> ReLU -> maxpool 2 -> flatten -> linear(64->3),
    trained with full-batch SGD: initial weights, every loss, final weights."""
    gen = torch.Generator().manual_seed(4)
    torch.manual_seed(4)
    x, y = bars(16, gen)
    conv, lin = torch.nn.Conv2d(1, 4, 3, padding=1), torch.nn.Linear(64, 3)
    init = {"cnn.conv_w": conv.weight.clone(), "cnn.conv_b": conv.bias.clone(),
            "cnn.lin_w": lin.weight.T.clone(), "cnn.lin_b": lin.bias.clone()}
    lr = 0.1
    opt = torch.optim.SGD(list(conv.parameters()) + list(lin.parameters()), lr=lr)
    losses = []
    for _ in range(60):
        opt.zero_grad()
        h = F.max_pool2d(F.relu(conv(x)), 2)
        loss = F.cross_entropy(lin(h.flatten(1)), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert math.isfinite(losses[-1]) and losses[-1] < 0.8 * losses[0], losses
    final = {"cnn.final.conv_w": conv.weight, "cnn.final.conv_b": conv.bias,
             "cnn.final.lin_w": lin.weight.T, "cnn.final.lin_b": lin.bias}
    return {"cnn.x": x, "cnn.y": y.float(), "cnn.lr": torch.tensor([lr]), **init,
            "cnn.losses": torch.tensor(losses), **final}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)  # one fixed summation order on every machine
    write_ref(OUT / "ops.ref", ops_cases())
    write_ref(OUT / "mlp.ref", mlp_case())
    write_ref(OUT / "conv.ref", conv_cases())
    write_ref(OUT / "cnn.ref", cnn_case())
    (OUT / "TORCH_VERSION").write_text(torch.__version__ + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
