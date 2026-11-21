//! The operations a transformer block is built from, forward and backward:
//! embedding lookup, layer norm, GELU, softmax, causal mask, batched matrix
//! product and axis permutation.

use crate::tensor::Tensor;

/// Rows of `table` (V, D) selected by `ids`: output (ids.len(), D).
pub(crate) fn embedding(table: &Tensor, ids: &[usize]) -> Tensor {
    let (v, d) = table.dims2();
    let mut out = Vec::with_capacity(ids.len() * d);
    for &id in ids {
        assert!(id < v, "embedding: id {id} out of range for {v} rows");
        out.extend_from_slice(&table.data()[id * d..(id + 1) * d]);
    }
    Tensor::new(vec![ids.len(), d], out)
}

/// Gradient of the table: each output row's gradient adds into the row it came from.
pub(crate) fn embedding_backward(table: &Tensor, ids: &[usize], g: &Tensor) -> Tensor {
    let (_, d) = table.dims2();
    let mut dt = vec![0.0f32; table.numel()];
    for (row, &id) in g.data().chunks_exact(d).zip(ids) {
        for (acc, v) in dt[id * d..(id + 1) * d].iter_mut().zip(row) {
            *acc += v;
        }
    }
    Tensor::new(table.shape().to_vec(), dt)
}

fn last_dim(t: &Tensor, what: &str) -> usize {
    *t.shape()
        .last()
        .unwrap_or_else(|| panic!("{what}: needs at least one dimension"))
}

/// Mean and 1/sqrt(variance + eps) of one row (biased variance, as PyTorch).
fn row_stats(row: &[f32], eps: f32) -> (f32, f32) {
    let n = row.len() as f32;
    let mean = row.iter().sum::<f32>() / n;
    let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    (mean, 1.0 / (var + eps).sqrt())
}

/// Normalizes over the last dimension, then `* gamma + beta`.
pub(crate) fn layer_norm(x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> Tensor {
    let d = last_dim(x, "layer_norm");
    assert_eq!(gamma.shape(), &[d], "layer_norm: gamma shape must be [{d}]");
    assert_eq!(beta.shape(), &[d], "layer_norm: beta shape must be [{d}]");
    let mut out = Vec::with_capacity(x.numel());
    for row in x.data().chunks_exact(d) {
        let (mean, rstd) = row_stats(row, eps);
        for ((v, gm), bt) in row.iter().zip(gamma.data()).zip(beta.data()) {
            out.push((v - mean) * rstd * gm + bt);
        }
    }
    Tensor::new(x.shape().to_vec(), out)
}

/// Gradients (dx, dgamma, dbeta). With x̂ the normalized row and h = g * gamma:
/// dx = rstd * (h - mean(h) - x̂ * mean(h * x̂)).
pub(crate) fn layer_norm_backward(
    x: &Tensor,
    gamma: &Tensor,
    eps: f32,
    g: &Tensor,
) -> (Tensor, Tensor, Tensor) {
    let d = last_dim(x, "layer_norm");
    let mut dx = Vec::with_capacity(x.numel());
    let mut dgamma = vec![0.0f32; d];
    let mut dbeta = vec![0.0f32; d];
    let mut xhat = vec![0.0f32; d];
    let mut h = vec![0.0f32; d];
    for (row, grow) in x.data().chunks_exact(d).zip(g.data().chunks_exact(d)) {
        let (mean, rstd) = row_stats(row, eps);
        for j in 0..d {
            xhat[j] = (row[j] - mean) * rstd;
            h[j] = grow[j] * gamma.data()[j];
            dgamma[j] += grow[j] * xhat[j];
            dbeta[j] += grow[j];
        }
        let mean_h = h.iter().sum::<f32>() / d as f32;
        let mean_hx = h.iter().zip(&xhat).map(|(a, b)| a * b).sum::<f32>() / d as f32;
        for j in 0..d {
            dx.push(rstd * (h[j] - mean_h - xhat[j] * mean_hx));
        }
    }
    (
        Tensor::new(x.shape().to_vec(), dx),
        Tensor::new(vec![d], dgamma),
        Tensor::new(vec![d], dbeta),
    )
}

const GELU_C: f32 = 0.797_884_6; // sqrt(2 / pi)
const GELU_A: f32 = 0.044_715;

/// GELU, tanh approximation (GPT-2; PyTorch `approximate="tanh"`):
/// 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))).
pub(crate) fn gelu(x: &Tensor) -> Tensor {
    let out = x
        .data()
        .iter()
        .map(|&v| 0.5 * v * (1.0 + (GELU_C * (v + GELU_A * v * v * v)).tanh()))
        .collect();
    Tensor::new(x.shape().to_vec(), out)
}

pub(crate) fn gelu_backward(x: &Tensor, g: &Tensor) -> Tensor {
    let out = x
        .data()
        .iter()
        .zip(g.data())
        .map(|(&v, &gv)| {
            let t = (GELU_C * (v + GELU_A * v * v * v)).tanh();
            let du = GELU_C * (1.0 + 3.0 * GELU_A * v * v);
            gv * (0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * du)
        })
        .collect();
    Tensor::new(x.shape().to_vec(), out)
}

/// Softmax over the last dimension, shifted by the row maximum. Entries of
/// -inf (masked) become exactly 0.
pub(crate) fn softmax(x: &Tensor) -> Tensor {
    let d = last_dim(x, "softmax");
    let mut out = Vec::with_capacity(x.numel());
    for row in x.data().chunks_exact(d) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let start = out.len();
        out.extend(row.iter().map(|&v| (v - max).exp()));
        let sum: f32 = out[start..].iter().sum();
        out[start..].iter_mut().for_each(|v| *v /= sum);
    }
    Tensor::new(x.shape().to_vec(), out)
}

/// dx = y * (g - sum(g * y)) per row, with y the softmax output.
pub(crate) fn softmax_backward(y: &Tensor, g: &Tensor) -> Tensor {
    let d = last_dim(y, "softmax");
    let mut out = Vec::with_capacity(y.numel());
    for (yr, gr) in y.data().chunks_exact(d).zip(g.data().chunks_exact(d)) {
        let dot: f32 = yr.iter().zip(gr).map(|(a, b)| a * b).sum();
        out.extend(yr.iter().zip(gr).map(|(yv, gv)| yv * (gv - dot)));
    }
    Tensor::new(y.shape().to_vec(), out)
}

/// For square trailing matrices (..., t, t): true where column > row, the
/// positions a token must not see.
fn is_future(shape: &[usize], index: usize) -> bool {
    let t = shape[shape.len() - 1];
    let (row, col) = ((index / t) % t, index % t);
    col > row
}

/// Sets every future position (column > row) to -inf.
pub(crate) fn causal_mask(x: &Tensor) -> Tensor {
    let s = x.shape();
    assert!(
        s.len() >= 2 && s[s.len() - 1] == s[s.len() - 2],
        "causal_mask: needs square trailing matrices, got {s:?}"
    );
    let out = x
        .data()
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            if is_future(s, i) {
                f32::NEG_INFINITY
            } else {
                v
            }
        })
        .collect();
    Tensor::new(s.to_vec(), out)
}

/// The gradient passes through visible positions and is 0 at masked ones.
pub(crate) fn causal_mask_backward(g: &Tensor) -> Tensor {
    let s = g.shape();
    let out = g
        .data()
        .iter()
        .enumerate()
        .map(|(i, &v)| if is_future(s, i) { 0.0 } else { v })
        .collect();
    Tensor::new(s.to_vec(), out)
}

fn dims3(t: &Tensor, what: &str) -> (usize, usize, usize) {
    match t.shape()[..] {
        [a, b, c] => (a, b, c),
        _ => panic!("{what}: expected 3 dimensions, got shape {:?}", t.shape()),
    }
}

/// `a @ b` for each of B matrix pairs: (B, n, k) @ (B, k, m) -> (B, n, m).
/// With `ta`/`tb`, the trailing matrices of a/b are used transposed.
pub(crate) fn bmm(a: &Tensor, b: &Tensor, ta: bool, tb: bool) -> Tensor {
    let (ba, ar, ac) = dims3(a, "batched_matmul");
    let (bb, br, bc) = dims3(b, "batched_matmul");
    assert_eq!(ba, bb, "batched_matmul: batch sizes differ");
    let (n, k) = if ta { (ac, ar) } else { (ar, ac) };
    let (k2, m) = if tb { (bc, br) } else { (br, bc) };
    assert_eq!(
        k,
        k2,
        "batched_matmul: shape mismatch {:?} @ {:?}",
        a.shape(),
        b.shape()
    );
    let at = |bi: usize, i: usize, p: usize| {
        a.data()[bi * ar * ac + if ta { p * ac + i } else { i * ac + p }]
    };
    let bt = |bi: usize, p: usize, j: usize| {
        b.data()[bi * br * bc + if tb { j * bc + p } else { p * bc + j }]
    };
    let mut out = vec![0.0f32; ba * n * m];
    for bi in 0..ba {
        for i in 0..n {
            for j in 0..m {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += at(bi, i, p) * bt(bi, p, j);
                }
                out[(bi * n + i) * m + j] = acc;
            }
        }
    }
    Tensor::new(vec![ba, n, m], out)
}

/// Reorders axes: output axis i is input axis `axes[i]`, as `torch.permute`.
pub(crate) fn permute(x: &Tensor, axes: &[usize]) -> Tensor {
    let s = x.shape();
    let mut seen = vec![false; s.len()];
    assert_eq!(axes.len(), s.len(), "permute: one axis per dimension");
    for &a in axes {
        assert!(
            a < s.len() && !seen[a],
            "permute: {axes:?} is not a permutation"
        );
        seen[a] = true;
    }
    let out_shape: Vec<usize> = axes.iter().map(|&a| s[a]).collect();
    let mut in_strides = vec![1usize; s.len()];
    for i in (0..s.len().saturating_sub(1)).rev() {
        in_strides[i] = in_strides[i + 1] * s[i + 1];
    }
    let strides: Vec<usize> = axes.iter().map(|&a| in_strides[a]).collect();
    let mut out = Vec::with_capacity(x.numel());
    let mut idx = vec![0usize; s.len()];
    for _ in 0..x.numel() {
        out.push(
            x.data()[idx
                .iter()
                .zip(&strides)
                .map(|(i, st)| i * st)
                .sum::<usize>()],
        );
        for d in (0..idx.len()).rev() {
            idx[d] += 1;
            if idx[d] < out_shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    Tensor::new(out_shape, out)
}

/// The permutation that undoes `axes`.
pub(crate) fn inverse_axes(axes: &[usize]) -> Vec<usize> {
    let mut inv = vec![0; axes.len()];
    for (i, &a) in axes.iter().enumerate() {
        inv[a] = i;
    }
    inv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permute_transposes_a_matrix() {
        let x = Tensor::new(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]);
        assert_eq!(permute(&x, &[1, 0]).data(), &[1., 4., 2., 5., 3., 6.]);
    }

    #[test]
    fn permute_then_inverse_is_identity() {
        let x = Tensor::new(vec![2, 3, 4], (0..24).map(|v| v as f32).collect());
        let axes = [2, 0, 1];
        assert_eq!(permute(&permute(&x, &axes), &inverse_axes(&axes)), x);
    }

    #[test]
    fn softmax_of_masked_row_puts_zero_on_minus_infinity() {
        let x = Tensor::new(vec![1, 3], vec![0.0, f32::NEG_INFINITY, 0.0]);
        assert_eq!(softmax(&x).data(), &[0.5, 0.0, 0.5]);
    }

    #[test]
    fn transposed_bmm_matches_explicit_permute() {
        let a = Tensor::new(vec![2, 2, 3], (0..12).map(|v| v as f32 - 5.0).collect());
        let b = Tensor::new(vec![2, 2, 3], (0..12).map(|v| (v * 3 % 7) as f32).collect());
        let bt = permute(&b, &[0, 2, 1]);
        assert_eq!(bmm(&a, &b, false, true), bmm(&a, &bt, false, false));
    }
}
