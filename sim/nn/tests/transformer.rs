//! A one-block GPT (pre-norm, causal multi-head attention, GELU MLP) trained
//! for 80 full-batch SGD steps on a copy task, step by step against PyTorch.
mod common;
use common::{assert_close, labels, load, ATOL, RTOL};
use coresample_nn::models::{gpt_forward, GptConfig};
use coresample_nn::{sgd_step, Graph, Tensor};
use std::collections::HashMap;

#[test]
fn training_matches_pytorch() {
    let r = load("transformer.ref");
    let s: Vec<usize> = labels(&r["gpt.shape"]);
    let (batch, seq) = (s[0], s[1]);
    let cfg = GptConfig {
        vocab: s[2],
        seq,
        d: s[3],
        heads: s[4],
        hidden: s[5],
    };
    let inputs = labels(&r["gpt.inputs"]);
    let targets = labels(&r["gpt.targets"]);
    let lr = r["gpt.lr"].data()[0];
    let mut params: Vec<(&str, Tensor)> = GptConfig::PARAM_NAMES
        .iter()
        .map(|&n| (n, r[&format!("gpt.{n}")].clone()))
        .collect();
    let expected_losses = r["gpt.losses"].data();
    let mut max_loss_diff = 0.0f32;

    for (step, &expected) in expected_losses.iter().enumerate() {
        let mut g = Graph::new();
        let ids: HashMap<&str, _> = params
            .iter()
            .map(|(n, t)| (*n, g.param(t.clone())))
            .collect();
        let logits = gpt_forward(&cfg, &mut g, &ids, &inputs, batch);
        let loss = g.cross_entropy(logits, &targets);
        let got = g.value(loss).data()[0];
        max_loss_diff = max_loss_diff.max((got - expected).abs());
        assert!(
            (got - expected).abs() <= ATOL + RTOL * expected.abs(),
            "step {step}: loss {got}, PyTorch {expected}"
        );
        g.backward(loss);
        for (name, p) in params.iter_mut() {
            sgd_step(p, g.grad(ids[name]).unwrap(), lr);
        }
    }

    let mut max_weight_diff = 0.0f32;
    for (name, p) in &params {
        let expected = &r[&format!("gpt.final.{name}")];
        assert_close(name, p, expected);
        for (a, e) in p.data().iter().zip(expected.data()) {
            max_weight_diff = max_weight_diff.max((a - e).abs());
        }
    }
    eprintln!(
        "{} steps: max loss difference {max_loss_diff:e}, max final weight difference {max_weight_diff:e}",
        expected_losses.len()
    );
}

/// Gradients of a tiny GPT checked by central finite differences, with no
/// PyTorch involved: three entries (first, middle, last) of every parameter.
#[test]
fn gpt_gradients_match_finite_differences() {
    let cfg = GptConfig {
        vocab: 5,
        seq: 4,
        d: 8,
        heads: 2,
        hidden: 8,
    };
    let shapes: HashMap<&str, Vec<usize>> = HashMap::from([
        ("tok", vec![5, 8]),
        ("pos", vec![4, 8]),
        ("wq", vec![8, 8]),
        ("wk", vec![8, 8]),
        ("wv", vec![8, 8]),
        ("wo", vec![8, 8]),
        ("w1", vec![8, 8]),
        ("w2", vec![8, 8]),
        ("wout", vec![8, 5]),
        ("bout", vec![5]),
        ("b1", vec![8]),
    ]);
    let params: Vec<(&str, Tensor)> = GptConfig::PARAM_NAMES
        .iter()
        .enumerate()
        .map(|(k, &name)| {
            let shape = shapes.get(name).cloned().unwrap_or(vec![8]);
            let n: usize = shape.iter().product();
            let base = if name.ends_with("_g") { 1.0 } else { 0.0 };
            let data = (0..n)
                .map(|i| base + (((i * 7 + k * 13) % 17) as f32 - 8.0) / 20.0)
                .collect();
            (name, Tensor::new(shape, data))
        })
        .collect();
    let tokens = [1usize, 4, 2, 0, 3, 3, 1, 2];
    let targets = [4usize, 2, 0, 1, 3, 1, 2, 4];

    let loss_of = |params: &[(&str, Tensor)]| -> (f32, Vec<Tensor>) {
        let mut g = Graph::new();
        let ids: HashMap<&str, _> = params
            .iter()
            .map(|(n, t)| (*n, g.param(t.clone())))
            .collect();
        let logits = gpt_forward(&cfg, &mut g, &ids, &tokens, 2);
        let loss = g.cross_entropy(logits, &targets);
        g.backward(loss);
        let grads = params
            .iter()
            .map(|(n, _)| g.grad(ids[n]).unwrap().clone())
            .collect();
        (g.value(loss).data()[0], grads)
    };

    let (_, grads) = loss_of(&params);
    let eps = 1e-2f32;
    for (p, grad) in grads.iter().enumerate() {
        let n = params[p].1.numel();
        for i in [0, n / 2, n - 1] {
            let mut plus = params.clone();
            plus[p].1.data_mut()[i] += eps;
            let mut minus = params.clone();
            minus[p].1.data_mut()[i] -= eps;
            let numeric = (loss_of(&plus).0 - loss_of(&minus).0) / (2.0 * eps);
            let analytic = grad.data()[i];
            assert!(
                (numeric - analytic).abs() <= 3e-3,
                "{}[{i}]: analytic {analytic}, numeric {numeric}",
                params[p].0
            );
        }
    }
}
