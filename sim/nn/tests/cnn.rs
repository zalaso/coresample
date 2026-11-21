//! A small CNN trained for 60 full-batch SGD steps, step by step against PyTorch:
//! conv(1->4, 3x3, pad 1) -> ReLU -> max pool 2 -> flatten -> linear(64->3).
mod common;
use common::{assert_close, labels, load, ATOL, RTOL};
use coresample_nn::{sgd_step, Graph, Tensor};

#[test]
fn training_matches_pytorch() {
    let r = load("cnn.ref");
    let x = r["cnn.x"].clone();
    let y = labels(&r["cnn.y"]);
    let lr = r["cnn.lr"].data()[0];
    let names = ["conv_w", "conv_b", "lin_w", "lin_b"];
    let mut params: Vec<Tensor> = names
        .iter()
        .map(|n| r[&format!("cnn.{n}")].clone())
        .collect();
    let expected_losses = r["cnn.losses"].data();
    let mut max_loss_diff = 0.0f32;

    for (step, &expected) in expected_losses.iter().enumerate() {
        let mut g = Graph::new();
        let xi = g.input(x.clone());
        let ids: Vec<_> = params.iter().map(|p| g.param(p.clone())).collect();
        let h = g.conv2d(xi, ids[0], ids[1], 1, 1);
        let h = g.relu(h);
        let h = g.max_pool2d(h, 2, 2);
        let h = g.flatten(h);
        let logits = g.matmul(h, ids[2]);
        let logits = g.add_bias(logits, ids[3]);
        let loss = g.cross_entropy(logits, &y);
        let got = g.value(loss).data()[0];
        max_loss_diff = max_loss_diff.max((got - expected).abs());
        assert!(
            (got - expected).abs() <= ATOL + RTOL * expected.abs(),
            "step {step}: loss {got}, PyTorch {expected}"
        );
        g.backward(loss);
        for (p, &id) in params.iter_mut().zip(&ids) {
            sgd_step(p, g.grad(id).unwrap(), lr);
        }
    }

    let mut max_weight_diff = 0.0f32;
    for (p, name) in params.iter().zip(names) {
        let expected = &r[&format!("cnn.final.{name}")];
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
