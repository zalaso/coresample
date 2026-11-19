//! A 2-16-3 MLP trained for 100 full-batch SGD steps, step by step against PyTorch.
mod common;
use common::{assert_close, labels, load, ATOL, RTOL};
use coresample_nn::{sgd_step, Graph, Tensor};

#[test]
fn training_matches_pytorch() {
    let r = load("mlp.ref");
    let x = r["mlp.x"].clone();
    let y = labels(&r["mlp.y"]);
    let lr = r["mlp.lr"].data()[0];
    let mut params: Vec<Tensor> = ["mlp.w1", "mlp.b1", "mlp.w2", "mlp.b2"]
        .iter()
        .map(|n| r[*n].clone())
        .collect();
    let expected_losses = r["mlp.losses"].data();
    let mut max_loss_diff = 0.0f32;

    for (step, &expected) in expected_losses.iter().enumerate() {
        let mut g = Graph::new();
        let xi = g.input(x.clone());
        let ids: Vec<_> = params.iter().map(|p| g.param(p.clone())).collect();
        let h = g.matmul(xi, ids[0]);
        let h = g.add_bias(h, ids[1]);
        let h = g.relu(h);
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
    for (p, name) in params.iter().zip(["w1", "b1", "w2", "b2"]) {
        let expected = &r[&format!("mlp.final.{name}")];
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
