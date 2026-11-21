//! The operations of a transformer block, forward and backward, against PyTorch.
mod common;
use common::{assert_close, labels, load};
use coresample_nn::{Graph, NodeId, Tensor};

/// Runs a one-input op, then checks its output and input gradient.
fn check_unary(case: &str, input: &str, build: impl Fn(&mut Graph, NodeId) -> NodeId) {
    let r = load("attention.ref");
    let mut g = Graph::new();
    let x = g.param(r[&format!("{case}.{input}")].clone());
    let y = build(&mut g, x);
    assert_close("y", g.value(y), &r[&format!("{case}.y")]);
    g.backward_with(y, r[&format!("{case}.g")].clone());
    assert_close("dx", g.grad(x).unwrap(), &r[&format!("{case}.d{input}")]);
}

#[test]
fn embedding_with_a_repeated_id() {
    let r = load("attention.ref");
    let ids = labels(&r["embedding.ids"]);
    check_unary("embedding", "table", |g, t| g.embedding(t, &ids));
}

#[test]
fn layer_norm() {
    let r = load("attention.ref");
    let mut g = Graph::new();
    let x = g.param(r["layer_norm.x"].clone());
    let gamma = g.param(r["layer_norm.gamma"].clone());
    let beta = g.param(r["layer_norm.beta"].clone());
    let y = g.layer_norm(x, gamma, beta, 1e-5);
    assert_close("y", g.value(y), &r["layer_norm.y"]);
    g.backward_with(y, r["layer_norm.g"].clone());
    assert_close("dx", g.grad(x).unwrap(), &r["layer_norm.dx"]);
    assert_close("dgamma", g.grad(gamma).unwrap(), &r["layer_norm.dgamma"]);
    assert_close("dbeta", g.grad(beta).unwrap(), &r["layer_norm.dbeta"]);
}

#[test]
fn gelu_tanh_approximation() {
    check_unary("gelu", "x", |g, x| g.gelu(x));
}

#[test]
fn softmax_with_large_logits() {
    check_unary("softmax", "x", |g, x| g.softmax(x));
}

#[test]
fn causal_mask_then_softmax() {
    check_unary("causal_softmax", "x", |g, x| {
        let m = g.causal_mask(x);
        g.softmax(m)
    });
}

#[test]
fn causal_mask_hides_the_future() {
    let mut g = Graph::new();
    let x = g.input(Tensor::new(
        vec![1, 3, 3],
        (1..=9).map(|v| v as f32).collect(),
    ));
    let m = g.causal_mask(x);
    let inf = f32::NEG_INFINITY;
    assert_eq!(
        g.value(m).data(),
        &[1.0, inf, inf, 4.0, 5.0, inf, 7.0, 8.0, 9.0]
    );
}

#[test]
fn batched_matmul() {
    let r = load("attention.ref");
    let mut g = Graph::new();
    let a = g.param(r["bmm.a"].clone());
    let b = g.param(r["bmm.b"].clone());
    let y = g.batched_matmul(a, b);
    assert_close("y", g.value(y), &r["bmm.y"]);
    g.backward_with(y, r["bmm.g"].clone());
    assert_close("da", g.grad(a).unwrap(), &r["bmm.da"]);
    assert_close("db", g.grad(b).unwrap(), &r["bmm.db"]);
}

#[test]
fn permute() {
    let r = load("attention.ref");
    let axes = labels(&r["permute.axes"]);
    check_unary("permute", "x", |g, x| g.permute(x, &axes));
}

#[test]
fn scale() {
    let r = load("attention.ref");
    let s = r["scale.s"].data()[0];
    check_unary("scale", "x", |g, x| g.scale(x, s));
}

#[test]
#[should_panic(expected = "permutation")]
fn permute_rejects_a_repeated_axis() {
    let mut g = Graph::new();
    let x = g.input(Tensor::zeros(vec![2, 3, 4]));
    g.permute(x, &[0, 0, 2]);
}
