//! Every operation, forward and backward, against PyTorch.
mod common;
use common::{assert_close, labels, load};
use coresample_nn::{Graph, Tensor};

#[test]
fn matmul() {
    let r = load("ops.ref");
    let mut g = Graph::new();
    let a = g.param(r["matmul.a"].clone());
    let b = g.param(r["matmul.b"].clone());
    let y = g.matmul(a, b);
    assert_close("y", g.value(y), &r["matmul.y"]);
    g.backward_with(y, r["matmul.g"].clone());
    assert_close("da", g.grad(a).unwrap(), &r["matmul.da"]);
    assert_close("db", g.grad(b).unwrap(), &r["matmul.db"]);
}

#[test]
fn add_bias() {
    let r = load("ops.ref");
    let mut g = Graph::new();
    let x = g.param(r["add_bias.x"].clone());
    let b = g.param(r["add_bias.b"].clone());
    let y = g.add_bias(x, b);
    assert_close("y", g.value(y), &r["add_bias.y"]);
    g.backward_with(y, r["add_bias.g"].clone());
    assert_close("dx", g.grad(x).unwrap(), &r["add_bias.dx"]);
    assert_close("db", g.grad(b).unwrap(), &r["add_bias.db"]);
}

#[test]
fn relu() {
    let r = load("ops.ref");
    let mut g = Graph::new();
    let x = g.param(r["relu.x"].clone());
    let y = g.relu(x);
    assert_close("y", g.value(y), &r["relu.y"]);
    g.backward_with(y, r["relu.g"].clone());
    assert_close("dx", g.grad(x).unwrap(), &r["relu.dx"]);
}

#[test]
fn cross_entropy() {
    let r = load("ops.ref");
    let mut g = Graph::new();
    let logits = g.param(r["cross_entropy.logits"].clone());
    let loss = g.cross_entropy(logits, &labels(&r["cross_entropy.targets"]));
    assert_eq!(g.value(loss).shape(), &[] as &[usize], "loss is a scalar");
    assert_close(
        "loss",
        &g.value(loss).reshape(&[1]),
        &r["cross_entropy.loss"],
    );
    g.backward(loss);
    assert_close(
        "dlogits",
        g.grad(logits).unwrap(),
        &r["cross_entropy.dlogits"],
    );
}

#[test]
fn inputs_get_no_gradient() {
    let mut g = Graph::new();
    let x = g.input(Tensor::new(vec![1, 2], vec![1.0, -2.0]));
    let w = g.param(Tensor::new(vec![2, 1], vec![3.0, 4.0]));
    let y = g.matmul(x, w);
    g.backward_with(y, Tensor::new(vec![1, 1], vec![1.0]));
    assert!(g.grad(x).is_none());
    assert_eq!(g.grad(w).unwrap().data(), &[1.0, -2.0]);
}

#[test]
fn reused_node_accumulates_gradient() {
    // y = relu(x) @ w + relu(x) @ w uses relu(x) twice: each path adds its share.
    let mut g = Graph::new();
    let x = g.param(Tensor::new(vec![1, 2], vec![1.0, 2.0]));
    let w = g.param(Tensor::new(vec![2, 1], vec![3.0, 4.0]));
    let h = g.relu(x);
    let p = g.matmul(h, w);
    let q = g.matmul(h, w);
    let y = g.add(p, q);
    g.backward_with(y, Tensor::new(vec![1, 1], vec![1.0]));
    assert_eq!(g.grad(x).unwrap().data(), &[6.0, 8.0]);
    assert_eq!(g.grad(w).unwrap().data(), &[2.0, 4.0]);
}

#[test]
#[should_panic(expected = "shape")]
fn matmul_rejects_mismatched_shapes() {
    let mut g = Graph::new();
    let a = g.input(Tensor::zeros(vec![2, 3]));
    let b = g.input(Tensor::zeros(vec![2, 3]));
    g.matmul(a, b);
}

/// Gradients of a small network checked by central finite differences, with no
/// PyTorch involved: an independent second check of every backward rule.
#[test]
fn gradients_match_finite_differences() {
    let x = Tensor::new(vec![3, 2], vec![0.5, -1.0, 1.5, 0.25, -0.75, 1.0]);
    let w1 = Tensor::new(vec![2, 4], vec![0.3, -0.2, 0.8, 0.1, -0.5, 0.7, 0.2, -0.4]);
    let b1 = Tensor::new(vec![4], vec![0.05, -0.1, 0.2, 0.15]);
    let w2 = Tensor::new(
        vec![4, 3],
        vec![
            0.2, -0.3, 0.5, 0.6, 0.1, -0.2, -0.4, 0.3, 0.7, 0.25, -0.6, 0.1,
        ],
    );
    let targets = [2usize, 0, 1];

    let loss_of = |params: &[Tensor]| -> (f32, Vec<Tensor>) {
        let mut g = Graph::new();
        let xi = g.input(x.clone());
        let ids: Vec<_> = params.iter().map(|p| g.param(p.clone())).collect();
        let h = g.matmul(xi, ids[0]);
        let h = g.add_bias(h, ids[1]);
        let h = g.relu(h);
        let logits = g.matmul(h, ids[2]);
        let loss = g.cross_entropy(logits, &targets);
        g.backward(loss);
        let grads = ids.iter().map(|&id| g.grad(id).unwrap().clone()).collect();
        (g.value(loss).data()[0], grads)
    };

    let params = vec![w1, b1, w2];
    let (_, grads) = loss_of(&params);
    let eps = 1e-2f32;
    for (p, grad) in grads.iter().enumerate() {
        for i in 0..params[p].data().len() {
            let mut plus = params.clone();
            plus[p].data_mut()[i] += eps;
            let mut minus = params.clone();
            minus[p].data_mut()[i] -= eps;
            let numeric = (loss_of(&plus).0 - loss_of(&minus).0) / (2.0 * eps);
            let analytic = grad.data()[i];
            assert!(
                (numeric - analytic).abs() <= 2e-3,
                "param {p}[{i}]: analytic {analytic}, numeric {numeric}"
            );
        }
    }
}
