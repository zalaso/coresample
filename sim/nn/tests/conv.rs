//! Convolution, max pooling and flatten, forward and backward, against PyTorch.
mod common;
use common::{assert_close, load};
use coresample_nn::{Graph, Tensor};

fn check_conv(case: &str) {
    let r = load("conv.ref");
    let p = r[&format!("{case}.params")].data();
    let (stride, padding) = (p[0] as usize, p[1] as usize);
    let mut g = Graph::new();
    let x = g.param(r[&format!("{case}.x")].clone());
    let w = g.param(r[&format!("{case}.w")].clone());
    let b = g.param(r[&format!("{case}.b")].clone());
    let y = g.conv2d(x, w, b, stride, padding);
    assert_close("y", g.value(y), &r[&format!("{case}.y")]);
    g.backward_with(y, r[&format!("{case}.g")].clone());
    assert_close("dx", g.grad(x).unwrap(), &r[&format!("{case}.dx")]);
    assert_close("dw", g.grad(w).unwrap(), &r[&format!("{case}.dw")]);
    assert_close("db", g.grad(b).unwrap(), &r[&format!("{case}.db")]);
}

#[test]
fn conv2d_stride1_no_padding_rectangular_kernel() {
    check_conv("conv_s1p0");
}

#[test]
fn conv2d_stride2_padding1() {
    check_conv("conv_s2p1");
}

#[test]
fn max_pool2d_including_ties() {
    // The reference has a window of equal zeros and one with three equal maxima:
    // the gradient must go to the first maximum in row-major order, as in PyTorch.
    let r = load("conv.ref");
    let mut g = Graph::new();
    let x = g.param(r["maxpool.x"].clone());
    let y = g.max_pool2d(x, 2, 2);
    assert_close("y", g.value(y), &r["maxpool.y"]);
    g.backward_with(y, r["maxpool.g"].clone());
    assert_close("dx", g.grad(x).unwrap(), &r["maxpool.dx"]);
}

#[test]
fn flatten() {
    let r = load("conv.ref");
    let mut g = Graph::new();
    let x = g.param(r["flatten.x"].clone());
    let y = g.flatten(x);
    assert_close("y", g.value(y), &r["flatten.y"]);
    g.backward_with(y, r["flatten.g"].clone());
    assert_close("dx", g.grad(x).unwrap(), &r["flatten.dx"]);
}

#[test]
fn conv2d_output_shape() {
    // H_out = (H + 2p - kH) / s + 1
    let mut g = Graph::new();
    let x = g.input(Tensor::zeros(vec![1, 2, 9, 8]));
    let w = g.param(Tensor::zeros(vec![5, 2, 3, 2]));
    let b = g.param(Tensor::zeros(vec![5]));
    let y = g.conv2d(x, w, b, 2, 1);
    assert_eq!(g.value(y).shape(), &[1, 5, 5, 5]);
}

#[test]
#[should_panic(expected = "channels")]
fn conv2d_rejects_channel_mismatch() {
    let mut g = Graph::new();
    let x = g.input(Tensor::zeros(vec![1, 3, 5, 5]));
    let w = g.param(Tensor::zeros(vec![4, 2, 3, 3]));
    let b = g.param(Tensor::zeros(vec![4]));
    g.conv2d(x, w, b, 1, 0);
}

/// conv -> relu -> max pool -> flatten -> linear -> cross entropy, checked by
/// central finite differences (no ties: the inputs are distinct).
#[test]
fn cnn_gradients_match_finite_differences() {
    let x = Tensor::new(
        vec![2, 1, 4, 4],
        (0..32).map(|i| ((i * 7 % 13) as f32 - 6.0) / 5.0).collect(),
    );
    let conv_w = Tensor::new(
        vec![2, 1, 3, 3],
        (0..18)
            .map(|i| ((i * 5 % 11) as f32 - 5.0) / 10.0)
            .collect(),
    );
    let conv_b = Tensor::new(vec![2], vec![0.1, -0.05]);
    let lin_w = Tensor::new(
        vec![8, 3],
        (0..24).map(|i| ((i * 3 % 7) as f32 - 3.0) / 8.0).collect(),
    );
    let targets = [1usize, 2];

    let loss_of = |params: &[Tensor]| -> (f32, Vec<Tensor>) {
        let mut g = Graph::new();
        let xi = g.input(x.clone());
        let ids: Vec<_> = params.iter().map(|p| g.param(p.clone())).collect();
        let h = g.conv2d(xi, ids[0], ids[1], 1, 1);
        let h = g.relu(h);
        let h = g.max_pool2d(h, 2, 2);
        let h = g.flatten(h);
        let logits = g.matmul(h, ids[2]);
        let loss = g.cross_entropy(logits, &targets);
        g.backward(loss);
        let grads = ids.iter().map(|&id| g.grad(id).unwrap().clone()).collect();
        (g.value(loss).data()[0], grads)
    };

    let params = vec![conv_w, conv_b, lin_w];
    let (_, grads) = loss_of(&params);
    let eps = 1e-3f32;
    for (p, grad) in grads.iter().enumerate() {
        for i in 0..params[p].numel() {
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
