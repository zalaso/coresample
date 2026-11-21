//! The bars CNN in int8, layer by layer, against PyTorch's quantized kernels:
//! quant -> conv 3x3 pad 1 + ReLU -> max pool 2 -> flatten -> linear.
//! This is the network the compiler puts on the core for G2.
mod common;
use common::{ints, labels, load, within};
use coresample_nn::quant::{
    max_pool2d_u8, quantize_i8_symmetric, quantize_u8, QConv2d, QLayer, QLinear, QNetwork, QParams,
};
use coresample_nn::refdata::RefFile;
use coresample_nn::Tensor;

fn qparams(r: &RefFile, name: &str) -> QParams {
    let t = r[&format!("qcnn.{name}_qparams")].data();
    QParams {
        scale: t[0],
        zero_point: t[1] as i32,
    }
}

fn network(r: &RefFile) -> QNetwork {
    let conv = QConv2d::new(
        &r["qcnn.conv_w"],
        &r["qcnn.conv_b"],
        qparams(r, "x"),
        qparams(r, "conv_w").scale,
        qparams(r, "conv"),
        1,
        1,
        true,
    );
    let lin = QLinear::new(
        &r["qcnn.lin_w"],
        &r["qcnn.lin_b"],
        qparams(r, "pool"),
        qparams(r, "lin_w").scale,
        qparams(r, "out"),
        false,
    );
    QNetwork {
        input: qparams(r, "x"),
        input_shape: vec![1, 8, 8],
        layers: vec![
            QLayer::Conv2d(conv),
            QLayer::MaxPool2d {
                kernel: 2,
                stride: 2,
            },
            QLayer::Flatten,
            QLayer::Linear(lin),
        ],
    }
}

fn as_u8(v: &[i32]) -> Vec<u8> {
    v.iter().map(|&x| x as u8).collect()
}

#[test]
fn conv_weights_quantize_exactly_like_pytorch() {
    let r = load("quant_cnn.ref");
    let got: Vec<i32> = quantize_i8_symmetric(&r["qcnn.conv_w"], qparams(&r, "conv_w").scale)
        .iter()
        .map(|&v| v as i32)
        .collect();
    assert_eq!(got, ints(&r["qcnn.conv_w_int"]));
}

#[test]
fn conv_layer_within_one_of_pytorch() {
    let r = load("quant_cnn.ref");
    let QLayer::Conv2d(conv) = &network(&r).layers[0] else {
        unreachable!()
    };
    let n = r["qcnn.x"].shape()[0];
    let (y, shape) = conv.forward(&as_u8(&ints(&r["qcnn.x_int"])), [n, 1, 8, 8]);
    assert_eq!(shape, [n, 4, 8, 8]);
    let expected = ints(&r["qcnn.conv_int"]);
    let exact = within("conv", &y, &expected, 1);
    assert!(
        exact * 100 >= 99 * expected.len(),
        "fewer than 99% identical"
    );
}

#[test]
fn max_pool_is_exact_on_uint8() {
    // Pooling keeps scale and zero point, and max is monotonic: no rounding at all.
    let r = load("quant_cnn.ref");
    let n = r["qcnn.x"].shape()[0];
    let (y, shape) = max_pool2d_u8(&as_u8(&ints(&r["qcnn.conv_int"])), [n, 4, 8, 8], 2, 2);
    assert_eq!(shape, [n, 4, 4, 4]);
    assert_eq!(y, as_u8(&ints(&r["qcnn.pool_int"])));
}

#[test]
fn linear_layer_within_one_of_pytorch() {
    let r = load("quant_cnn.ref");
    let QLayer::Linear(lin) = &network(&r).layers[3] else {
        unreachable!()
    };
    let n = r["qcnn.x"].shape()[0];
    let expected = ints(&r["qcnn.out_int"]);
    let exact = within(
        "linear",
        &lin.forward(&as_u8(&ints(&r["qcnn.pool_int"])), n),
        &expected,
        1,
    );
    assert!(
        exact * 100 >= 99 * expected.len(),
        "fewer than 99% identical"
    );
}

#[test]
fn network_end_to_end() {
    let r = load("quant_cnn.ref");
    let net = network(&r);
    let n = r["qcnn.x"].shape()[0];
    let x = quantize_u8(&r["qcnn.x"], net.input);
    assert_eq!(x, as_u8(&ints(&r["qcnn.x_int"])), "input quantization");
    let outputs = net.forward_layers(&x, n);
    assert_eq!(outputs.len(), 4, "one output per layer");
    within(
        "conv (end to end)",
        &outputs[0],
        &ints(&r["qcnn.conv_int"]),
        1,
    );
    within(
        "pool (end to end)",
        &outputs[1],
        &ints(&r["qcnn.pool_int"]),
        1,
    );
    let out = net.forward(&x, n);
    assert_eq!(&out, outputs.last().unwrap());
    within("output (end to end)", &out, &ints(&r["qcnn.out_int"]), 2);

    let y = labels(&r["qcnn.y"]);
    let correct = out
        .chunks_exact(3)
        .zip(&y)
        .filter(|(row, &t)| argmax(row) == t)
        .count();
    let pytorch = (r["qcnn.accuracy"].data()[1] * n as f32).round() as usize;
    eprintln!("accuracy: Coresample int8 {correct}/{n}, PyTorch int8 {pytorch}/{n}");
    assert!(correct.abs_diff(pytorch) <= 1);
}

fn argmax(row: &[u8]) -> usize {
    (0..row.len()).fold(0, |best, i| if row[i] > row[best] { i } else { best })
}

#[test]
fn conv_padding_is_the_zero_point_not_zero() {
    // An image of real zeros is stored as the zero point (here 100). Padding
    // must read as real zero too, so every output position, border included,
    // sees the same all-zero window and equals the requantized bias.
    let x = QParams {
        scale: 0.05,
        zero_point: 100,
    };
    let y = QParams {
        scale: 0.1,
        zero_point: 20,
    };
    let w = Tensor::new(vec![1, 1, 3, 3], vec![0.5; 9]);
    let b = Tensor::new(vec![1], vec![0.7]);
    let conv = QConv2d::new(&w, &b, x, 0.5 / 127.0, y, 1, 1, false);
    let (out, _) = conv.forward(&[100u8; 16], [1, 1, 4, 4]);
    assert!(
        out.iter().all(|&v| v == out[0]),
        "border differs from center: {out:?}"
    );
    assert_eq!(out[0], 27); // 0.7 / 0.1 = 7, plus the zero point 20
}
