//! Integer-only int8 inference against PyTorch's quantized kernels (qnnpack).
//!
//! Quantizing inputs and weights must match PyTorch exactly. Layer outputs may
//! differ by 1 in the last integer place: PyTorch requantizes in floating
//! point, Coresample with a fixed-point multiplier that the RISC-V core can run.
mod common;
use common::{labels, load, within};
use coresample_nn::quant::{quantize_i8_symmetric, quantize_u8, QLinear, QParams, Requant};
use coresample_nn::refdata::RefFile;

fn qparams(r: &RefFile, name: &str) -> QParams {
    let t = r[&format!("qmlp.{name}_qparams")].data();
    QParams {
        scale: t[0],
        zero_point: t[1] as i32,
    }
}

fn ints(r: &RefFile, name: &str) -> Vec<i32> {
    r[&format!("qmlp.{name}_int")]
        .data()
        .iter()
        .map(|&v| v as i32)
        .collect()
}

fn layers(r: &RefFile) -> (QLinear, QLinear) {
    let (x, h, out) = (qparams(r, "x"), qparams(r, "h"), qparams(r, "out"));
    let l1 = QLinear::new(
        &r["qmlp.w1"],
        &r["qmlp.b1"],
        x,
        qparams(r, "w1").scale,
        h,
        true,
    );
    let l2 = QLinear::new(
        &r["qmlp.w2"],
        &r["qmlp.b2"],
        h,
        qparams(r, "w2").scale,
        out,
        false,
    );
    (l1, l2)
}

#[test]
fn input_quantization_matches_pytorch_exactly() {
    let r = load("quant_mlp.ref");
    let got: Vec<i32> = quantize_u8(&r["qmlp.x"], qparams(&r, "x"))
        .iter()
        .map(|&v| v as i32)
        .collect();
    assert_eq!(got, ints(&r, "x"));
}

#[test]
fn weight_quantization_matches_pytorch_exactly() {
    let r = load("quant_mlp.ref");
    for w in ["w1", "w2"] {
        let got: Vec<i32> = quantize_i8_symmetric(&r[&format!("qmlp.{w}")], qparams(&r, w).scale)
            .iter()
            .map(|&v| v as i32)
            .collect();
        assert_eq!(got, ints(&r, w), "{w}");
    }
}

#[test]
fn hidden_layer_within_one_of_pytorch() {
    let r = load("quant_mlp.ref");
    let (l1, _) = layers(&r);
    let x: Vec<u8> = ints(&r, "x").iter().map(|&v| v as u8).collect();
    let n = r["qmlp.x"].shape()[0];
    let exact = within("hidden", &l1.forward(&x, n), &ints(&r, "h"), 1);
    assert!(
        exact * 100 >= 99 * n * 16,
        "fewer than 99% identical: the rounding is off"
    );
}

#[test]
fn output_layer_within_one_of_pytorch() {
    // Fed with PyTorch's own hidden values, so this layer is checked alone.
    let r = load("quant_mlp.ref");
    let (_, l2) = layers(&r);
    let h: Vec<u8> = ints(&r, "h").iter().map(|&v| v as u8).collect();
    let n = r["qmlp.x"].shape()[0];
    let exact = within("output", &l2.forward(&h, n), &ints(&r, "out"), 1);
    assert!(
        exact * 100 >= 99 * n * 3,
        "fewer than 99% identical: the rounding is off"
    );
}

#[test]
fn end_to_end_from_float_input() {
    let r = load("quant_mlp.ref");
    let (l1, l2) = layers(&r);
    let n = r["qmlp.x"].shape()[0];
    let x = quantize_u8(&r["qmlp.x"], qparams(&r, "x"));
    let out = l2.forward(&l1.forward(&x, n), n);
    within("end to end", &out, &ints(&r, "out"), 2);

    // Same classification accuracy as PyTorch's quantized model, within one sample.
    let y = labels(&r["qmlp.y"]);
    let correct = out
        .chunks_exact(3)
        .zip(&y)
        .filter(|(row, &t)| argmax(row) == t)
        .count();
    let pytorch = (r["qmlp.accuracy"].data()[1] * n as f32).round() as usize;
    eprintln!("accuracy: Coresample int8 {correct}/{n}, PyTorch int8 {pytorch}/{n}");
    assert!(correct.abs_diff(pytorch) <= 1);
}

fn argmax(row: &[u8]) -> usize {
    (0..row.len()).fold(0, |best, i| if row[i] > row[best] { i } else { best })
}

/// The fixed-point multiplier rounds exactly once: the result is
/// round_half_up(acc * m0 / 2^shift), checked with 128-bit arithmetic.
#[test]
fn requantization_rounds_once_and_exactly() {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for _ in 0..2000 {
        let m = (next() % 1_000_000 + 1) as f64 / 1_000_001.0; // in (0, 1)
        let rq = Requant::from_scale(m);
        assert!(
            (rq.m0 as f64 / 2f64.powi(rq.shift as i32) - m).abs() <= m * 1e-9,
            "m0 approximates m"
        );
        let acc = next() as i32;
        let exact = (acc as i128 * rq.m0 as i128 + (1i128 << (rq.shift - 1))) >> rq.shift;
        assert_eq!(rq.apply(acc) as i128, exact, "acc {acc}, m {m}");
    }
}
