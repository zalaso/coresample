//! Integer-only int8 inference, the arithmetic the RISC-V core will run.
//!
//! Scheme (as TFLite / gemmlowp): a real value r is stored as an integer q with
//! r = scale * (q - zero_point).
//! * activations: uint8, asymmetric (zero point in 0..=255);
//! * weights: int8, symmetric per tensor (zero point 0);
//! * bias: int32 with scale = input scale * weight scale, so it adds straight
//!   into the int32 accumulator;
//! * requantization: acc * M with M = input scale * weight scale / output scale,
//!   done as a fixed-point multiplier M = m0 / 2^shift with one rounding.
//!
//! Everything after quantizing the input is integer arithmetic, so a compiled
//! program on the core can reproduce it bit for bit.

use crate::tensor::Tensor;

/// Scale and zero point of one quantized tensor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QParams {
    pub scale: f32,
    pub zero_point: i32,
}

/// PyTorch's quantize: zero_point + round_half_even(x * (1 / scale)), clamped.
fn quantize(x: f32, scale: f32, zero_point: i32, lo: i32, hi: i32) -> i32 {
    let inv = 1.0f32 / scale;
    (zero_point + (x * inv).round_ties_even() as i32).clamp(lo, hi)
}

/// Activations to uint8, as `torch.quantize_per_tensor(x, scale, zp, torch.quint8)`.
pub fn quantize_u8(x: &Tensor, q: QParams) -> Vec<u8> {
    x.data()
        .iter()
        .map(|&v| quantize(v, q.scale, q.zero_point, 0, 255) as u8)
        .collect()
}

/// Weights to int8 with zero point 0.
pub fn quantize_i8_symmetric(w: &Tensor, scale: f32) -> Vec<i8> {
    w.data()
        .iter()
        .map(|&v| quantize(v, scale, 0, -128, 127) as i8)
        .collect()
}

/// A real multiplier in (0, 1) as m0 / 2^shift, with m0 in [2^30, 2^31) so it
/// keeps 31 significant bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Requant {
    pub m0: i32,
    pub shift: u32,
}

impl Requant {
    pub fn from_scale(m: f64) -> Self {
        assert!(
            m > 0.0 && m < 1.0,
            "requantization multiplier {m} must be in (0, 1)"
        );
        let mut shift = 31u32;
        while m * 2f64.powi(shift as i32) < (1u64 << 30) as f64 {
            shift += 1;
            assert!(shift <= 62, "requantization multiplier {m} too small");
        }
        let mut m0 = (m * 2f64.powi(shift as i32)).round() as i64;
        if m0 == 1i64 << 31 {
            // Rounding reached the next power of two.
            m0 >>= 1;
            shift -= 1;
        }
        Self {
            m0: m0 as i32,
            shift,
        }
    }

    /// round_half_up(acc * m0 / 2^shift). On RV32IM: `mul` and `mulh` give the
    /// 64-bit product; add the rounding constant, shift right.
    pub fn apply(&self, acc: i32) -> i32 {
        let product = acc as i64 * self.m0 as i64;
        ((product + (1i64 << (self.shift - 1))) >> self.shift) as i32
    }
}

/// A fully connected layer in int8: y = requant((x_q - x_zp) @ w_q + b_q) + y_zp,
/// optionally with ReLU fused in as the lower clamp.
#[derive(Clone, Debug)]
pub struct QLinear {
    pub inputs: usize,
    pub outputs: usize,
    /// (inputs, outputs), row-major.
    pub weight: Vec<i8>,
    pub bias: Vec<i32>,
    pub x_zero_point: i32,
    pub y_zero_point: i32,
    pub requant: Requant,
    pub relu: bool,
}

impl QLinear {
    /// Quantizes a float layer (`w` of shape (in, out), `b` of shape (out)).
    pub fn new(w: &Tensor, b: &Tensor, x: QParams, w_scale: f32, y: QParams, relu: bool) -> Self {
        let (inputs, outputs) = w.dims2();
        assert_eq!(b.shape(), &[outputs], "QLinear: bias shape");
        let bias_scale = x.scale * w_scale;
        let bias = b
            .data()
            .iter()
            .map(|&v| (v * (1.0 / bias_scale)).round_ties_even() as i32)
            .collect();
        Self {
            inputs,
            outputs,
            weight: quantize_i8_symmetric(w, w_scale),
            bias,
            x_zero_point: x.zero_point,
            y_zero_point: y.zero_point,
            requant: Requant::from_scale(x.scale as f64 * w_scale as f64 / y.scale as f64),
            relu,
        }
    }

    /// The int32 accumulator of output `o` for one input row.
    pub fn accumulate(&self, row: &[u8], o: usize) -> i32 {
        let mut acc = self.bias[o];
        for (i, &xv) in row.iter().enumerate() {
            acc += (xv as i32 - self.x_zero_point) * self.weight[i * self.outputs + o] as i32;
        }
        acc
    }

    /// `n` rows of `inputs` values to `n` rows of `outputs` values.
    pub fn forward(&self, x: &[u8], n: usize) -> Vec<u8> {
        assert_eq!(x.len(), n * self.inputs, "QLinear: input length");
        let lo = if self.relu {
            self.y_zero_point.max(0)
        } else {
            0
        };
        let mut out = Vec::with_capacity(n * self.outputs);
        for row in x.chunks_exact(self.inputs) {
            for o in 0..self.outputs {
                let y = self.requant.apply(self.accumulate(row, o)) + self.y_zero_point;
                out.push(y.clamp(lo, 255) as u8);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_rounds_half_to_even_like_pytorch() {
        let x = Tensor::new(vec![5], vec![0.5, 1.5, 2.5, -0.5, -1.5]);
        let got: Vec<i8> = quantize_i8_symmetric(&x, 1.0);
        assert_eq!(got, vec![0, 2, 2, 0, -2]);
    }

    #[test]
    fn quantize_saturates() {
        let x = Tensor::new(vec![2], vec![-1000.0, 1000.0]);
        assert_eq!(
            quantize_u8(
                &x,
                QParams {
                    scale: 1.0,
                    zero_point: 10
                }
            ),
            vec![0, 255]
        );
    }

    #[test]
    fn requant_of_one_half() {
        let rq = Requant::from_scale(0.5);
        assert_eq!((rq.m0, rq.shift), (1 << 30, 31));
        assert_eq!(rq.apply(7), 4); // 3.5 rounds up
        assert_eq!(rq.apply(-7), -3); // -3.5 rounds up too: half-up, not half-away
    }

    #[test]
    fn relu_clamps_at_the_zero_point() {
        // One input, one output, weight 1, input scale 1, output scale 2: y = round(x / 2).
        let w = Tensor::new(vec![1, 1], vec![1.0]);
        let b = Tensor::new(vec![1], vec![0.0]);
        let q = |scale| QParams {
            scale,
            zero_point: 0,
        };
        let mut layer = QLinear::new(
            &w,
            &b,
            QParams {
                scale: 1.0,
                zero_point: 100,
            },
            1.0,
            q(2.0),
            true,
        );
        // x = 40 means real value 40 - 100 = -60: ReLU gives 0.
        assert_eq!(layer.forward(&[40, 120], 2), vec![0, 10]);
        layer.relu = false;
        layer.y_zero_point = 50;
        assert_eq!(layer.forward(&[40], 1), vec![20]);
    }
}
