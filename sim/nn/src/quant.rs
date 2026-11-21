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
        let bias = quantize_bias(b, x.scale * w_scale);
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

/// A 2-D convolution in int8 on NCHW data, weight (C_out, C_in, kH, kW).
/// Zero padding means a real zero, which is stored as the input zero point:
/// padded taps contribute (zp - zp) * w = 0, so they are simply skipped.
#[derive(Clone, Debug)]
pub struct QConv2d {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel: (usize, usize),
    pub stride: usize,
    pub padding: usize,
    /// (C_out, C_in, kH, kW), row-major.
    pub weight: Vec<i8>,
    pub bias: Vec<i32>,
    pub x_zero_point: i32,
    pub y_zero_point: i32,
    pub requant: Requant,
    pub relu: bool,
}

impl QConv2d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        w: &Tensor,
        b: &Tensor,
        x: QParams,
        w_scale: f32,
        y: QParams,
        stride: usize,
        padding: usize,
        relu: bool,
    ) -> Self {
        let [cout, cin, kh, kw] = w.shape()[..] else {
            panic!("QConv2d: weight must be (C_out, C_in, kH, kW)")
        };
        assert_eq!(b.shape(), &[cout], "QConv2d: bias shape");
        assert!(stride > 0, "QConv2d: stride must be positive");
        Self {
            in_channels: cin,
            out_channels: cout,
            kernel: (kh, kw),
            stride,
            padding,
            weight: quantize_i8_symmetric(w, w_scale),
            bias: quantize_bias(b, x.scale * w_scale),
            x_zero_point: x.zero_point,
            y_zero_point: y.zero_point,
            requant: Requant::from_scale(x.scale as f64 * w_scale as f64 / y.scale as f64),
            relu,
        }
    }

    pub fn output_shape(&self, [n, c, h, w]: [usize; 4]) -> [usize; 4] {
        assert_eq!(
            c, self.in_channels,
            "QConv2d: input has {c} channels, expected {}",
            self.in_channels
        );
        let (kh, kw) = self.kernel;
        assert!(
            h + 2 * self.padding >= kh && w + 2 * self.padding >= kw,
            "QConv2d: kernel larger than input"
        );
        let oh = (h + 2 * self.padding - kh) / self.stride + 1;
        let ow = (w + 2 * self.padding - kw) / self.stride + 1;
        [n, self.out_channels, oh, ow]
    }

    /// The int32 accumulator of one output element.
    pub fn accumulate(
        &self,
        x: &[u8],
        [_, cin, h, w]: [usize; 4],
        n: usize,
        co: usize,
        oy: usize,
        ox: usize,
    ) -> i32 {
        let (kh, kw) = self.kernel;
        let mut acc = self.bias[co];
        for ci in 0..cin {
            for ky in 0..kh {
                let Some(iy) = (oy * self.stride + ky)
                    .checked_sub(self.padding)
                    .filter(|&i| i < h)
                else {
                    continue;
                };
                for kx in 0..kw {
                    let Some(ix) = (ox * self.stride + kx)
                        .checked_sub(self.padding)
                        .filter(|&i| i < w)
                    else {
                        continue;
                    };
                    let xv = x[((n * cin + ci) * h + iy) * w + ix] as i32 - self.x_zero_point;
                    acc += xv * self.weight[((co * cin + ci) * kh + ky) * kw + kx] as i32;
                }
            }
        }
        acc
    }

    pub fn forward(&self, x: &[u8], shape: [usize; 4]) -> (Vec<u8>, [usize; 4]) {
        assert_eq!(
            x.len(),
            shape.iter().product::<usize>(),
            "QConv2d: input length"
        );
        let out_shape = self.output_shape(shape);
        let [n, cout, oh, ow] = out_shape;
        let lo = if self.relu {
            self.y_zero_point.max(0)
        } else {
            0
        };
        let mut out = Vec::with_capacity(out_shape.iter().product());
        for ni in 0..n {
            for co in 0..cout {
                for oy in 0..oh {
                    for ox in 0..ow {
                        let acc = self.accumulate(x, shape, ni, co, oy, ox);
                        out.push((self.requant.apply(acc) + self.y_zero_point).clamp(lo, 255) as u8);
                    }
                }
            }
        }
        (out, out_shape)
    }
}

fn quantize_bias(b: &Tensor, scale: f32) -> Vec<i32> {
    b.data()
        .iter()
        .map(|&v| (v * (1.0 / scale)).round_ties_even() as i32)
        .collect()
}

/// Max pooling on uint8 values, no padding. Scale and zero point are unchanged
/// and max is monotonic, so this is exact: no rounding happens.
pub fn max_pool2d_u8(
    x: &[u8],
    [n, c, h, w]: [usize; 4],
    kernel: usize,
    stride: usize,
) -> (Vec<u8>, [usize; 4]) {
    assert_eq!(x.len(), n * c * h * w, "max_pool2d_u8: input length");
    assert!(
        kernel > 0 && stride > 0 && h >= kernel && w >= kernel,
        "max_pool2d_u8: bad geometry"
    );
    let (oh, ow) = ((h - kernel) / stride + 1, (w - kernel) / stride + 1);
    let mut out = Vec::with_capacity(n * c * oh * ow);
    for plane in 0..n * c {
        for oy in 0..oh {
            for ox in 0..ow {
                let mut best = 0u8;
                for ky in 0..kernel {
                    for kx in 0..kernel {
                        best = best.max(x[(plane * h + oy * stride + ky) * w + ox * stride + kx]);
                    }
                }
                out.push(best);
            }
        }
    }
    (out, [n, c, oh, ow])
}

/// One step of a quantized network.
#[derive(Clone, Debug)]
pub enum QLayer {
    Conv2d(QConv2d),
    MaxPool2d {
        kernel: usize,
        stride: usize,
    },
    /// (N, C, H, W) to (N, C * H * W); the bytes do not move.
    Flatten,
    Linear(QLinear),
}

/// A quantized network: what the compiler turns into a RISC-V program.
#[derive(Clone, Debug)]
pub struct QNetwork {
    /// How float inputs are quantized; the program itself starts from uint8.
    pub input: QParams,
    /// Shape of one input sample, without the batch dimension.
    pub input_shape: Vec<usize>,
    pub layers: Vec<QLayer>,
}

impl QNetwork {
    /// The uint8 output of every layer, for `n` samples.
    pub fn forward_layers(&self, x: &[u8], n: usize) -> Vec<Vec<u8>> {
        let mut shape: Vec<usize> = std::iter::once(n)
            .chain(self.input_shape.iter().copied())
            .collect();
        assert_eq!(
            x.len(),
            shape.iter().product::<usize>(),
            "QNetwork: input length"
        );
        let mut cur = x.to_vec();
        let mut outputs = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let as4 = |s: &[usize]| -> [usize; 4] {
                s.try_into()
                    .unwrap_or_else(|_| panic!("QNetwork: layer needs (N, C, H, W), got {s:?}"))
            };
            (cur, shape) = match layer {
                QLayer::Conv2d(conv) => {
                    let (y, s) = conv.forward(&cur, as4(&shape));
                    (y, s.to_vec())
                }
                QLayer::MaxPool2d { kernel, stride } => {
                    let (y, s) = max_pool2d_u8(&cur, as4(&shape), *kernel, *stride);
                    (y, s.to_vec())
                }
                QLayer::Flatten => {
                    let rest = shape[1..].iter().product();
                    (cur, vec![shape[0], rest])
                }
                QLayer::Linear(lin) => {
                    assert_eq!(shape, [n, lin.inputs], "QNetwork: linear layer input shape");
                    (lin.forward(&cur, n), vec![n, lin.outputs])
                }
            };
            outputs.push(cur.clone());
        }
        outputs
    }

    /// The network output for `n` samples.
    pub fn forward(&self, x: &[u8], n: usize) -> Vec<u8> {
        self.forward_layers(x, n)
            .pop()
            .unwrap_or_else(|| x.to_vec())
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
