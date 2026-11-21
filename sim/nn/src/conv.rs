//! 2-D convolution and max pooling on NCHW tensors, forward and backward.
//!
//! Direct loops, no im2col: slower, but each output element is computed in
//! one obvious place, which is what the later compiler stages need to follow.

use crate::tensor::Tensor;

/// Sizes of one convolution: input (n, cin, h, w), weight (cout, cin, kh, kw),
/// output (n, cout, oh, ow).
#[derive(Clone, Copy, Debug)]
struct ConvGeom {
    n: usize,
    cin: usize,
    h: usize,
    w: usize,
    cout: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
    oh: usize,
    ow: usize,
}

fn dims4(t: &Tensor, what: &str) -> (usize, usize, usize, usize) {
    match t.shape()[..] {
        [a, b, c, d] => (a, b, c, d),
        _ => panic!("{what}: expected 4 dimensions, got shape {:?}", t.shape()),
    }
}

impl ConvGeom {
    fn new(x: &Tensor, w: &Tensor, b: &Tensor, stride: usize, pad: usize) -> Self {
        let (n, cin, h, wd) = dims4(x, "conv2d input");
        let (cout, wcin, kh, kw) = dims4(w, "conv2d weight");
        assert_eq!(
            cin, wcin,
            "conv2d: input has {cin} channels, weight expects {wcin}"
        );
        assert_eq!(b.shape(), &[cout], "conv2d: bias shape must be [{cout}]");
        assert!(stride > 0, "conv2d: stride must be positive");
        assert!(
            h + 2 * pad >= kh && wd + 2 * pad >= kw,
            "conv2d: kernel larger than padded input"
        );
        let oh = (h + 2 * pad - kh) / stride + 1;
        let ow = (wd + 2 * pad - kw) / stride + 1;
        Self {
            n,
            cin,
            h,
            w: wd,
            cout,
            kh,
            kw,
            stride,
            pad,
            oh,
            ow,
        }
    }

    /// Input coordinate for output position `o` and kernel offset `k`, or
    /// `None` when it falls in the zero padding.
    fn input_pos(&self, o: usize, k: usize, size: usize) -> Option<usize> {
        (o * self.stride + k)
            .checked_sub(self.pad)
            .filter(|&i| i < size)
    }

    fn x_index(&self, n: usize, c: usize, y: usize, x: usize) -> usize {
        ((n * self.cin + c) * self.h + y) * self.w + x
    }

    fn w_index(&self, co: usize, ci: usize, ky: usize, kx: usize) -> usize {
        ((co * self.cin + ci) * self.kh + ky) * self.kw + kx
    }

    fn y_index(&self, n: usize, co: usize, oy: usize, ox: usize) -> usize {
        ((n * self.cout + co) * self.oh + oy) * self.ow + ox
    }

    /// Calls `f(input index, weight index)` for every tap of output element
    /// (n, co, oy, ox) that is not in the padding.
    fn for_each_tap(
        &self,
        n: usize,
        co: usize,
        oy: usize,
        ox: usize,
        mut f: impl FnMut(usize, usize),
    ) {
        for ci in 0..self.cin {
            for ky in 0..self.kh {
                let Some(iy) = self.input_pos(oy, ky, self.h) else {
                    continue;
                };
                for kx in 0..self.kw {
                    let Some(ix) = self.input_pos(ox, kx, self.w) else {
                        continue;
                    };
                    f(self.x_index(n, ci, iy, ix), self.w_index(co, ci, ky, kx));
                }
            }
        }
    }

    fn outputs(&self) -> impl Iterator<Item = (usize, usize, usize, usize)> + '_ {
        (0..self.n).flat_map(move |n| {
            (0..self.cout).flat_map(move |co| {
                (0..self.oh).flat_map(move |oy| (0..self.ow).map(move |ox| (n, co, oy, ox)))
            })
        })
    }
}

/// Cross-correlation plus bias, as `torch.nn.functional.conv2d`.
pub(crate) fn conv2d(x: &Tensor, w: &Tensor, b: &Tensor, stride: usize, pad: usize) -> Tensor {
    let geo = ConvGeom::new(x, w, b, stride, pad);
    let mut y = vec![0.0f32; geo.n * geo.cout * geo.oh * geo.ow];
    for (n, co, oy, ox) in geo.outputs() {
        let mut acc = b.data()[co];
        geo.for_each_tap(n, co, oy, ox, |xi, wi| acc += x.data()[xi] * w.data()[wi]);
        y[geo.y_index(n, co, oy, ox)] = acc;
    }
    Tensor::new(vec![geo.n, geo.cout, geo.oh, geo.ow], y)
}

/// Gradients (dx, dw, db) of a convolution, given the output gradient `g`.
pub(crate) fn conv2d_backward(
    x: &Tensor,
    w: &Tensor,
    b: &Tensor,
    g: &Tensor,
    stride: usize,
    pad: usize,
) -> (Tensor, Tensor, Tensor) {
    let geo = ConvGeom::new(x, w, b, stride, pad);
    assert_eq!(
        g.shape(),
        &[geo.n, geo.cout, geo.oh, geo.ow],
        "conv2d backward: gradient shape"
    );
    let mut dx = vec![0.0f32; x.numel()];
    let mut dw = vec![0.0f32; w.numel()];
    let mut db = vec![0.0f32; geo.cout];
    for (n, co, oy, ox) in geo.outputs() {
        let gv = g.data()[geo.y_index(n, co, oy, ox)];
        db[co] += gv;
        geo.for_each_tap(n, co, oy, ox, |xi, wi| {
            dw[wi] += gv * x.data()[xi];
            dx[xi] += gv * w.data()[wi];
        });
    }
    (
        Tensor::new(x.shape().to_vec(), dx),
        Tensor::new(w.shape().to_vec(), dw),
        Tensor::new(vec![geo.cout], db),
    )
}

/// Max pooling without padding. Returns the output and, for each output
/// element, the input index it came from: the first maximum of its window in
/// row-major order, the same choice PyTorch makes when values tie.
pub(crate) fn max_pool2d(x: &Tensor, kernel: usize, stride: usize) -> (Tensor, Vec<usize>) {
    let (n, c, h, w) = dims4(x, "max_pool2d input");
    assert!(
        kernel > 0 && stride > 0,
        "max_pool2d: kernel and stride must be positive"
    );
    assert!(
        h >= kernel && w >= kernel,
        "max_pool2d: kernel larger than input"
    );
    let (oh, ow) = ((h - kernel) / stride + 1, (w - kernel) / stride + 1);
    let mut y = Vec::with_capacity(n * c * oh * ow);
    let mut argmax = Vec::with_capacity(n * c * oh * ow);
    for plane in 0..n * c {
        for oy in 0..oh {
            for ox in 0..ow {
                let mut best = (f32::NEG_INFINITY, usize::MAX);
                for ky in 0..kernel {
                    for kx in 0..kernel {
                        let i = (plane * h + oy * stride + ky) * w + ox * stride + kx;
                        if x.data()[i] > best.0 || best.1 == usize::MAX {
                            best = (x.data()[i], i);
                        }
                    }
                }
                y.push(best.0);
                argmax.push(best.1);
            }
        }
    }
    (Tensor::new(vec![n, c, oh, ow], y), argmax)
}

/// Gradient of max pooling: each output gradient goes to its argmax.
pub(crate) fn max_pool2d_backward(x: &Tensor, kernel: usize, stride: usize, g: &Tensor) -> Tensor {
    let (y, argmax) = max_pool2d(x, kernel, stride);
    assert_eq!(g.shape(), y.shape(), "max_pool2d backward: gradient shape");
    let mut dx = vec![0.0f32; x.numel()];
    for (&i, &gv) in argmax.iter().zip(g.data()) {
        dx[i] += gv;
    }
    Tensor::new(x.shape().to_vec(), dx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_by_hand() {
        // One 3x3 input, one 2x2 kernel, stride 1, no padding, bias 0.5.
        let x = Tensor::new(vec![1, 1, 3, 3], (1..=9).map(|v| v as f32).collect());
        let w = Tensor::new(vec![1, 1, 2, 2], vec![1.0, 0.0, 0.0, -1.0]);
        let b = Tensor::new(vec![1], vec![0.5]);
        // Each output is x[i][j] - x[i+1][j+1] + 0.5 = -4 + 0.5.
        assert_eq!(conv2d(&x, &w, &b, 1, 0).data(), &[-3.5; 4]);
    }

    #[test]
    fn padding_reads_zeros() {
        let x = Tensor::new(vec![1, 1, 1, 1], vec![2.0]);
        let w = Tensor::new(vec![1, 1, 3, 3], vec![1.0; 9]);
        let b = Tensor::new(vec![1], vec![0.0]);
        // With padding 1 the only nonzero tap is the center.
        assert_eq!(conv2d(&x, &w, &b, 1, 1).data(), &[2.0]);
    }

    #[test]
    fn max_pool_picks_the_first_of_equal_maxima() {
        let x = Tensor::new(vec![1, 1, 2, 2], vec![1.0, 3.0, 3.0, 3.0]);
        let (y, argmax) = max_pool2d(&x, 2, 2);
        assert_eq!(y.data(), &[3.0]);
        assert_eq!(argmax, vec![1]);
    }
}
