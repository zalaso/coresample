//! Dense f32 tensors in row-major order.

/// A dense f32 tensor. The last dimension is contiguous in memory.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    /// Panics if `data` does not have exactly one value per element of `shape`.
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            numel,
            "tensor of shape {shape:?} needs {numel} values, got {}",
            data.len()
        );
        Self { shape, data }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let numel = shape.iter().product();
        Self {
            shape,
            data: vec![0.0; numel],
        }
    }

    /// A 0-dimensional tensor holding one value.
    pub fn scalar(value: f32) -> Self {
        Self {
            shape: vec![],
            data: vec![value],
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Same values, new shape with the same number of elements.
    pub fn reshape(&self, shape: &[usize]) -> Tensor {
        Tensor::new(shape.to_vec(), self.data.clone())
    }

    /// (rows, columns) of a matrix; panics for any other rank.
    pub fn dims2(&self) -> (usize, usize) {
        match self.shape[..] {
            [r, c] => (r, c),
            _ => panic!("expected a matrix, got shape {:?}", self.shape),
        }
    }

    /// Element-wise sum, for tensors of the same shape.
    pub(crate) fn add_assign(&mut self, other: &Tensor) {
        assert_eq!(self.shape, other.shape, "add: shape mismatch");
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += b;
        }
    }
}

/// `a @ b` for matrices (n, k) and (k, m).
pub(crate) fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let (n, k) = a.dims2();
    let (k2, m) = b.dims2();
    assert_eq!(
        k,
        k2,
        "matmul: shape mismatch {:?} @ {:?}",
        a.shape(),
        b.shape()
    );
    let mut out = vec![0.0f32; n * m];
    for i in 0..n {
        for p in 0..k {
            let av = a.data[i * k + p];
            for j in 0..m {
                out[i * m + j] += av * b.data[p * m + j];
            }
        }
    }
    Tensor::new(vec![n, m], out)
}

/// `a @ b^T` for matrices (n, k) and (m, k).
pub(crate) fn matmul_bt(a: &Tensor, b: &Tensor) -> Tensor {
    let (n, k) = a.dims2();
    let (m, k2) = b.dims2();
    assert_eq!(k, k2, "matmul_bt: shape mismatch");
    let mut out = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a.data[i * k + p] * b.data[j * k + p];
            }
            out[i * m + j] = acc;
        }
    }
    Tensor::new(vec![n, m], out)
}

/// `a^T @ b` for matrices (k, n) and (k, m).
pub(crate) fn matmul_at(a: &Tensor, b: &Tensor) -> Tensor {
    let (k, n) = a.dims2();
    let (k2, m) = b.dims2();
    assert_eq!(k, k2, "matmul_at: shape mismatch");
    let mut out = vec![0.0f32; n * m];
    for p in 0..k {
        for i in 0..n {
            let av = a.data[p * n + i];
            for j in 0..m {
                out[i * m + j] += av * b.data[p * m + j];
            }
        }
    }
    Tensor::new(vec![n, m], out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_small() {
        let a = Tensor::new(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]);
        let b = Tensor::new(vec![3, 2], vec![7., 8., 9., 10., 11., 12.]);
        assert_eq!(matmul(&a, &b).data(), &[58., 64., 139., 154.]);
    }

    #[test]
    fn transposed_products_agree_with_matmul() {
        let a = Tensor::new(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]);
        let b = Tensor::new(vec![2, 3], vec![0.5, -1., 2., 3., 0., -2.]);
        // a @ b^T and a^T @ b, written out by hand.
        assert_eq!(matmul_bt(&a, &b).data(), &[4.5, -3., 9., 0.]);
        assert_eq!(
            matmul_at(&a, &b).data(),
            &[12.5, -1., -6., 16., -2., -6., 19.5, -3., -6.]
        );
    }

    #[test]
    #[should_panic(expected = "needs 6 values")]
    fn new_checks_the_element_count() {
        Tensor::new(vec![2, 3], vec![0.0; 5]);
    }
}
