//! Neural network simulator for Coresample: tensors, a computation graph with
//! reverse-mode autograd, and training, each operation checked against PyTorch
//! (see `tests/` and `reference/gen_reference.py`).
//!
//! Conventions: f32 everywhere, row-major tensors, and a linear layer is
//! `x @ W + b` with `W` of shape (in, out).

mod attention;
mod conv;
pub mod graph;
pub mod models;
pub mod quant;
pub mod refdata;
pub mod tensor;

pub use graph::{Graph, NodeId, Op};
pub use tensor::Tensor;

/// Plain SGD, as `torch.optim.SGD` without momentum: `p -= lr * grad`.
pub fn sgd_step(param: &mut Tensor, grad: &Tensor, lr: f32) {
    assert_eq!(param.shape(), grad.shape(), "sgd_step: gradient shape");
    for (p, g) in param.data_mut().iter_mut().zip(grad.data()) {
        *p -= lr * g;
    }
}
