//! Computation graph with reverse-mode automatic differentiation.
//!
//! Every operation is a node with an explicit [`Op`], not an opaque closure,
//! so the graph can be inspected: later stages of Coresample map compiled
//! instructions back to these nodes.
//!
//! Nodes are only ever appended, and an operation's inputs always exist
//! before it, so the node order is already a topological order. Backward
//! walks the nodes in reverse.

use crate::conv::{conv2d, conv2d_backward, max_pool2d, max_pool2d_backward};
use crate::tensor::{matmul, matmul_at, matmul_bt, Tensor};

/// Handle to a node of one [`Graph`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

/// What a node computes.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// Data that needs no gradient (network inputs).
    Input,
    /// A trainable parameter; backward fills its gradient.
    Param,
    /// `a @ b` for matrices (n, k) and (k, m).
    MatMul(NodeId, NodeId),
    /// `a + b`, same shape.
    Add(NodeId, NodeId),
    /// `x + b` with x of shape (n, m) and b of shape (m), b added to every row.
    AddBias(NodeId, NodeId),
    /// `max(x, 0)` element-wise. The gradient at exactly 0 is 0, as in PyTorch.
    Relu(NodeId),
    /// Mean over the batch of `-log softmax(logits)[target]`; a scalar.
    CrossEntropy { logits: NodeId, targets: Vec<usize> },
    /// 2-D cross-correlation plus bias on NCHW tensors, weight (C_out, C_in, kH, kW),
    /// zero padding on every side.
    Conv2d {
        x: NodeId,
        w: NodeId,
        b: NodeId,
        stride: usize,
        padding: usize,
    },
    /// Max over square windows, no padding. Ties go to the first maximum in
    /// row-major order, as in PyTorch.
    MaxPool2d {
        x: NodeId,
        kernel: usize,
        stride: usize,
    },
    /// Same values, new shape (the node's own shape).
    Reshape(NodeId),
}

impl Op {
    /// Short name, for traces and error messages.
    pub fn name(&self) -> &'static str {
        match self {
            Op::Input => "input",
            Op::Param => "param",
            Op::MatMul(..) => "matmul",
            Op::Add(..) => "add",
            Op::AddBias(..) => "add_bias",
            Op::Relu(..) => "relu",
            Op::CrossEntropy { .. } => "cross_entropy",
            Op::Conv2d { .. } => "conv2d",
            Op::MaxPool2d { .. } => "max_pool2d",
            Op::Reshape(..) => "reshape",
        }
    }

    fn inputs(&self) -> Vec<NodeId> {
        match self {
            Op::Input | Op::Param => vec![],
            Op::MatMul(a, b) | Op::Add(a, b) | Op::AddBias(a, b) => vec![*a, *b],
            Op::Relu(x) | Op::Reshape(x) | Op::MaxPool2d { x, .. } => vec![*x],
            Op::CrossEntropy { logits, .. } => vec![*logits],
            Op::Conv2d { x, w, b, .. } => vec![*x, *w, *b],
        }
    }
}

struct Node {
    op: Op,
    value: Tensor,
    requires_grad: bool,
    grad: Option<Tensor>,
}

/// A computation graph: build it forward, then call [`Graph::backward`].
#[derive(Default)]
pub struct Graph {
    nodes: Vec<Node>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn input(&mut self, value: Tensor) -> NodeId {
        self.push(Op::Input, value)
    }

    pub fn param(&mut self, value: Tensor) -> NodeId {
        self.push(Op::Param, value)
    }

    pub fn matmul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let value = matmul(self.value(a), self.value(b));
        self.push(Op::MatMul(a, b), value)
    }

    pub fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let mut value = self.value(a).clone();
        value.add_assign(self.value(b));
        self.push(Op::Add(a, b), value)
    }

    pub fn add_bias(&mut self, x: NodeId, b: NodeId) -> NodeId {
        let (n, m) = self.value(x).dims2();
        assert_eq!(
            self.value(b).shape(),
            &[m],
            "add_bias: bias shape must be [{m}]"
        );
        let mut value = self.value(x).clone();
        let bias = self.value(b).data().to_vec();
        for row in value.data_mut().chunks_exact_mut(m).take(n) {
            for (v, bv) in row.iter_mut().zip(&bias) {
                *v += bv;
            }
        }
        self.push(Op::AddBias(x, b), value)
    }

    pub fn relu(&mut self, x: NodeId) -> NodeId {
        let src = self.value(x);
        let value = Tensor::new(
            src.shape().to_vec(),
            src.data().iter().map(|&v| v.max(0.0)).collect(),
        );
        self.push(Op::Relu(x), value)
    }

    pub fn cross_entropy(&mut self, logits: NodeId, targets: &[usize]) -> NodeId {
        let (n, c) = self.value(logits).dims2();
        assert_eq!(targets.len(), n, "cross_entropy: one target per row");
        assert!(
            targets.iter().all(|&t| t < c),
            "cross_entropy: target out of range"
        );
        let data = self.value(logits).data();
        let mut total = 0.0f32;
        for (row, &t) in data.chunks_exact(c).zip(targets) {
            total += log_sum_exp(row) - row[t];
        }
        let value = Tensor::scalar(total / n as f32);
        self.push(
            Op::CrossEntropy {
                logits,
                targets: targets.to_vec(),
            },
            value,
        )
    }

    pub fn conv2d(
        &mut self,
        x: NodeId,
        w: NodeId,
        b: NodeId,
        stride: usize,
        padding: usize,
    ) -> NodeId {
        let value = conv2d(self.value(x), self.value(w), self.value(b), stride, padding);
        self.push(
            Op::Conv2d {
                x,
                w,
                b,
                stride,
                padding,
            },
            value,
        )
    }

    pub fn max_pool2d(&mut self, x: NodeId, kernel: usize, stride: usize) -> NodeId {
        let (value, _) = max_pool2d(self.value(x), kernel, stride);
        self.push(Op::MaxPool2d { x, kernel, stride }, value)
    }

    pub fn reshape(&mut self, x: NodeId, shape: &[usize]) -> NodeId {
        let value = self.value(x).reshape(shape);
        self.push(Op::Reshape(x), value)
    }

    /// (N, ...) to (N, everything else), as `torch.flatten(x, 1)`.
    pub fn flatten(&mut self, x: NodeId) -> NodeId {
        let shape = self.value(x).shape();
        assert!(!shape.is_empty(), "flatten: needs a batch dimension");
        let n = shape[0];
        let rest = shape[1..].iter().product();
        self.reshape(x, &[n, rest])
    }

    pub fn value(&self, id: NodeId) -> &Tensor {
        &self.nodes[id.0].value
    }

    /// Gradient of the last backward pass; `None` for nodes that need none.
    pub fn grad(&self, id: NodeId) -> Option<&Tensor> {
        self.nodes[id.0].grad.as_ref()
    }

    pub fn op(&self, id: NodeId) -> &Op {
        &self.nodes[id.0].op
    }

    /// Every node in creation order, which is also a valid execution order.
    pub fn node_ids(&self) -> impl Iterator<Item = NodeId> {
        (0..self.nodes.len()).map(NodeId)
    }

    /// Backward from a scalar, with d(root)/d(root) = 1.
    pub fn backward(&mut self, root: NodeId) {
        let value = self.value(root);
        assert_eq!(
            value.numel(),
            1,
            "backward needs a scalar; use backward_with for {:?}",
            value.shape()
        );
        let seed = Tensor::new(value.shape().to_vec(), vec![1.0]);
        self.backward_with(root, seed);
    }

    /// Backward from any node, given the gradient flowing into it.
    pub fn backward_with(&mut self, root: NodeId, seed: Tensor) {
        assert_eq!(
            seed.shape(),
            self.value(root).shape(),
            "backward_with: seed shape"
        );
        for node in &mut self.nodes {
            node.grad = None;
        }
        self.nodes[root.0].grad = Some(seed);

        for i in (0..=root.0).rev() {
            let Some(g) = self.nodes[i].grad.take() else {
                continue;
            };
            for (input, contribution) in self.input_grads(i, &g) {
                if !self.nodes[input.0].requires_grad {
                    continue;
                }
                match &mut self.nodes[input.0].grad {
                    Some(acc) => acc.add_assign(&contribution),
                    slot => *slot = Some(contribution),
                }
            }
            self.nodes[i].grad = Some(g);
        }
        // Gradients stay readable on every node that requires one, intermediate
        // values included; the seed is dropped if the root itself needs none.
        for node in &mut self.nodes {
            if !node.requires_grad {
                node.grad = None;
            }
        }
    }

    fn push(&mut self, op: Op, value: Tensor) -> NodeId {
        let requires_grad = match op {
            Op::Input => false,
            Op::Param => true,
            _ => op.inputs().iter().any(|i| self.nodes[i.0].requires_grad),
        };
        self.nodes.push(Node {
            op,
            value,
            requires_grad,
            grad: None,
        });
        NodeId(self.nodes.len() - 1)
    }

    /// The gradient of each input of node `i`, given the gradient `g` of its output.
    fn input_grads(&self, i: usize, g: &Tensor) -> Vec<(NodeId, Tensor)> {
        match &self.nodes[i].op {
            Op::Input | Op::Param => vec![],
            Op::MatMul(a, b) => vec![
                (*a, matmul_bt(g, self.value(*b))), // dA = G @ B^T
                (*b, matmul_at(self.value(*a), g)), // dB = A^T @ G
            ],
            Op::Add(a, b) => vec![(*a, g.clone()), (*b, g.clone())],
            Op::AddBias(x, b) => {
                let (_, m) = g.dims2();
                let mut db = vec![0.0f32; m];
                for row in g.data().chunks_exact(m) {
                    for (acc, v) in db.iter_mut().zip(row) {
                        *acc += v;
                    }
                }
                vec![(*x, g.clone()), (*b, Tensor::new(vec![m], db))]
            }
            Op::Relu(x) => {
                let src = self.value(*x).data();
                let dx = g
                    .data()
                    .iter()
                    .zip(src)
                    .map(|(&gv, &xv)| if xv > 0.0 { gv } else { 0.0 })
                    .collect();
                vec![(*x, Tensor::new(g.shape().to_vec(), dx))]
            }
            Op::CrossEntropy { logits, targets } => {
                // d/dlogits = (softmax(logits) - onehot(target)) / n, times the upstream scalar.
                let (n, c) = self.value(*logits).dims2();
                let scale = g.data()[0] / n as f32;
                let mut d = Vec::with_capacity(n * c);
                for (row, &t) in self.value(*logits).data().chunks_exact(c).zip(targets) {
                    let lse = log_sum_exp(row);
                    for (j, &v) in row.iter().enumerate() {
                        let p = (v - lse).exp();
                        d.push((p - if j == t { 1.0 } else { 0.0 }) * scale);
                    }
                }
                vec![(*logits, Tensor::new(vec![n, c], d))]
            }
            Op::Conv2d {
                x,
                w,
                b,
                stride,
                padding,
            } => {
                let (dx, dw, db) = conv2d_backward(
                    self.value(*x),
                    self.value(*w),
                    self.value(*b),
                    g,
                    *stride,
                    *padding,
                );
                vec![(*x, dx), (*w, dw), (*b, db)]
            }
            Op::MaxPool2d { x, kernel, stride } => {
                vec![(*x, max_pool2d_backward(self.value(*x), *kernel, *stride, g))]
            }
            Op::Reshape(x) => vec![(*x, g.reshape(self.value(*x).shape()))],
        }
    }
}

/// `log(sum(exp(row)))`, shifted by the maximum so large logits do not overflow.
fn log_sum_exp(row: &[f32]) -> f32 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    max + row.iter().map(|&v| (v - max).exp()).sum::<f32>().ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_sum_exp_is_stable_for_large_logits() {
        let v = log_sum_exp(&[1000.0, 1000.0]);
        assert!((v - (1000.0 + 2f32.ln())).abs() < 1e-3, "{v}");
    }

    #[test]
    fn backward_needs_a_scalar() {
        let mut g = Graph::new();
        let x = g.param(Tensor::zeros(vec![2, 2]));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| g.backward(x)));
        assert!(result.is_err());
    }

    #[test]
    fn ops_are_inspectable() {
        let mut g = Graph::new();
        let x = g.input(Tensor::zeros(vec![1, 2]));
        let w = g.param(Tensor::zeros(vec![2, 2]));
        let y = g.matmul(x, w);
        let names: Vec<_> = g.node_ids().map(|id| g.op(id).name()).collect();
        assert_eq!(names, ["input", "param", "matmul"]);
        assert_eq!(g.op(y), &Op::MatMul(x, w));
    }
}
