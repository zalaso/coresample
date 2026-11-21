//! Models built from graph operations. The compiler stages start from these.

use crate::graph::{Graph, NodeId};
use std::collections::HashMap;

/// A one-block GPT: token and position embeddings, a pre-norm transformer
/// block (causal multi-head self-attention, then a GELU MLP, each with a
/// residual connection), a final layer norm and a linear head.
#[derive(Clone, Copy, Debug)]
pub struct GptConfig {
    pub vocab: usize,
    /// Maximum sequence length (rows of the position embedding).
    pub seq: usize,
    pub d: usize,
    pub heads: usize,
    pub hidden: usize,
}

impl GptConfig {
    /// Parameter names, as `gpt_forward` expects them. Linear weights are (in, out).
    pub const PARAM_NAMES: [&'static str; 22] = [
        "tok", "pos", "ln1_g", "ln1_b", "wq", "bq", "wk", "bk", "wv", "bv", "wo", "bo", "ln2_g",
        "ln2_b", "w1", "b1", "w2", "b2", "lnf_g", "lnf_b", "wout", "bout",
    ];
}

const LN_EPS: f32 = 1e-5;

/// Logits (batch * t, vocab) for `tokens`, `batch` sequences of t tokens each,
/// laid out one sequence after another.
pub fn gpt_forward(
    cfg: &GptConfig,
    g: &mut Graph,
    p: &HashMap<&str, NodeId>,
    tokens: &[usize],
    batch: usize,
) -> NodeId {
    assert!(
        batch > 0 && tokens.len() % batch == 0,
        "gpt_forward: tokens must split into {batch} sequences"
    );
    assert_eq!(
        cfg.d % cfg.heads,
        0,
        "gpt_forward: d must be a multiple of heads"
    );
    let t = tokens.len() / batch;
    assert!(
        t <= cfg.seq,
        "gpt_forward: sequence of {t} tokens, model supports {}",
        cfg.seq
    );
    let (h, dh, d) = (cfg.heads, cfg.d / cfg.heads, cfg.d);
    let linear = |g: &mut Graph, x: NodeId, w: &str, b: &str| {
        let y = g.matmul(x, p[w]);
        g.add_bias(y, p[b])
    };
    // (batch * t, d) -> (batch * heads, t, dh)
    let split_heads = |g: &mut Graph, z: NodeId| {
        let z = g.reshape(z, &[batch, t, h, dh]);
        let z = g.permute(z, &[0, 2, 1, 3]);
        g.reshape(z, &[batch * h, t, dh])
    };

    let positions: Vec<usize> = (0..batch).flat_map(|_| 0..t).collect();
    let tok = g.embedding(p["tok"], tokens);
    let pos = g.embedding(p["pos"], &positions);
    let mut x = g.add(tok, pos);

    // Attention.
    let n = g.layer_norm(x, p["ln1_g"], p["ln1_b"], LN_EPS);
    let q = linear(g, n, "wq", "bq");
    let q = split_heads(g, q);
    let k = linear(g, n, "wk", "bk");
    let k = split_heads(g, k);
    let v = linear(g, n, "wv", "bv");
    let v = split_heads(g, v);
    let kt = g.permute(k, &[0, 2, 1]);
    let scores = g.batched_matmul(q, kt);
    let scores = g.scale(scores, (dh as f64).sqrt().recip() as f32);
    let scores = g.causal_mask(scores);
    let att = g.softmax(scores);
    let ctx = g.batched_matmul(att, v);
    let ctx = g.reshape(ctx, &[batch, h, t, dh]);
    let ctx = g.permute(ctx, &[0, 2, 1, 3]);
    let ctx = g.reshape(ctx, &[batch * t, d]);
    let out = linear(g, ctx, "wo", "bo");
    x = g.add(x, out);

    // MLP.
    let n = g.layer_norm(x, p["ln2_g"], p["ln2_b"], LN_EPS);
    let m = linear(g, n, "w1", "b1");
    let m = g.gelu(m);
    let m = linear(g, m, "w2", "b2");
    x = g.add(x, m);

    let n = g.layer_norm(x, p["lnf_g"], p["lnf_b"], LN_EPS);
    linear(g, n, "wout", "bout")
}
