//! Helpers shared by the integration tests.
use coresample_nn::refdata::RefFile;
use coresample_nn::Tensor;
use std::path::Path;

/// Tolerance against PyTorch: |actual - expected| <= ATOL + RTOL * |expected| for
/// every element. Not exact because summation order differs from PyTorch's kernels.
pub const ATOL: f32 = 1e-5;
pub const RTOL: f32 = 1e-4;

pub fn load(name: &str) -> RefFile {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    RefFile::load(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

pub fn assert_close(what: &str, actual: &Tensor, expected: &Tensor) {
    assert_eq!(actual.shape(), expected.shape(), "{what}: shape");
    for (i, (&a, &e)) in actual.data().iter().zip(expected.data()).enumerate() {
        let ok = (a - e).abs() <= ATOL + RTOL * e.abs();
        assert!(
            ok,
            "{what}[{i}]: got {a}, PyTorch {e} (diff {})",
            (a - e).abs()
        );
    }
}

/// Class labels are stored as f32 in the reference files.
pub fn labels(t: &Tensor) -> Vec<usize> {
    t.data().iter().map(|&v| v as usize).collect()
}
