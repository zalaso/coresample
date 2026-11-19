//! Reader for the reference files written by `reference/gen_reference.py`.
//!
//! Format (little-endian): `b"CSREF001"`, u32 tensor count, then per tensor:
//! u32 name length, name (UTF-8), u32 ndim, u32 dims[ndim], f32 data.

use crate::tensor::Tensor;
use std::collections::BTreeMap;
use std::ops::Index;
use std::path::Path;

const MAGIC: &[u8; 8] = b"CSREF001";

/// Named tensors loaded from one reference file.
#[derive(Debug)]
pub struct RefFile {
    tensors: BTreeMap<String, Tensor>,
}

impl RefFile {
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        Self::parse(&bytes)
    }

    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let mut r = Reader { bytes, pos: 0 };
        if r.take(8)? != MAGIC {
            return Err("not a CSREF001 file".into());
        }
        let count = r.u32()?;
        let mut tensors = BTreeMap::new();
        for _ in 0..count {
            let len = r.u32()? as usize;
            let name = String::from_utf8(r.take(len)?.to_vec()).map_err(|e| e.to_string())?;
            let ndim = r.u32()? as usize;
            let shape = (0..ndim)
                .map(|_| r.u32().map(|d| d as usize))
                .collect::<Result<Vec<_>, _>>()?;
            let numel: usize = shape.iter().product();
            let data = r
                .take(numel * 4)?
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            if tensors
                .insert(name.clone(), Tensor::new(shape, data))
                .is_some()
            {
                return Err(format!("duplicate tensor {name}"));
            }
        }
        if r.pos != bytes.len() {
            return Err(format!("{} trailing bytes", bytes.len() - r.pos));
        }
        Ok(Self { tensors })
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }
}

impl Index<&str> for RefFile {
    type Output = Tensor;
    fn index(&self, name: &str) -> &Tensor {
        self.get(name)
            .unwrap_or_else(|| panic!("no tensor named {name:?} in reference file"))
    }
}

impl Index<&String> for RefFile {
    type Output = Tensor;
    fn index(&self, name: &String) -> &Tensor {
        &self[name.as_str()]
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.bytes.len())
            .ok_or("truncated file")?;
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u32(&mut self) -> Result<u32, String> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(name: &str, shape: &[u32], data: &[f32]) -> Vec<u8> {
        let mut b = MAGIC.to_vec();
        b.extend(1u32.to_le_bytes());
        b.extend((name.len() as u32).to_le_bytes());
        b.extend(name.as_bytes());
        b.extend((shape.len() as u32).to_le_bytes());
        shape.iter().for_each(|d| b.extend(d.to_le_bytes()));
        data.iter().for_each(|v| b.extend(v.to_le_bytes()));
        b
    }

    #[test]
    fn round_trip() {
        let r = RefFile::parse(&file("w", &[2, 2], &[1.0, -2.5, 0.0, 3.25])).unwrap();
        assert_eq!(r["w"], Tensor::new(vec![2, 2], vec![1.0, -2.5, 0.0, 3.25]));
    }

    #[test]
    fn rejects_bad_files() {
        assert!(RefFile::parse(b"NOTAREF!").is_err());
        let mut b = file("w", &[2], &[1.0, 2.0]);
        b.pop();
        assert!(RefFile::parse(&b).unwrap_err().contains("truncated"));
        let mut b = file("w", &[1], &[1.0]);
        b.push(0);
        assert!(RefFile::parse(&b).unwrap_err().contains("trailing"));
    }
}
