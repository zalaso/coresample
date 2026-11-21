//! cs-compile: compiles the quantized bars CNN to an RV32IM program.
//!
//!   cs-compile --ref sim/nn/tests/data/quant_cnn.ref --samples 48 -o cnn.S
//!
//! Writes the assembly and, next to it, the region table (`.regions.json`).
//! Inputs are the reference images quantized with the network's input
//! parameters; the expected outputs come from the Rust int8 reference.

use coresample_compiler::compile;
use coresample_nn::models::quantized_bars_cnn;
use coresample_nn::quant::quantize_u8;
use coresample_nn::refdata::RefFile;
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut reference = None;
    let mut samples = None;
    let mut output = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--ref" => reference = args.next().map(PathBuf::from),
            "--samples" => samples = args.next().and_then(|s| s.parse::<usize>().ok()),
            "-o" => output = args.next().map(PathBuf::from),
            _ => {
                eprintln!("unknown argument {arg}");
                return ExitCode::from(2);
            }
        }
    }
    let (Some(reference), Some(output)) = (reference, output) else {
        eprintln!("usage: cs-compile --ref quant_cnn.ref [--samples N] -o program.S");
        return ExitCode::from(2);
    };

    let r = match RefFile::load(&reference) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{}: {e}", reference.display());
            return ExitCode::from(1);
        }
    };
    let net = quantized_bars_cnn(&r);
    let all = quantize_u8(&r["qcnn.x"], net.input);
    let per_sample: usize = net.input_shape.iter().product();
    let available = all.len() / per_sample;
    let n = samples.unwrap_or(available);
    if n == 0 || n > available {
        eprintln!("--samples must be between 1 and {available}");
        return ExitCode::from(2);
    }
    let inputs = &all[..n * per_sample];
    let expected = net.forward(inputs, n);
    let program = compile(&net, inputs, n, &expected);

    let regions = output.with_extension("regions.json");
    if let Err(e) = std::fs::write(&output, &program.asm)
        .and_then(|_| std::fs::write(&regions, program.regions_json()))
    {
        eprintln!("cannot write output: {e}");
        return ExitCode::from(1);
    }
    println!(
        "{}: {n} samples, {} regions; {}",
        output.display(),
        program.regions.len(),
        regions.display()
    );
    ExitCode::SUCCESS
}
