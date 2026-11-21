//! Compiled networks run on the Coresample core (and on Spike) and must
//! reproduce `QNetwork::forward`, the Rust int8 reference, byte for byte.
//!
//! Each program checks its own output against the expected bytes and reports
//! through `tohost`: pass, or fail with 1 + the index of the first mismatch.
//! Needs the RISC-V GCC toolchain and the simulator (`make sim`).

use coresample_compiler::elf::read_section;
use coresample_compiler::{compile, Program};
use coresample_nn::models::quantized_bars_cnn;
use coresample_nn::quant::{quantize_u8, QConv2d, QLayer, QLinear, QNetwork, QParams, Requant};
use coresample_nn::refdata::RefFile;
use std::path::{Path, PathBuf};
use std::process::Command;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn sim() -> PathBuf {
    let p = root().join("build/sim/cs_sim");
    assert!(
        p.exists(),
        "{} not found: run `make sim` first",
        p.display()
    );
    p
}

fn spike() -> String {
    if let Ok(s) = std::env::var("SPIKE") {
        return s;
    }
    let local = PathBuf::from(std::env::var("HOME").unwrap()).join(".local/bin/spike");
    if local.exists() {
        local.display().to_string()
    } else {
        "spike".into()
    }
}

struct Built {
    elf: PathBuf,
    program: Program,
    expected: Vec<u8>,
}

/// Compiles `net` for `n` samples, with the expected output from the Rust
/// reference (optionally corrupted, to test the self-check), and links it.
fn build(name: &str, net: &QNetwork, inputs: &[u8], n: usize, corrupt: Option<usize>) -> Built {
    let mut expected = net.forward(inputs, n);
    if let Some(i) = corrupt {
        expected[i] ^= 1;
    }
    let program = compile(net, inputs, n, &expected);
    let dir = root().join("target/compiler-tests").join(name);
    std::fs::create_dir_all(&dir).unwrap();
    let asm = dir.join("program.S");
    std::fs::write(&asm, &program.asm).unwrap();
    std::fs::write(dir.join("regions.json"), program.regions_json()).unwrap();
    let elf = dir.join("program.elf");
    let out = Command::new("riscv64-unknown-elf-gcc")
        .args([
            "-march=rv32im_zicsr",
            "-mabi=ilp32",
            "-static",
            "-mcmodel=medany",
            "-nostdlib",
            "-nostartfiles",
        ])
        .arg("-Wl,--no-warn-rwx-segments")
        .arg(format!("-T{}", root().join("sw/runtime/link.ld").display()))
        .arg(root().join("sw/runtime/crt0.S"))
        .arg(&asm)
        .arg("-o")
        .arg(&elf)
        .output()
        .expect("riscv64-unknown-elf-gcc not found");
    assert!(
        out.status.success(),
        "assembling {name} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    Built {
        elf,
        program,
        expected,
    }
}

/// The simulator's RESULT line, and its commit log if `trace` is set.
fn run_core(elf: &Path, trace: Option<&Path>) -> String {
    let mut cmd = Command::new(sim());
    cmd.args(["--max-cycles", "50000000"]);
    if let Some(t) = trace {
        cmd.arg("--trace").arg(t);
    }
    let out = cmd.arg(elf).output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    eprintln!(
        "{}: {}",
        elf.parent().unwrap().file_name().unwrap().to_string_lossy(),
        stdout.replace('\n', " | ")
    );
    stdout
        .lines()
        .find(|l| l.starts_with("RESULT"))
        .unwrap_or("no RESULT line")
        .to_string()
}

/// A deterministic pseudo-random sequence (xorshift), so tests are reproducible.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo + 1) as u64) as i32
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.range(0, 255) as u8).collect()
    }
}

fn random_linear(rng: &mut Rng, inputs: usize, outputs: usize, m: f64, relu: bool) -> QLinear {
    QLinear {
        inputs,
        outputs,
        weight: (0..inputs * outputs)
            .map(|_| rng.range(-128, 127) as i8)
            .collect(),
        bias: (0..outputs).map(|_| rng.range(-20000, 20000)).collect(),
        x_zero_point: rng.range(0, 255),
        y_zero_point: rng.range(0, 255),
        requant: Requant::from_scale(m),
        relu,
    }
}

fn net(input_shape: Vec<usize>, layers: Vec<QLayer>) -> QNetwork {
    QNetwork {
        input: QParams {
            scale: 1.0,
            zero_point: 0,
        },
        input_shape,
        layers,
    }
}

#[test]
fn linear_layers_with_every_requantization_path() {
    // m >= 0.5 gives shift 31 (the 64-bit shift combines both words); smaller
    // m gives shift > 32 (only the high word); ReLU on and off; saturation both ways.
    let mut rng = Rng(1);
    for (i, &(m, relu)) in [
        (0.7, false),
        (0.6, true),
        (0.003, false),
        (0.0004, true),
        (2e-7, false),
    ]
    .iter()
    .enumerate()
    {
        let layer = random_linear(&mut rng, 23, 5, m, relu);
        assert_eq!(
            layer.requant.shift == 31,
            m >= 0.5,
            "case {i} covers the intended path"
        );
        let n = 6;
        let net = net(vec![23], vec![QLayer::Linear(layer)]);
        let b = build(&format!("linear{i}"), &net, &rng.bytes(23 * n), n, None);
        assert_eq!(
            run_core(&b.elf, None),
            "RESULT PASS",
            "case {i}: m {m}, relu {relu}"
        );
    }
}

#[test]
fn convolutions_of_several_geometries() {
    let mut rng = Rng(2);
    // (C_in, H, W, C_out, kH, kW, stride, padding, relu)
    let cases = [
        (1, 6, 6, 2, 3, 3, 1, 1, true),
        (2, 5, 7, 3, 3, 2, 2, 1, false),
        (3, 6, 5, 2, 2, 2, 1, 0, true),
    ];
    for (i, &(cin, h, w, cout, kh, kw, stride, padding, relu)) in cases.iter().enumerate() {
        let conv = QConv2d {
            in_channels: cin,
            out_channels: cout,
            kernel: (kh, kw),
            stride,
            padding,
            weight: (0..cout * cin * kh * kw)
                .map(|_| rng.range(-128, 127) as i8)
                .collect(),
            bias: (0..cout).map(|_| rng.range(-5000, 5000)).collect(),
            x_zero_point: rng.range(0, 255),
            y_zero_point: rng.range(0, 255),
            requant: Requant::from_scale(0.002),
            relu,
        };
        let n = 2;
        let net = net(vec![cin, h, w], vec![QLayer::Conv2d(conv)]);
        let b = build(
            &format!("conv{i}"),
            &net,
            &rng.bytes(cin * h * w * n),
            n,
            None,
        );
        assert_eq!(run_core(&b.elf, None), "RESULT PASS", "case {i}");
    }
}

#[test]
fn max_pool_flatten_linear() {
    let mut rng = Rng(3);
    let lin = random_linear(&mut rng, 2 * 3 * 3, 4, 0.01, false);
    let net = net(
        vec![2, 6, 7],
        vec![
            QLayer::MaxPool2d {
                kernel: 2,
                stride: 2,
            },
            QLayer::Flatten,
            QLayer::Linear(lin),
        ],
    );
    let b = build("pool", &net, &rng.bytes(2 * 6 * 7 * 3), 3, None);
    assert_eq!(run_core(&b.elf, None), "RESULT PASS");
}

#[test]
fn self_check_reports_the_first_mismatch() {
    let mut rng = Rng(4);
    let net = net(
        vec![8],
        vec![QLayer::Linear(random_linear(&mut rng, 8, 3, 0.01, false))],
    );
    let b = build("corrupt", &net, &rng.bytes(8 * 4), 4, Some(7));
    assert_eq!(run_core(&b.elf, None), "RESULT FAIL 8");
}

fn bars_cnn(n: usize) -> (QNetwork, Vec<u8>) {
    let r = RefFile::load(&root().join("sim/nn/tests/data/quant_cnn.ref")).unwrap();
    let net = quantized_bars_cnn(&r);
    let x = quantize_u8(&r["qcnn.x"], net.input);
    (net, x[..n * 64].to_vec())
}

/// G2: the bars CNN on the core, all 48 images, output identical to the int8
/// reference (which matches PyTorch's quantized model on every integer).
#[test]
fn g2_bars_cnn_on_the_core() {
    let (net, x) = bars_cnn(48);
    let b = build("g2_cnn48", &net, &x, 48, None);
    assert_eq!(b.expected.len(), 48 * 3);
    assert_eq!(run_core(&b.elf, None), "RESULT PASS");
    let spike = Command::new(spike())
        .args(["--isa=rv32im", "-m0x80000000:0x100000"])
        .arg(&b.elf)
        .status();
    assert!(
        spike.expect("spike not found").success(),
        "the same program must also pass on Spike"
    );
}

/// The compiled CNN retires exactly the same instructions as on Spike.
#[test]
fn g2_bars_cnn_identical_to_spike_instruction_by_instruction() {
    let (net, x) = bars_cnn(2);
    let b = build("g2_cnn2", &net, &x, 2, None);
    let out = Command::new("python3")
        .arg(root().join("scripts/run_tests.py"))
        .arg("--sim")
        .arg(sim())
        .arg("--spike")
        .arg(spike())
        .arg("--trace-dir")
        .arg(b.elf.parent().unwrap())
        .arg(&b.elf)
        .output()
        .unwrap();
    let report = String::from_utf8_lossy(&out.stdout);
    eprintln!("{report}");
    assert!(
        out.status.success() && report.contains("identical to Spike"),
        "{report}"
    );
}

/// The `.cs_map` section maps code to (layer, operation): regions do not
/// overlap, and every instruction the network retires lies in exactly one.
#[test]
fn region_map_covers_every_retired_instruction() {
    let (net, x) = bars_cnn(1);
    let b = build("g2_map", &net, &x, 1, None);
    let elf = std::fs::read(&b.elf).unwrap();
    let raw = read_section(&elf, ".cs_map").expect("no .cs_map section");
    let mut regions: Vec<(u32, u32, u32)> = raw
        .chunks_exact(12)
        .map(|c| {
            let w = |i: usize| u32::from_le_bytes(c[i..i + 4].try_into().unwrap());
            (w(0), w(4), w(8))
        })
        .collect();
    assert_eq!(
        regions.len(),
        b.program.regions.len(),
        "one map entry per region"
    );
    regions.sort();
    for pair in regions.windows(2) {
        assert!(pair[0].0 < pair[0].1, "empty region {:?}", pair[0]);
        assert!(
            pair[0].1 <= pair[1].0,
            "regions overlap: {:?} {:?}",
            pair[0],
            pair[1]
        );
    }

    let trace = b.elf.with_extension("core.log");
    assert_eq!(run_core(&b.elf, Some(&trace)), "RESULT PASS");
    let (lo, hi) = (regions[0].0, regions.last().unwrap().1);
    let mut per_layer = std::collections::BTreeMap::<String, usize>::new();
    let mut in_network = 0;
    for line in std::fs::read_to_string(&trace).unwrap().lines() {
        let pc = u32::from_str_radix(&line.split_whitespace().nth(3).unwrap()[2..], 16).unwrap();
        if pc < lo || pc >= hi {
            continue; // start-up code in crt0
        }
        let hits: Vec<_> = regions.iter().filter(|r| r.0 <= pc && pc < r.1).collect();
        assert_eq!(hits.len(), 1, "pc 0x{pc:08x} is in {} regions", hits.len());
        let region = &b.program.regions[hits[0].2 as usize];
        *per_layer.entry(region.function.clone()).or_default() += 1;
        in_network += 1;
    }
    eprintln!("retired instructions per function: {per_layer:?}");
    assert!(in_network > 0);
    for f in [
        "main",
        "layer0_conv2d",
        "layer1_max_pool2d",
        "layer3_linear",
    ] {
        assert!(per_layer.contains_key(f), "{f} never ran");
    }
}
