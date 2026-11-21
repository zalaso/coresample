//! Assembly text with regions: every emitted instruction belongs to exactly
//! one region, which names the function, the network layer and the operation
//! the instruction serves.
//!
//! Each region is bracketed by two local labels; the `.cs_map` section lists
//! (begin, end, region id) for every region, so after linking the ELF itself
//! maps each instruction address to its operation.

/// What a stretch of code does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    pub id: usize,
    /// The function it belongs to (`main`, `layer0_conv2d`, ...).
    pub function: String,
    /// Index of the network layer, or `None` for the driver code in `main`.
    pub layer: Option<usize>,
    /// The operation: `prologue`, `loop`, `bias`, `mac`, `requant`, `store`, `max`, `check`, `epilogue`.
    pub op: &'static str,
}

#[derive(Default)]
pub(crate) struct Asm {
    text: String,
    pub(crate) regions: Vec<Region>,
    open: Option<usize>,
    labels: usize,
    function: String,
    layer: Option<usize>,
}

impl Asm {
    /// One instruction.
    pub(crate) fn i(&mut self, ins: impl AsRef<str>) {
        assert!(
            self.open.is_some(),
            "instruction outside a region: {}",
            ins.as_ref()
        );
        self.text.push_str("  ");
        self.text.push_str(ins.as_ref());
        self.text.push('\n');
    }

    /// A directive or label, written as is.
    pub(crate) fn raw(&mut self, line: impl AsRef<str>) {
        self.text.push_str(line.as_ref());
        self.text.push('\n');
    }

    pub(crate) fn label(&mut self, l: &str) {
        self.raw(format!("{l}:"));
    }

    /// A new unique local label.
    pub(crate) fn fresh(&mut self, stem: &str) -> String {
        self.labels += 1;
        format!(".L{stem}_{}", self.labels)
    }

    /// Starts a region; the previous one ends here.
    pub(crate) fn region(&mut self, op: &'static str) {
        self.close();
        let id = self.regions.len();
        self.regions.push(Region {
            id,
            function: self.function.clone(),
            layer: self.layer,
            op,
        });
        self.raw(format!(".Lcs_r{id}_b:"));
        self.open = Some(id);
    }

    fn close(&mut self) {
        if let Some(id) = self.open.take() {
            self.raw(format!(".Lcs_r{id}_e:"));
        }
    }

    /// A global function; `body` emits its code, starting with a region.
    pub(crate) fn function(
        &mut self,
        name: &str,
        layer: Option<usize>,
        body: impl FnOnce(&mut Asm),
    ) {
        self.function = name.to_string();
        self.layer = layer;
        self.raw(format!(
            "\n  .globl {name}\n  .type {name}, @function\n  .p2align 2\n{name}:"
        ));
        body(self);
        self.close();
        self.raw(format!("  .size {name}, .-{name}"));
    }

    /// The finished text plus the `.cs_map` section.
    pub(crate) fn finish(mut self) -> (String, Vec<Region>) {
        self.close();
        self.raw("\n  .section .cs_map, \"\", @progbits");
        for r in 0..self.regions.len() {
            self.raw(format!("  .word .Lcs_r{r}_b, .Lcs_r{r}_e, {r}"));
        }
        (self.text, self.regions)
    }
}
