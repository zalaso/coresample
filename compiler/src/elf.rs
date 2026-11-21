//! Minimal ELF32 little-endian reader: enough to find a section by name.

fn u16_at(b: &[u8], off: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(off..off + 2)?.try_into().ok()?))
}

fn u32_at(b: &[u8], off: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(off..off + 4)?.try_into().ok()?))
}

/// The contents of section `name`, or `None` if the file is not a 32-bit
/// little-endian ELF or has no such section.
pub fn read_section<'a>(elf: &'a [u8], name: &str) -> Option<&'a [u8]> {
    if elf.get(0..4)? != b"\x7fELF" || elf[4] != 1 || elf[5] != 1 {
        return None; // not ELFCLASS32 / ELFDATA2LSB
    }
    let shoff = u32_at(elf, 0x20)? as usize;
    let shentsize = u16_at(elf, 0x2e)? as usize;
    let shnum = u16_at(elf, 0x30)? as usize;
    let shstrndx = u16_at(elf, 0x32)? as usize;
    let header = |i: usize| shoff + i * shentsize;
    let strtab_off = u32_at(elf, header(shstrndx) + 16)? as usize;
    for i in 0..shnum {
        let h = header(i);
        let name_off = strtab_off + u32_at(elf, h)? as usize;
        let end = elf.get(name_off..)?.iter().position(|&c| c == 0)?;
        if &elf[name_off..name_off + end] == name.as_bytes() {
            let off = u32_at(elf, h + 16)? as usize;
            let size = u32_at(elf, h + 20)? as usize;
            return elf.get(off..off + size);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_elf_input() {
        assert!(read_section(b"not an elf file at all", ".text").is_none());
        assert!(read_section(b"", ".text").is_none());
    }
}
