//! Dependency-free structural validation of compiled CUDA cache images.
//! This file can be tested with rustc --test without initializing CUDA.

use std::ffi::{CString, c_char};

#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub(crate) enum CacheArtifactFormat {
    #[default]
    Ptx,
    Cubin,
}

/// Stream a versioned, length-delimited checksum input without choosing a hasher.
/// The production consumer uses XXH3-128 for accidental-bitrot detection only.
pub(crate) fn write_checksum_input(
    expected_kernel_hash: u128,
    format: CacheArtifactFormat,
    entrypoint_name: &str,
    shared_mem_bytes: u64,
    image: &[u8],
    mut write: impl FnMut(&[u8]),
) {
    write(b"teamy-cuda-cache-entry-v1\0");
    write(&expected_kernel_hash.to_le_bytes());
    let format = format.directory().as_bytes();
    write(&(format.len() as u64).to_le_bytes());
    write(format);
    write(&(entrypoint_name.len() as u64).to_le_bytes());
    write(entrypoint_name.as_bytes());
    write(&shared_mem_bytes.to_le_bytes());
    write(&(image.len() as u64).to_le_bytes());
    write(image);
}

impl CacheArtifactFormat {
    pub(crate) const fn selected() -> Self {
        if cfg!(feature = "teamy-cubin-cache") {
            Self::Cubin
        } else {
            Self::Ptx
        }
    }

    pub(crate) const fn directory(self) -> &'static str {
        match self {
            Self::Ptx => "ptx",
            Self::Cubin => "cubin",
        }
    }
}

pub(crate) fn validate_cache_entry(
    image: &[c_char],
    format: CacheArtifactFormat,
    entrypoint_name: &str,
) -> Result<CString, &'static str> {
    match format {
        CacheArtifactFormat::Ptx if image.last() != Some(&0) => {
            return Err("PTX cache entry is missing its terminating NUL");
        }
        CacheArtifactFormat::Cubin => validate_cubin(image)?,
        CacheArtifactFormat::Ptx => {}
    }
    CString::new(entrypoint_name).map_err(|_| "CUDA cache entry contains an invalid entry point")
}

fn elf_bytes<const N: usize>(image: &[c_char], offset: usize) -> Option<[u8; N]> {
    let bytes = image.get(offset..offset.checked_add(N)?)?;
    Some(std::array::from_fn(|index| bytes[index] as u8))
}

fn elf_u16(image: &[c_char], offset: usize) -> Option<u16> {
    elf_bytes(image, offset).map(u16::from_le_bytes)
}

fn elf_u32(image: &[c_char], offset: usize) -> Option<u32> {
    elf_bytes(image, offset).map(u32::from_le_bytes)
}

fn elf_offset(image: &[c_char], offset: usize) -> Option<usize> {
    usize::try_from(u64::from_le_bytes(elf_bytes(image, offset)?)).ok()
}

fn valid_image_range(image: &[c_char], offset: usize, size: usize) -> bool {
    offset
        .checked_add(size)
        .is_some_and(|end| end <= image.len())
}

fn validate_cubin(image: &[c_char]) -> Result<(), &'static str> {
    let invalid = || "CUBIN cache entry is not a bounded ELF64 module image";
    if elf_bytes::<7>(image, 0) != Some([0x7f, b'E', b'L', b'F', 2, 1, 1])
        || elf_u32(image, 20) != Some(1)
        || elf_u16(image, 52) != Some(64)
        || image.len() < 64
    {
        return Err(invalid());
    }

    let section_offset = elf_offset(image, 40).ok_or_else(invalid)?;
    let section_size = usize::from(elf_u16(image, 58).ok_or_else(invalid)?);
    let section_count = usize::from(elf_u16(image, 60).ok_or_else(invalid)?);
    let section_names = usize::from(elf_u16(image, 62).ok_or_else(invalid)?);
    // NVRTC modules use ordinary ELF64 counts. Reject extended-count encodings
    // instead of letting unvalidated offsets reach the driver.
    if section_names == 0xffff
        || (section_count == 0 && section_offset != 0)
        || (section_names != 0 && section_names >= section_count)
    {
        return Err(invalid());
    }
    if section_count != 0 {
        let table_size = section_size
            .checked_mul(section_count)
            .ok_or_else(invalid)?;
        if section_offset < 64
            || section_size < 64
            || !valid_image_range(image, section_offset, table_size)
        {
            return Err(invalid());
        }
        for index in 0..section_count {
            // The validated table range guarantees these additions do not overflow.
            let header = section_offset + index * section_size;
            let section_type = elf_u32(image, header + 4).ok_or_else(invalid)?;
            let data_offset = elf_offset(image, header + 24).ok_or_else(invalid)?;
            let data_size = elf_offset(image, header + 32).ok_or_else(invalid)?;
            // SHT_NOBITS consumes memory but has no corresponding bytes in the file.
            if section_type != 8 && !valid_image_range(image, data_offset, data_size) {
                return Err(invalid());
            }
        }
    }

    let program_offset = elf_offset(image, 32).ok_or_else(invalid)?;
    let program_size = usize::from(elf_u16(image, 54).ok_or_else(invalid)?);
    let program_count = usize::from(elf_u16(image, 56).ok_or_else(invalid)?);
    if program_count == 0xffff {
        return Err(invalid());
    }
    if program_count != 0 {
        let table_size = program_size
            .checked_mul(program_count)
            .ok_or_else(invalid)?;
        if program_offset < 64
            || program_size < 56
            || !valid_image_range(image, program_offset, table_size)
        {
            return Err(invalid());
        }
        for index in 0..program_count {
            let header = program_offset + index * program_size;
            let data_offset = elf_offset(image, header + 8).ok_or_else(invalid)?;
            let data_size = elf_offset(image, header + 32).ok_or_else(invalid)?;
            if !valid_image_range(image, data_offset, data_size) {
                return Err(invalid());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod cache_entry_validation_tests {
    use super::*;

    fn write_bytes(image: &mut [c_char], offset: usize, bytes: &[u8]) {
        for (target, byte) in image[offset..offset + bytes.len()].iter_mut().zip(bytes) {
            *target = *byte as c_char;
        }
    }

    fn bounded_elf_fixture() -> Vec<c_char> {
        // One section, one program header, and four file-backed data bytes.
        // This tests structural bounds only; CUDA validates actual module semantics.
        let mut image = vec![0; 188];
        write_bytes(&mut image, 0, &[0x7f, b'E', b'L', b'F', 2, 1, 1]);
        write_bytes(&mut image, 20, &1u32.to_le_bytes());
        write_bytes(&mut image, 32, &128u64.to_le_bytes());
        write_bytes(&mut image, 40, &64u64.to_le_bytes());
        write_bytes(&mut image, 52, &64u16.to_le_bytes());
        write_bytes(&mut image, 54, &56u16.to_le_bytes());
        write_bytes(&mut image, 56, &1u16.to_le_bytes());
        write_bytes(&mut image, 58, &64u16.to_le_bytes());
        write_bytes(&mut image, 60, &1u16.to_le_bytes());
        write_bytes(&mut image, 68, &1u32.to_le_bytes());
        write_bytes(&mut image, 88, &184u64.to_le_bytes());
        write_bytes(&mut image, 96, &4u64.to_le_bytes());
        write_bytes(&mut image, 136, &184u64.to_le_bytes());
        write_bytes(&mut image, 160, &4u64.to_le_bytes());
        write_bytes(&mut image, 184, &[0, 1, 2, 0xff]);
        image
    }

    #[test]
    fn valid_terminated_ptx_preserves_entrypoint() {
        let ptx = [b'p' as c_char, b't' as c_char, b'x' as c_char, 0];
        assert_eq!(
            validate_cache_entry(&ptx, CacheArtifactFormat::Ptx, "kernel")
                .unwrap()
                .as_bytes(),
            b"kernel"
        );
    }

    #[test]
    fn malformed_cache_strings_are_rejected_before_driver_ffi() {
        assert!(validate_cache_entry(&[], CacheArtifactFormat::Ptx, "kernel").is_err());
        assert!(
            validate_cache_entry(&[b'x' as c_char], CacheArtifactFormat::Ptx, "kernel").is_err()
        );
        assert!(validate_cache_entry(&[0], CacheArtifactFormat::Ptx, "bad\0entry").is_err());
    }

    #[test]
    fn bounded_cubin_does_not_need_ptx_string_termination() {
        let image = bounded_elf_fixture();
        assert_ne!(image.last(), Some(&0));
        assert_eq!(
            validate_cache_entry(&image, CacheArtifactFormat::Cubin, "kernel")
                .unwrap()
                .as_bytes(),
            b"kernel"
        );
        assert!(validate_cache_entry(&image, CacheArtifactFormat::Cubin, "bad\0entry").is_err());
    }

    #[test]
    fn malformed_elf_headers_are_rejected_before_driver_ffi() {
        let original = bounded_elf_fixture();
        assert!(validate_cubin(&original[..63]).is_err());
        for index in [0, 4, 5, 6, 20, 52] {
            let mut image = original.clone();
            image[index] = 0;
            assert!(validate_cubin(&image).is_err(), "header byte {index}");
        }
    }

    #[test]
    fn out_of_bounds_section_table_and_data_are_rejected() {
        let original = bounded_elf_fixture();
        let mut image = original.clone();
        write_bytes(&mut image, 40, &u64::MAX.to_le_bytes());
        assert!(validate_cubin(&image).is_err());
        image = original.clone();
        write_bytes(&mut image, 58, &63u16.to_le_bytes());
        assert!(validate_cubin(&image).is_err());
        image = original.clone();
        write_bytes(&mut image, 88, &187u64.to_le_bytes());
        assert!(validate_cubin(&image).is_err());
        image = original;
        write_bytes(&mut image, 96, &u64::MAX.to_le_bytes());
        assert!(validate_cubin(&image).is_err());
    }

    #[test]
    fn nobits_section_has_no_file_backed_range() {
        let mut image = bounded_elf_fixture();
        write_bytes(&mut image, 68, &8u32.to_le_bytes());
        write_bytes(&mut image, 88, &1000u64.to_le_bytes());
        write_bytes(&mut image, 96, &2000u64.to_le_bytes());
        assert!(validate_cubin(&image).is_ok());
    }

    #[test]
    fn out_of_bounds_program_table_and_data_are_rejected() {
        let original = bounded_elf_fixture();
        let mut image = original.clone();
        write_bytes(&mut image, 32, &u64::MAX.to_le_bytes());
        assert!(validate_cubin(&image).is_err());
        image = original.clone();
        write_bytes(&mut image, 54, &55u16.to_le_bytes());
        assert!(validate_cubin(&image).is_err());
        image = original.clone();
        write_bytes(&mut image, 136, &187u64.to_le_bytes());
        assert!(validate_cubin(&image).is_err());
        image = original;
        write_bytes(&mut image, 160, &u64::MAX.to_le_bytes());
        assert!(validate_cubin(&image).is_err());
    }

    #[test]
    fn extended_elf_counts_are_rejected() {
        let original = bounded_elf_fixture();
        let mut image = original.clone();
        write_bytes(&mut image, 60, &0u16.to_le_bytes());
        assert!(validate_cubin(&image).is_err());
        image = original.clone();
        write_bytes(&mut image, 62, &0xffffu16.to_le_bytes());
        assert!(validate_cubin(&image).is_err());
        image = original;
        write_bytes(&mut image, 56, &0xffffu16.to_le_bytes());
        assert!(validate_cubin(&image).is_err());
    }

    #[test]
    fn feature_selects_an_explicit_separate_cache_format() {
        assert_ne!(
            CacheArtifactFormat::Ptx.directory(),
            CacheArtifactFormat::Cubin.directory()
        );
        #[cfg(feature = "teamy-cubin-cache")]
        assert_eq!(CacheArtifactFormat::selected(), CacheArtifactFormat::Cubin);
        #[cfg(not(feature = "teamy-cubin-cache"))]
        assert_eq!(CacheArtifactFormat::selected(), CacheArtifactFormat::Ptx);
    }

    fn checksum_input(
        key: u128,
        format: CacheArtifactFormat,
        name: &str,
        shared_mem: u64,
        image: &[u8],
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        write_checksum_input(key, format, name, shared_mem, image, |chunk| {
            bytes.extend_from_slice(chunk);
        });
        bytes
    }

    #[test]
    fn every_load_affecting_field_is_bound_to_checksum_input() {
        let original = bounded_elf_fixture();
        let payload: Vec<u8> = original.iter().map(|byte| *byte as u8).collect();
        let baseline = checksum_input(11, CacheArtifactFormat::Cubin, "kernel", 32, &payload);
        assert_ne!(
            baseline,
            checksum_input(12, CacheArtifactFormat::Cubin, "kernel", 32, &payload)
        );
        assert_ne!(
            baseline,
            checksum_input(11, CacheArtifactFormat::Ptx, "kernel", 32, &payload)
        );
        assert_ne!(
            baseline,
            checksum_input(11, CacheArtifactFormat::Cubin, "other", 32, &payload)
        );
        assert_ne!(
            baseline,
            checksum_input(11, CacheArtifactFormat::Cubin, "kernel", 64, &payload)
        );

        let mut flipped = original;
        flipped[184] ^= 1;
        // A payload bitflip can pass ELF bounds validation; it must still change
        // checksum input before the driver sees the cached executable image.
        assert!(validate_cubin(&flipped).is_ok());
        let flipped: Vec<u8> = flipped.iter().map(|byte| *byte as u8).collect();
        assert_ne!(
            baseline,
            checksum_input(11, CacheArtifactFormat::Cubin, "kernel", 32, &flipped)
        );
    }

    #[test]
    fn lengths_preserve_checksum_field_boundaries() {
        let first = checksum_input(11, CacheArtifactFormat::Cubin, "a", 0, b"bc");
        let second = checksum_input(11, CacheArtifactFormat::Cubin, "ab", 0, b"c");
        assert_ne!(first, second);
    }
}
