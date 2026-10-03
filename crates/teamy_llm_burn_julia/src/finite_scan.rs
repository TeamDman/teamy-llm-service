//! Exact IEEE-754 exponent validation for little-endian FP32 checkpoint bytes.
//! This checks data; it never changes model values or numerical precision.

pub(crate) fn all_finite_f32_le(bytes: &[u8]) -> bool {
    if !bytes.len().is_multiple_of(4) {
        return false;
    }
    #[cfg(all(
        target_endian = "little",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime detection precedes the target-feature call. The helper
        // checks bounds before every unaligned vector load and retains tails.
        return unsafe { avx2_all_finite(bytes) };
    }
    scalar_all_finite(bytes)
}

fn scalar_all_finite(bytes: &[u8]) -> bool {
    bytes.chunks_exact(4).all(|value| {
        u32::from_le_bytes(value.try_into().expect("four-byte chunk")) & 0x7f80_0000 != 0x7f80_0000
    })
}

#[cfg(all(
    target_endian = "little",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2")]
unsafe fn avx2_all_finite(bytes: &[u8]) -> bool {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    // SAFETY: the caller verified AVX2. All vector offsets below have at least
    // 32 remaining initialized bytes. loadu has no alignment requirement; no
    // pointer or reference escapes the slice's lifetime.
    unsafe {
        let mask = _mm256_set1_epi32(0x7f80_0000);
        let mut found = [_mm256_setzero_si256(); 4];
        let mut offset = 0;
        while bytes.len() - offset >= 128 {
            for (lane, accumulated) in found.iter_mut().enumerate() {
                let values = _mm256_loadu_si256(bytes.as_ptr().add(offset + lane * 32).cast());
                let nonfinite = _mm256_cmpeq_epi32(_mm256_and_si256(values, mask), mask);
                *accumulated = _mm256_or_si256(*accumulated, nonfinite);
            }
            offset += 128;
        }
        let mut combined = _mm256_or_si256(
            _mm256_or_si256(found[0], found[1]),
            _mm256_or_si256(found[2], found[3]),
        );
        while bytes.len() - offset >= 32 {
            let values = _mm256_loadu_si256(bytes.as_ptr().add(offset).cast());
            combined = _mm256_or_si256(
                combined,
                _mm256_cmpeq_epi32(_mm256_and_si256(values, mask), mask),
            );
            offset += 32;
        }
        _mm256_movemask_epi8(combined) == 0 && scalar_all_finite(&bytes[offset..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(bytes: &[u8]) -> bool {
        bytes.len().is_multiple_of(4)
            && bytes.chunks_exact(4).all(|value| {
                f32::from_le_bytes(value.try_into().expect("FP32 fixture")).is_finite()
            })
    }

    #[test]
    fn finite_ieee_edges_and_incomplete_sequences_match_reference() {
        let bits = [
            0_u32,
            0x8000_0000,
            1,
            0x807f_ffff,
            0x0080_0000,
            0x7f7f_ffff,
            0xff7f_ffff,
            0x3f80_0000,
        ];
        let bytes = bits
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        assert!(all_finite_f32_le(&bytes));
        assert!(all_finite_f32_le(&[]));
        for length in 0..bytes.len() {
            assert_eq!(
                all_finite_f32_le(&bytes[..length]),
                reference(&bytes[..length])
            );
        }
    }

    #[test]
    fn every_vector_lane_tail_and_unaligned_offset_rejects_nan_and_infinity() {
        let nonfinite = [
            0x7f80_0000_u32,
            0xff80_0000,
            0x7fc0_0000,
            0xffc0_0001,
            0x7f80_0001,
        ];
        for prefix in 0..32 {
            for length in [1_usize, 7, 8, 9, 31, 32, 33, 95, 96, 97] {
                let mut storage = vec![0_u8; prefix + length * 4];
                for row in storage[prefix..].chunks_exact_mut(4) {
                    row.copy_from_slice(&1.25_f32.to_le_bytes());
                }
                assert!(all_finite_f32_le(&storage[prefix..]));
                for position in 0..length {
                    let offset = prefix + position * 4;
                    for bits in nonfinite {
                        storage[offset..offset + 4].copy_from_slice(&bits.to_le_bytes());
                        assert!(!all_finite_f32_le(&storage[prefix..]));
                        assert_eq!(
                            scalar_all_finite(&storage[prefix..]),
                            reference(&storage[prefix..])
                        );
                    }
                    storage[offset..offset + 4].copy_from_slice(&1.25_f32.to_le_bytes());
                }
            }
        }
    }
}
