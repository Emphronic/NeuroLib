//! Packing and unpacking N-bit integers (type [`DType::Bits`](crate::DType::Bits)).
//!
//! The layout is described in [`DType::Bits`](crate::DType::Bits): element `i` occupies bits
//! `[i·n, (i+1)·n)` of the stream, bit 0 of each byte is the least significant. The scale and displacement of such
//! tensors lie as separate tensors, so here there are only whole codes `0..2ⁿ`.

use crate::error::{StoreError, StoreResult};

fn check_bits(bits: u8) -> StoreResult<()> {
    if (1..=7).contains(&bits) {
        Ok(())
    } else {
        Err(StoreError::Invalid(format!(
            "the number of bits {bits} must be from 1 to 7"
        )))
    }
}

fn packed_len(numel: usize, bits: u8) -> StoreResult<usize> {
    numel
        .checked_mul(bits as usize)
        .map(|b| b.div_ceil(8))
        .ok_or_else(|| StoreError::Invalid("number of elements overflows usize".into()))
}

/// Packs `0..2^bits` codes (one per byte) into a `DType::Bits(bits)` bit stream.
pub fn pack_bits(codes: &[u8], bits: u8) -> StoreResult<Vec<u8>> {
    check_bits(bits)?;
    let max = (1u16 << bits) - 1;
    let mut out = vec![0u8; packed_len(codes.len(), bits)?];
    for (i, &c) in codes.iter().enumerate() {
        if c as u16 > max {
            return Err(StoreError::Invalid(format!(
                "the code {c} at position {i} does not fit into the {bits} bit"
            )));
        }
        let bit = i * bits as usize;
        let byte = bit / 8;
        let shift = bit % 8;
        out[byte] |= c << shift;
        if shift + bits as usize > 8 {
            out[byte + 1] |= c >> (8 - shift);
        }
    }
    Ok(out)
}

/// Unpacks `numel` codes from the `DType::Bits(bits)` bit stream (one per byte).
pub fn unpack_bits(bytes: &[u8], bits: u8, numel: usize) -> StoreResult<Vec<u8>> {
    check_bits(bits)?;
    let need = packed_len(numel, bits)?;
    if bytes.len() < need {
        return Err(StoreError::Invalid(format!(
            "for {numel} elements {bits} bits are needed {need} B, transferred {}",
            bytes.len()
        )));
    }
    let mask = ((1u16 << bits) - 1) as u8;
    let mut out = Vec::with_capacity(numel);
    for i in 0..numel {
        let bit = i * bits as usize;
        let byte = bit / 8;
        let shift = bit % 8;
        let mut v = (bytes[byte] as u16) >> shift;
        if shift + bits as usize > 8 {
            v |= (bytes[byte + 1] as u16) << (8 - shift);
        }
        out.push((v as u8) & mask);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_layout() {
        // 2 bits: elements go from the least significant bits of the byte to the most significant ones
        assert_eq!(
            unpack_bits(&[0b1001_1000, 0b01], 2, 5).unwrap(),
            vec![0, 2, 1, 2, 1]
        );
        assert_eq!(
            pack_bits(&[0, 2, 1, 2, 1], 2).unwrap(),
            vec![0b1001_1000, 0b01]
        );
        // 1 bit
        assert_eq!(unpack_bits(&[0b0000_0101], 1, 3).unwrap(), vec![1, 0, 1]);
    }

    #[test]
    fn roundtrip_every_width_and_length() {
        for bits in 1..=7u8 {
            let max = (1u16 << bits) as usize;
            for n in 0..40usize {
                let codes: Vec<u8> = (0..n).map(|i| ((i * 5 + 3) % max) as u8).collect();
                let packed = pack_bits(&codes, bits).unwrap();
                assert_eq!(packed.len(), (n * bits as usize).div_ceil(8));
                assert_eq!(
                    unpack_bits(&packed, bits, n).unwrap(),
                    codes,
                    "bits={bits} n={n}"
                );
            }
        }
    }

    #[test]
    fn rejects_bad_input() {
        assert!(pack_bits(&[4], 2).is_err());
        assert!(pack_bits(&[0], 0).is_err());
        assert!(pack_bits(&[0], 8).is_err());
        assert!(unpack_bits(&[0], 2, 5).is_err());
    }
}
