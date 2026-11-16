//! CRC-32C (Castagnoli) and FNV-1a-64 without external dependencies.
//!
//! CRC-32C is implemented using “slicing-by-8”, tables are built at the compilation stage.

const POLY: u32 = 0x82F6_3B78;

const fn make_tables() -> [[u32; 256]; 8] {
    let mut t = [[0u32; 256]; 8];
    let mut i = 0usize;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { (c >> 1) ^ POLY } else { c >> 1 };
            k += 1;
        }
        t[0][i] = c;
        i += 1;
    }
    let mut i = 0usize;
    while i < 256 {
        let mut c = t[0][i];
        let mut j = 1usize;
        while j < 8 {
            c = t[0][(c & 0xff) as usize] ^ (c >> 8);
            t[j][i] = c;
            j += 1;
        }
        i += 1;
    }
    t
}

static TABLES: [[u32; 256]; 8] = make_tables();

/// Incremental CRC-32C.
#[derive(Clone, Debug)]
pub struct Crc32c {
    state: u32,
}

impl Default for Crc32c {
    fn default() -> Self {
        Self::new()
    }
}

impl Crc32c {
    /// New counter.
    pub fn new() -> Self {
        Self { state: !0 }
    }

    /// Adds data.
    pub fn update(&mut self, data: &[u8]) {
        let mut crc = self.state;
        let (chunks, remainder) = data.as_chunks::<8>();
        for ch in chunks {
            let lo = u32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]) ^ crc;
            let hi = u32::from_le_bytes([ch[4], ch[5], ch[6], ch[7]]);
            crc = TABLES[7][(lo & 0xff) as usize]
                ^ TABLES[6][((lo >> 8) & 0xff) as usize]
                ^ TABLES[5][((lo >> 16) & 0xff) as usize]
                ^ TABLES[4][(lo >> 24) as usize]
                ^ TABLES[3][(hi & 0xff) as usize]
                ^ TABLES[2][((hi >> 8) & 0xff) as usize]
                ^ TABLES[1][((hi >> 16) & 0xff) as usize]
                ^ TABLES[0][(hi >> 24) as usize];
        }
        for &b in remainder {
            crc = TABLES[0][((crc ^ b as u32) & 0xff) as usize] ^ (crc >> 8);
        }
        self.state = crc;
    }

    /// Total value (the counter can be continued).
    pub fn finish(&self) -> u32 {
        !self.state
    }
}

/// CRC-32C with one call.
pub fn crc32c(data: &[u8]) -> u32 {
    let mut c = Crc32c::new();
    c.update(data);
    c.finish()
}

/// FNV-1a, 64 bit. Used to fingerprint a set of tensors (not cryptographic).
#[derive(Clone, Debug)]
pub struct Fnv64(u64);

impl Default for Fnv64 {
    fn default() -> Self {
        Self::new()
    }
}

impl Fnv64 {
    /// New hash.
    pub fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    /// Adds data.
    pub fn write(&mut self, data: &[u8]) {
        for &b in data {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    /// Bottom line.
    pub fn finish(&self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vectors() {
        assert_eq!(crc32c(b""), 0);
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
        assert_eq!(crc32c(&[0u8; 32]), 0x8A91_36AA);
    }

    #[test]
    fn incremental_equals_one_shot() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 31 % 251) as u8).collect();
        let mut c = Crc32c::new();
        for part in data.chunks(7) {
            c.update(part);
        }
        assert_eq!(c.finish(), crc32c(&data));
    }

    #[test]
    fn fnv_known() {
        let mut h = Fnv64::new();
        h.write(b"a");
        assert_eq!(h.finish(), 0xaf63_dc4c_8601_ec8c);
    }
}
