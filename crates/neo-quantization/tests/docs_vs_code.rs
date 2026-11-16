//! Documentation vs code: list of supported formats derived from
//! the code itself (`pack::gguf_backend`), and comments are checked for
//! consistency with it.
//!
//! What exactly breaks this test:
//! * format is implemented, but the documentation says “not implemented”
//!   (this happened with IQ2/IQ3 in the `lib.rs` header);
//! * the format is stated in the documentation of the `PackedScheme::Gguf` scheme, but in reality
//!   not implemented;
//! * the implemented format is not mentioned in the scheme documentation;
//! * `gguf_backend` is at odds with what `quantize()` actually does.

use std::panic::{catch_unwind, AssertUnwindSafe};

use neo_quantization::pack::{gguf_backend, GgufBackend};
use neo_quantization::{dequantize, quantize, DType, PackedScheme, QuantizeConfig};

const LIB_RS: &str = include_str!("../src/lib.rs");
const SCHEME_RS: &str = include_str!("../src/scheme.rs");
const FILES: &[(&str, &str)] = &[
    ("lib.rs", LIB_RS),
    ("scheme.rs", SCHEME_RS),
    ("quantize.rs", include_str!("../src/quantize.rs")),
    ("pack/mod.rs", include_str!("../src/pack/mod.rs")),
    ("pack/iq.rs", include_str!("../src/pack/iq.rs")),
    ("pack/kquant.rs", include_str!("../src/pack/kquant.rs")),
    ("pack/gguf.rs", include_str!("../src/pack/gguf.rs")),
    ("pack/tq.rs", include_str!("../src/pack/tq.rs")),
];

/// Phrases used by the documentation to declare the format unimplemented.
const NEGATIVE: &[&str] = &[
    "not implemented",
    "not implemented",
    "not implemented",
    "I won't support",
    "not implemented",
    "unimplemented",
];

fn name(d: DType) -> String {
    format!("{d:?}")
}

fn supported() -> Vec<DType> {
    DType::ALL
        .iter()
        .copied()
        .filter(|&d| gguf_backend(d).is_some())
        .collect()
}

fn gauss(n: usize) -> Vec<f32> {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    (0..n)
        .map(|_| {
            let mut acc = 0.0f32;
            for _ in 0..6 {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                acc += ((s >> 40) as f32) / ((1u64 << 24) as f32);
            }
            acc - 3.0
        })
        .collect()
}

fn gguf_config(d: DType) -> QuantizeConfig {
    let mut cfg = QuantizeConfig::float();
    cfg.scheme = PackedScheme::Gguf(d);
    cfg
}

/// The entire word (not a substring of another identifier: `Q4_0` ≠ `Q4_0X`).
fn has_word(text: &str, word: &str) -> bool {
    let b = text.as_bytes();
    let mut from = 0;
    while let Some(i) = text[from..].find(word) {
        let (s, e) = (from + i, from + i + word.len());
        let ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
        if (s == 0 || !ident(b[s - 1])) && (e >= b.len() || !ident(b[e])) {
            return true;
        }
        from = e;
    }
    false
}

/// Text of the doc-commentary to the `Gguf(DType)` variant in `PackedScheme`.
fn scheme_gguf_doc() -> String {
    let end = SCHEME_RS
        .find("    Gguf(DType),")
        .expect("there is no Gguf(DType) option in scheme.rs");
    let start = SCHEME_RS[..end]
        .rfind("    /// GGUF-compatible scheme")
        .expect("no Gguf doc block");
    SCHEME_RS[start..end].to_string()
}

#[test]
fn every_quantized_dtype_has_a_backend_or_is_listed_as_unsupported() {
    // Quantized types without backend - conscious list; if it is not empty,
    // it needs to be listed explicitly, and not lost silently.
    let unsupported: Vec<String> = DType::ALL
        .iter()
        .copied()
        .filter(|d| d.is_quantized() && gguf_backend(*d).is_none())
        .map(name)
        .collect();
    assert!(
        unsupported.is_empty(),
        "quantized DType without wrapper: {unsupported:?} - implement or add to test exception list"
    );
}

#[test]
fn backend_registry_matches_what_quantize_actually_does() {
    let data = gauss(1024);
    for &d in DType::ALL.iter() {
        let cfg = gguf_config(d);
        let result = catch_unwind(AssertUnwindSafe(|| quantize(&data, &cfg, 0)));
        match gguf_backend(d) {
            Some(_) => {
                let packed = result.unwrap_or_else(|_| {
                    panic!("quantize({d:?}) panics even though gguf_backend states it")
                });
                let back = dequantize(&packed);
                assert!(
                    back.len() >= data.len(),
                    "{d:?}: dequantize returned {} values from {}",
                    back.len(),
                    data.len()
                );
                assert!(
                    back.iter().all(|v| v.is_finite()),
                    "{d:?}: non-finite values after round-trip"
                );
            }
            None => assert!(
                result.is_err(),
                "{d:?}: gguf_backend says “not supported”, but quantize worked"
            ),
        }
    }
}

#[test]
fn registry_families_are_disjoint_and_complete() {
    let count = |b: GgufBackend| {
        DType::ALL
            .iter()
            .filter(|&&d| gguf_backend(d) == Some(b))
            .count()
    };
    assert_eq!(
        count(GgufBackend::Simple),
        6,
        "Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q8_1"
    );
    assert_eq!(count(GgufBackend::KQuant), 6, "Q2_K..Q8_K");
    assert_eq!(
        count(GgufBackend::Iq),
        8,
        "IQ1_S IQ1_M IQ2_XXS IQ2_XS IQ3_XXS IQ3_S IQ4_NL IQ4_XS"
    );
    assert_eq!(count(GgufBackend::Tq), 2, "TQ1_0 TQ2_0");
}

#[test]
fn docs_never_call_a_supported_format_unimplemented() {
    let names: Vec<String> = supported().into_iter().map(name).collect();
    let mut violations = Vec::new();
    for (file, text) in FILES {
        let lines: Vec<&str> = text.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            let t = line.trim_start();
            // Comments only: lines of code and panic format strings are not documentation.
            if !(t.starts_with("//") || t.starts_with("///") || t.starts_with("//!")) {
                continue;
            }
            let lower = line.to_lowercase();
            if !NEGATIVE.iter().any(|n| lower.contains(n)) {
                continue;
            }
            // "...not implemented in this file" is an accurate statement of boundaries
            // module (for example, K-quants live in pack::kquant, not in
            // pack::gguf), and not that the format is not supported by the crate.
            if lower.contains("in this file") || lower.contains("in this file") {
                continue;
            }
            // Window ±2 lines: the list of formats in the frame is moved.
            let lo = i.saturating_sub(2);
            let hi = (i + 2).min(lines.len() - 1);
            let window = lines[lo..=hi].join("\n");
            for n in &names {
                if has_word(&window, n) {
                    violations.push(format!("{file}:{}: “{}” next to {n}", i + 1, line.trim()));
                }
            }
        }
    }
    assert!(
        violations.is_empty(),
        "documentation contradicts code:\n{}",
        violations.join("\n")
    );
}

#[test]
fn scheme_doc_lists_exactly_the_supported_formats() {
    let doc = scheme_gguf_doc();
    let mut missing = Vec::new();
    let mut phantom = Vec::new();
    for &d in DType::ALL.iter().filter(|d| d.is_quantized()) {
        let n = name(d);
        let mentioned = has_word(&doc, &n);
        let implemented = gguf_backend(d).is_some();
        if implemented && !mentioned {
            missing.push(n.clone());
        }
        if !implemented && mentioned {
            phantom.push(n);
        }
    }
    assert!(
        missing.is_empty(),
        "implemented but not mentioned in the doc PackedScheme::Gguf: {missing:?}"
    );
    assert!(
        phantom.is_empty(),
        "mentioned in doc PackedScheme::Gguf, but not implemented: {phantom:?}"
    );
}

#[test]
fn lib_header_does_not_reference_missing_files() {
    // Links to documentation files (`implementation.md`, `ADDED.md`, etc.) must
    // point to existing files near the crate or are absent.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for (file, text) in FILES {
        for word in text.split(|c: char| c.is_whitespace() || "`()[],;:'\"".contains(c)) {
            if word.ends_with(".md")
                && !word.contains('/')
                && word.chars().next().is_some_and(|c| c.is_ascii_uppercase())
            {
                assert!(
                    root.join(word).exists() || root.join("..").join("..").join(word).exists(),
                    "{file}: link to non-existent file {word}"
                );
            }
        }
    }
}

/// The round trip of each implemented format must bring the data closer:
/// relative error (MSE/variance) is not higher than a fixed threshold.
/// This is exactly the test that would catch IQ2_XS, which has quantization and
/// dequantization diverged with block size (74 bytes) and dequantization
/// crashed on any non-empty block.
///
/// Thresholds are REGRESSIONAL (≈1.2× from measured on Gaussian data without
/// imatrix), not a quality statement: IQ2_XXS/IQ2_XS/IQ3_XXS has an error without
/// importance scales are high (0.6–0.7), this should be checked against the reference
/// llama.cpp separately; the test only prevents its from growing unnoticed.
#[test]
fn every_supported_format_round_trips_with_bounded_error() {
    let data = gauss(2048);
    let var: f64 = data.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / data.len() as f64;
    let bound = |d: DType| -> f64 {
        match d {
            DType::IQ2_XXS | DType::IQ2_XS => 0.85,
            DType::IQ3_XXS => 0.75,
            DType::TQ1_0 | DType::TQ2_0 => 0.75,
            DType::IQ1_S => 0.5,
            DType::IQ1_M => 0.42,
            DType::Q2_K => 0.09,
            DType::IQ3_S => 0.04,
            DType::Q3_K => 0.03,
            _ => 0.01,
        }
    };
    for d in supported() {
        let packed = quantize(&data, &gguf_config(d), 0);
        let back = dequantize(&packed);
        let mse: f64 = data
            .iter()
            .zip(&back)
            .map(|(&a, &b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            / data.len() as f64;
        let rel = mse / var;
        assert!(
            rel < bound(d),
            "{d:?}: relative MSE {rel:.4} above threshold {}",
            bound(d)
        );
    }
}
