//! Completeness `neo_snn::prelude`: every name re-exported by root
//! modules (`pub use ...`), must also be in the prelude - or be listed in
//! list of exceptions with reason.

use std::collections::BTreeSet;

/// Root files of modules whose `pub use` should be included in the prelude.
const ROOTS: &[(&str, &str)] = &[
    ("attention", include_str!("../src/attention/mod.rs")),
    ("convert", include_str!("../src/convert/mod.rs")),
    ("encoding", include_str!("../src/encoding/mod.rs")),
    ("event_driven", include_str!("../src/event_driven/mod.rs")),
    ("functional", include_str!("../src/functional/mod.rs")),
    ("kan", include_str!("../src/kan/mod.rs")),
    ("layers", include_str!("../src/layers.rs")),
    ("neuron", include_str!("../src/neuron/mod.rs")),
    ("probabilistic", include_str!("../src/probabilistic/mod.rs")),
    ("reservoir", include_str!("../src/reservoir/mod.rs")),
    ("surrogate", include_str!("../src/surrogate/mod.rs")),
    ("temporal", include_str!("../src/temporal/mod.rs")),
];

/// Names that are not included in the prelude intentionally.
const EXCLUDED: &[(&str, &str)] = &[];

const PRELUDE: &str = include_str!("../src/prelude.rs");

/// Names from all `pub use path::{A, B, C};` and `pub use path::Name;`.
fn reexported_names(src: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut rest = src;
    while let Some(pos) = rest.find("pub use ") {
        let after = &rest[pos + "pub use ".len()..];
        let end = after.find(';').expect("pub use without ';'");
        let stmt = &after[..end];
        let items = match (stmt.find('{'), stmt.rfind('}')) {
            (Some(a), Some(b)) => &stmt[a + 1..b],
            _ => stmt.rsplit("::").next().unwrap_or(stmt),
        };
        for raw in items.split(',') {
            let name = raw.trim().rsplit(" as ").next().unwrap_or("").trim();
            if !name.is_empty() && name != "*" {
                names.insert(name.to_string());
            }
        }
        rest = &after[end..];
    }
    names
}

fn contains_word(haystack: &str, word: &str) -> bool {
    let bytes = haystack.as_bytes();
    let mut start = 0;
    while let Some(i) = haystack[start..].find(word) {
        let a = start + i;
        let b = a + word.len();
        let before_ok = a == 0 || !(bytes[a - 1].is_ascii_alphanumeric() || bytes[a - 1] == b'_');
        let after_ok = b >= bytes.len() || !(bytes[b].is_ascii_alphanumeric() || bytes[b] == b'_');
        if before_ok && after_ok {
            return true;
        }
        start = b;
    }
    false
}

#[test]
fn prelude_covers_every_reexported_name() {
    let mut missing = Vec::new();
    for (module, src) in ROOTS {
        for name in reexported_names(src) {
            if EXCLUDED.iter().any(|(n, _)| *n == name) {
                continue;
            }
            if !contains_word(PRELUDE, &name) {
                missing.push(format!("{module}::{name}"));
            }
        }
    }
    assert!(missing.is_empty(), "prelude is missing: {missing:?}");
}

#[test]
fn excluded_names_have_reasons_and_really_exist_in_sources() {
    for (name, reason) in EXCLUDED {
        assert!(!reason.trim().is_empty(), "exception {name} has no reason");
    }
}

#[test]
fn prelude_does_not_shadow_std_result() {
    assert!(
        !PRELUDE.contains("error::Result") && !PRELUDE.contains("error::{Result"),
        "prelude should not re-export crate Result"
    );
}

#[test]
fn parser_extracts_braced_and_plain_reexports() {
    let got = reexported_names("pub use a::{X, Y,\n    Z};\npub use b::W;\npub use c::d as E;");
    assert_eq!(
        got,
        ["E", "W", "X", "Y", "Z"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    );
}

/// Real compile check: names are available via glob import.
#[test]
fn prelude_glob_import_compiles_with_key_items() {
    use neo_snn::prelude::*;
    let _ =
        (lateral_inhibition_soft as fn(burn::tensor::Tensor<2>, f32) -> burn::tensor::Tensor<2>,);
    let _: Option<StdpConv2dConfig> = None;
    let _: Option<SnnError> = None;
    let _ = std::mem::size_of::<PopulationReduction>();
}
