//! `synctl` utility: `info`, `ls`, `verify` for neo-store files (compressed tensors in `ls` and
//! `info` are marked with encoding and size in the file).

use std::process::ExitCode;

use neo_store::{
    NirImportOptions, NirJsonOptions, Store, Unsupported, WriteOptions, nir_json_file_to_syn,
};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("from-nir") {
        return from_nir(&args[1..]);
    }
    let (cmd, path) = match (args.first(), args.get(1)) {
        (Some(c), Some(p)) => (c.as_str(), p.as_str()),
        _ => {
            eprintln!(
                "usage: synctl <info|ls|verify> <file.syn>\n synctl nir <file.syn> <output.json> [--compact] [--lossy] [--passthrough]\n synctl from-nir <input.json> <output.syn> [--passthrough] [--f64]"
            );
            return ExitCode::from(2);
        }
    };
    let store = match Store::open(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to open {path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    match cmd {
        "info" => {
            print!("{}", store.summary());
            match store.validate_graph() {
                Ok(Some(shapes)) => {
                    println!("  the graph is consistent with the weights:");
                    for (name, s) in &shapes.nodes {
                        println!("    {name}: {:?} → {:?}", s.input, s.output);
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    eprintln!("the graph is not consistent with the weights: {e}");
                    return ExitCode::FAILURE;
                }
            }
            ExitCode::SUCCESS
        }
        "ls" => {
            for t in store.tensors() {
                if t.is_compressed() {
                    println!(
                        "{}\t{}\t{:?}\t{} → {} B ({:.0}%)",
                        t.name,
                        t.dtype,
                        t.shape,
                        t.encoding.name(),
                        t.stored_bytes,
                        t.stored_ratio() * 100.0
                    );
                } else {
                    println!("{}\t{}\t{:?}", t.name, t.dtype, t.shape);
                }
            }
            ExitCode::SUCCESS
        }
        "verify" => match store.verify() {
            Ok(r) if r.is_ok() => {
                println!("OK: {} tensors, {} B", r.tensors_checked, r.bytes_checked);
                ExitCode::SUCCESS
            }
            Ok(r) => {
                for f in &r.failures {
                    match &f.detail {
                        Some(d) => eprintln!("DAMAGED {}: {d}", f.name),
                        None => eprintln!(
                            "CORRUPT {}: written {:#010x}, calculated {:#010x}",
                            f.name, f.expected, f.actual
                        ),
                    }
                }
                ExitCode::FAILURE
            }
            Err(e) => {
                eprintln!("validation error: {e}");
                ExitCode::FAILURE
            }
        },
        "nir" => {
            let Some(out) = args.get(2) else {
                eprintln!("specify the output file: synctl nir <file.syn> <output.json>");
                return ExitCode::from(2);
            };
            let flags = &args[3..];
            let opts = NirJsonOptions {
                pretty: !flags.iter().any(|f| f == "--compact"),
                lossy: flags.iter().any(|f| f == "--lossy"),
                unsupported: if flags.iter().any(|f| f == "--passthrough") {
                    Unsupported::Passthrough
                } else {
                    Unsupported::Error
                },
                ..NirJsonOptions::default()
            };
            match store.write_nir_json(out, &opts) {
                Ok(warnings) => {
                    for w in &warnings {
                        eprintln!("warning: {w}");
                    }
                    println!("written: {out}");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("export to NIR JSON failed: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        other => {
            eprintln!("unknown command '{other}'");
            ExitCode::from(2)
        }
    }
}

/// `synctl from-nir <input.json> <output.syn> [--passthrough] [--f64]`.
fn from_nir(args: &[String]) -> ExitCode {
    let (Some(src), Some(dst)) = (args.first(), args.get(1)) else {
        eprintln!("usage: synctl from-nir <input.json> <output.syn> [--passthrough] [--f64]");
        return ExitCode::from(2);
    };
    let flags = &args[2..];
    let mut opts = NirImportOptions::default();
    if flags.iter().any(|f| f == "--passthrough") {
        opts.unknown = Unsupported::Passthrough;
    }
    if flags.iter().any(|f| f == "--f64") {
        opts.dtype = neo_store::DType::F64;
    }
    match nir_json_file_to_syn(src, dst, WriteOptions::default(), &opts) {
        Ok((summary, warnings)) => {
            for w in &warnings {
                eprintln!("warning: {w}");
            }
            println!("written: {dst} ({} tensors)", summary.tensor_count);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("import from NIR JSON failed: {e}");
            ExitCode::FAILURE
        }
    }
}
