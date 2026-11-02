// Copyright 2026 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! CLI converter from Criterion (`target/criterion`) and nightly `libtest` outputs into Go
//! `benchfmt` streams consumable by `benchstat`.

use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;
use std::process;

use benchfmt::{BenchHeader, ConvertOptions, convert_criterion_dir, convert_libtest_output};

const USAGE: &str = "\
Usage:
  benchfmt criterion [DIR] [--pkg PKG] [--cpu N] [--epochs K] [--baseline NAME] [--tag KEY=VAL]
  benchfmt libtest [FILE]  [--pkg PKG] [--cpu N] [--tag KEY=VAL]

Subcommands:
  criterion   Convert Criterion JSON samples (default DIR: target/criterion) to Go benchfmt.
  libtest     Convert `cargo +nightly bench` output (stdin or FILE) to Go benchfmt.

Options:
  --pkg PKG         Override the `pkg:` metadata header (default: rust).
  --cpu N           Append `-N` CPU suffix to benchmark names (for example `--cpu 2`).
  --epochs K        Aggregate Criterion samples into K epochs (`0` emits all raw samples; default: 6).
  --baseline NAME   Criterion baseline folder to inspect (default: new).
  --tag KEY=VAL     Emit an extra `KEY: VAL` metadata header line for `benchstat -col /KEY`.
";

fn main() {
    if let Err(err) = run() {
        eprintln!("benchfmt: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let Some(subcmd) = args.next() else {
        eprintln!("{USAGE}");
        return Err("missing subcommand".to_owned());
    };
    if subcmd == "-h" || subcmd == "--help" || subcmd == "help" {
        print!("{USAGE}");
        return Ok(());
    }

    let mut opts = ConvertOptions::default();
    let mut positional: Option<String> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(());
            }
            "--pkg" => {
                opts.header.pkg = args.next().ok_or("missing value for --pkg")?;
            }
            "--cpu" => {
                let raw = args.next().ok_or("missing value for --cpu")?;
                let n: usize = raw.parse().map_err(|_| format!("invalid --cpu {raw:?}"))?;
                opts.cpu_procs = Some(n);
            }
            "--epochs" => {
                let raw = args.next().ok_or("missing value for --epochs")?;
                opts.criterion_epochs = raw
                    .parse()
                    .map_err(|_| format!("invalid --epochs {raw:?}"))?;
            }
            "--baseline" => {
                opts.criterion_baseline = args.next().ok_or("missing value for --baseline")?;
            }
            "--tag" => {
                let raw = args.next().ok_or("missing value for --tag")?;
                let (k, v) = raw
                    .split_once('=')
                    .ok_or_else(|| format!("expected KEY=VAL for --tag, got {raw:?}"))?;
                opts.header.extra.push((k.to_owned(), v.to_owned()));
            }
            _ if arg.starts_with('-') => {
                return Err(format!("unknown flag {arg:?}\n\n{USAGE}"));
            }
            _ => {
                if positional.is_some() {
                    return Err(format!("unexpected extra positional argument {arg:?}"));
                }
                positional = Some(arg);
            }
        }
    }

    // Re-detect host metadata while preserving user-provided `pkg` and tags.
    let pkg = opts.header.pkg.clone();
    let extra = std::mem::take(&mut opts.header.extra);
    opts.header = BenchHeader::detect(pkg);
    opts.header.extra = extra;

    let stdout = io::stdout();
    let mut out = stdout.lock();

    match subcmd.as_str() {
        "criterion" => {
            let dir = PathBuf::from(positional.unwrap_or_else(|| "target/criterion".to_owned()));
            convert_criterion_dir(&dir, &opts, &mut out).map_err(|e| e.to_string())?;
        }
        "libtest" => {
            let input = match positional {
                Some(path) if path != "-" => fs::read_to_string(path).map_err(|e| e.to_string())?,
                _ => {
                    let mut buf = String::new();
                    io::stdin()
                        .read_to_string(&mut buf)
                        .map_err(|e| e.to_string())?;
                    buf
                }
            };
            convert_libtest_output(&input, &opts, &mut out).map_err(|e| e.to_string())?;
        }
        other => {
            return Err(format!("unknown subcommand {other:?}\n\n{USAGE}"));
        }
    }

    Ok(())
}
