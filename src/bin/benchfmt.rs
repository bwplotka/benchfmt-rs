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
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process;

use benchfmt::{ConvertOptions, convert_criterion_dir, convert_libtest_output, validate_header_kv};

const USAGE: &str = "\
Usage:
  benchfmt criterion [DIR] [--pkg PKG] [--cpu N] [--epochs K] [--baseline NAME] [--tag key=val]
  benchfmt libtest [FILE]  [--pkg PKG] [--cpu N] [--tag key=val]

Subcommands:
  criterion   Convert Criterion JSON samples (default DIR: target/criterion) to Go benchfmt.
  libtest     Convert `cargo +nightly bench` output (stdin, `-`, or FILE) to Go benchfmt.

Options:
  --pkg PKG         Override the `pkg:` metadata header (default: rust).
  --cpu N           Append `-N` CPU suffix to benchmark names (for example `--cpu 2`).
  --epochs K        Aggregate Criterion samples into K epochs (`0` emits all raw samples; default: 6).
  --baseline NAME   Criterion baseline folder to inspect (default: new).
  --tag key=val     Emit an extra `key: val` metadata header line for `benchstat -col key`.
";

fn main() {
    let stdout = io::stdout();
    if let Err(err) = run_with(std::env::args().skip(1), io::stdin().lock(), stdout.lock()) {
        eprintln!("benchfmt: {err}");
        process::exit(1);
    }
}

fn take_opt<I, S>(arg: &str, flag: &str, iter: &mut I) -> Result<Option<String>, String>
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    if arg == flag {
        let val = iter
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        return Ok(Some(val.as_ref().to_owned()));
    }
    if let Some(rest) = arg.strip_prefix(flag).and_then(|s| s.strip_prefix('=')) {
        return Ok(Some(rest.to_owned()));
    }
    Ok(None)
}

fn run_with<I, S, R, W>(args: I, mut stdin: R, mut out: W) -> Result<(), String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
    R: Read,
    W: Write,
{
    let mut args = args.into_iter();
    let Some(subcmd_raw) = args.next() else {
        return Err(format!("missing subcommand\n\n{USAGE}"));
    };
    let subcmd = subcmd_raw.as_ref();
    if matches!(subcmd, "-h" | "--help" | "help") {
        write!(out, "{USAGE}").map_err(|e| e.to_string())?;
        return Ok(());
    }

    let mut opts = ConvertOptions::default();
    let mut positional: Option<String> = None;

    while let Some(raw_arg) = args.next() {
        let arg = raw_arg.as_ref();
        if matches!(arg, "-h" | "--help") {
            write!(out, "{USAGE}").map_err(|e| e.to_string())?;
            return Ok(());
        }
        if let Some(pkg) = take_opt(arg, "--pkg", &mut args)? {
            validate_header_kv("pkg", &pkg)?;
            opts.header.pkg = pkg;
            continue;
        }
        if let Some(raw) = take_opt(arg, "--cpu", &mut args)? {
            let n: usize = raw.parse().map_err(|_| format!("invalid --cpu {raw:?}"))?;
            opts.cpu_procs = Some(n);
            continue;
        }
        if let Some(raw) = take_opt(arg, "--epochs", &mut args)? {
            opts.criterion_epochs = raw
                .parse()
                .map_err(|_| format!("invalid --epochs {raw:?}"))?;
            continue;
        }
        if let Some(base) = take_opt(arg, "--baseline", &mut args)? {
            opts.criterion_baseline = base;
            continue;
        }
        if let Some(raw) = take_opt(arg, "--tag", &mut args)? {
            let (k, v) = raw
                .split_once('=')
                .ok_or_else(|| format!("expected key=val for --tag, got {raw:?}"))?;
            validate_header_kv(k, v)?;
            opts.header.extra.push((k.to_owned(), v.to_owned()));
            continue;
        }
        if arg.starts_with('-') && arg != "-" {
            return Err(format!("unknown flag {arg:?}\n\n{USAGE}"));
        }
        if positional.is_some() {
            return Err(format!("unexpected extra positional argument {arg:?}"));
        }
        positional = Some(arg.to_owned());
    }

    match subcmd {
        "criterion" => {
            let dir = PathBuf::from(positional.unwrap_or_else(|| "target/criterion".to_owned()));
            convert_criterion_dir(&dir, &opts, &mut out).map_err(|e| e.to_string())?;
        }
        "libtest" => {
            let input = match positional {
                Some(path) if path != "-" => fs::read_to_string(path).map_err(|e| e.to_string())?,
                _ => {
                    let mut buf = String::new();
                    stdin.read_to_string(&mut buf).map_err(|e| e.to_string())?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_libtest_stdin_dash_and_equals_flags_work() {
        let input = "test decode::fast ... bench: 42.00 ns/iter (+/- 1.00) = 100.00 MB/s\n";
        let mut out = Vec::new();
        run_with(
            [
                "libtest",
                "-",
                "--pkg=my_crate",
                "--cpu=4",
                "--tag=impl=rust",
            ],
            input.as_bytes(),
            &mut out,
        )
        .unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("pkg: my_crate\n"), "{text}");
        assert!(text.contains("impl: rust\n"), "{text}");
        assert!(text.contains("BenchmarkDecode::fast-4\t"), "{text}");
    }

    #[test]
    fn cli_help_and_error_paths_are_validated() {
        let mut help_out = Vec::new();
        run_with(["--help"], &b""[..], &mut help_out).unwrap();
        let help_text = String::from_utf8(help_out).unwrap();
        assert!(help_text.contains("benchstat -col key"), "{help_text}");

        // Rejects uppercase `--tag` keys that `benchfmt.Reader` would silently ignore.
        assert!(run_with(["libtest", "--tag", "IMPL=rust"], &b""[..], Vec::new()).is_err());
        assert!(run_with(["unknown"], &b""[..], Vec::new()).is_err());
        assert!(run_with(["libtest", "--bogus"], &b""[..], Vec::new()).is_err());
        assert!(run_with(["libtest", "a", "b"], &b""[..], Vec::new()).is_err());
        assert!(run_with::<[&str; 0], _, _, _>([], &b""[..], Vec::new()).is_err());

        let tmp = std::env::temp_dir().join(format!("benchfmt_cli_test_{}", std::process::id()));
        let base_dir = tmp.join("Grp/base");
        fs::create_dir_all(&base_dir).unwrap();
        fs::write(base_dir.join("benchmark.json"), r#"{"full_id":"Grp/k=1"}"#).unwrap();
        fs::write(
            base_dir.join("sample.json"),
            r#"{"iters":[1.0,2.0],"times":[10.0,20.0]}"#,
        )
        .unwrap();
        let mut crit_out = Vec::new();
        run_with(
            [
                "criterion",
                tmp.to_str().unwrap(),
                "--epochs=1",
                "--baseline=base",
            ],
            &b""[..],
            &mut crit_out,
        )
        .unwrap();
        let _ = fs::remove_dir_all(&tmp);
        let crit_text = String::from_utf8(crit_out).unwrap();
        assert!(crit_text.contains("BenchmarkGrp/k=1\t"), "{crit_text}");
        assert!(run_with(["criterion", tmp.to_str().unwrap()], &b""[..], Vec::new()).is_err());
    }
}
