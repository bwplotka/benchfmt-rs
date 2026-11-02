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

//! Go [Benchmark Data Format (`benchfmt`)](https://go.googlesource.com/proposal/+/master/design/14313-benchmark-format.md)
//! harness, heap allocation tracker, and Criterion/`libtest` converter for Rust.
//!
//! Designed for apples-to-apples performance regression testing (`benchstat old.txt new.txt`)
//! and cross-language Go vs. Rust benchmark comparisons on the same machine, reporting
//! `ns/op`, decimal `MB/s`, `B/op`, and `allocs/op`.
//!
//! # Example (`harness = false` benchmark binary)
//!
//! ```no_run
//! use std::hint::black_box;
//! use benchfmt::{CountingAlloc, Suite};
//!
//! #[global_allocator]
//! static ALLOC: CountingAlloc = CountingAlloc::new();
//!
//! fn main() {
//!     let payload = vec![b'x'; 4096];
//!     Suite::from_env("my_crate")
//!         .with_alloc(&ALLOC)
//!         .bench("ScanBytes/impl=rust", |b| {
//!             b.set_bytes(payload.len() as u64);
//!             b.report_allocs();
//!             while b.keep_running() {
//!                 black_box(payload.iter().filter(|&&c| c == b'x').count());
//!             }
//!         })
//!         .run();
//! }
//! ```

mod alloc;
mod convert;
mod json;

use std::fs;
use std::hint::black_box;
use std::io::{self, Write};
use std::thread;
use std::time::{Duration, Instant};

pub use alloc::{AllocSnapshot, AllocTracker, CountingAlloc};
pub use convert::{ConvertOptions, convert_criterion_dir, convert_libtest_output};

const MAX_ITERATIONS: u64 = 1_000_000_000;

/// Target benchmark duration or fixed iteration count (`-benchtime`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BenchTime {
    /// Calibrate iteration count until a single run takes at least this [`Duration`].
    Duration(Duration),
    /// Run an exact number of iterations (`-benchtime=1000x`).
    Iterations(u64),
}

impl Default for BenchTime {
    fn default() -> Self {
        Self::Duration(Duration::from_secs(1))
    }
}

impl BenchTime {
    /// Parses a Go `-benchtime` argument such as `"2s"`, `"500ms"`, `"100us"`, or `"1000x"`.
    pub fn parse(raw: &str) -> Result<Self, String> {
        let trimmed = raw.trim();
        if let Some(iters) = trimmed.strip_suffix('x') {
            let n = iters
                .parse::<u64>()
                .map_err(|_| format!("invalid iteration benchtime {raw:?}"))?;
            if n == 0 {
                return Err("iteration benchtime must be >= 1".to_owned());
            }
            return Ok(Self::Iterations(n));
        }
        let dur = parse_duration(trimmed)?;
        Ok(Self::Duration(dur))
    }
}

fn parse_duration(raw: &str) -> Result<Duration, String> {
    if let Some(ms) = raw.strip_suffix("ms") {
        let v: f64 = ms
            .parse()
            .map_err(|_| format!("invalid duration {raw:?}"))?;
        return Ok(Duration::from_secs_f64(v / 1_000.0));
    }
    if let Some(us) = raw.strip_suffix("us").or_else(|| raw.strip_suffix("µs")) {
        let v: f64 = us
            .parse()
            .map_err(|_| format!("invalid duration {raw:?}"))?;
        return Ok(Duration::from_secs_f64(v / 1_000_000.0));
    }
    if let Some(ns) = raw.strip_suffix("ns") {
        let v: u64 = ns
            .parse()
            .map_err(|_| format!("invalid duration {raw:?}"))?;
        return Ok(Duration::from_nanos(v));
    }
    if let Some(m) = raw.strip_suffix('m') {
        let v: f64 = m.parse().map_err(|_| format!("invalid duration {raw:?}"))?;
        return Ok(Duration::from_secs_f64(v * 60.0));
    }
    if let Some(s) = raw.strip_suffix('s') {
        let v: f64 = s.parse().map_err(|_| format!("invalid duration {raw:?}"))?;
        return Ok(Duration::from_secs_f64(v));
    }
    Err(format!(
        "unsupported duration unit in {raw:?} (expected ns, us, ms, s, m, or Nx)"
    ))
}

/// Top-of-file Go `benchfmt` configuration metadata (`goos`, `goarch`, `pkg`, `cpu`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BenchHeader {
    /// Operating system name matching `GOOS` (`linux`, `darwin`, `windows`, ...).
    pub goos: String,
    /// Architecture name matching `GOARCH` (`amd64`, `arm64`, ...).
    pub goarch: String,
    /// Package or crate identifier (`pkg: ...`).
    pub pkg: String,
    /// CPU model string (`cpu: ...`), when detectable.
    pub cpu: Option<String>,
    /// Extra `key: value` configuration lines (for example `impl: rust`).
    pub extra: Vec<(String, String)>,
}

impl BenchHeader {
    /// Detects host `goos`, `goarch`, and CPU model for `pkg`.
    #[must_use]
    pub fn detect(pkg: impl Into<String>) -> Self {
        let goos = match std::env::consts::OS {
            "macos" => "darwin",
            other => other,
        }
        .to_owned();
        let goarch = match std::env::consts::ARCH {
            "x86_64" => "amd64",
            "aarch64" => "arm64",
            "x86" => "386",
            other => other,
        }
        .to_owned();
        Self {
            goos,
            goarch,
            pkg: pkg.into(),
            cpu: detect_cpu_model(),
            extra: Vec::new(),
        }
    }

    /// Writes the configuration block in `benchfmt` format.
    pub fn write(&self, mut out: impl Write) -> io::Result<()> {
        writeln!(out, "goos: {}", self.goos)?;
        writeln!(out, "goarch: {}", self.goarch)?;
        writeln!(out, "pkg: {}", self.pkg)?;
        if let Some(cpu) = &self.cpu {
            writeln!(out, "cpu: {cpu}")?;
        }
        for (k, v) in &self.extra {
            writeln!(out, "{k}: {v}")?;
        }
        Ok(())
    }
}

fn detect_cpu_model() -> Option<String> {
    if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
        for line in cpuinfo.lines() {
            if let Some((lhs, rhs)) = line.split_once(':')
                && lhs.trim() == "model name"
            {
                let trimmed = rhs.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.to_owned());
                }
            }
        }
    }
    None
}

/// Single formatted `Benchmark...` result line matching Go's `testing.BenchmarkResult`.
#[derive(Clone, Copy, Debug)]
pub struct BenchRecord<'a> {
    /// Benchmark name without the leading `Benchmark` prefix (added automatically if absent).
    pub name: &'a str,
    /// Optional `-N` CPU/GOMAXPROCS suffix (omitted when `None` or `Some(1)`).
    pub cpu_procs: Option<usize>,
    /// Number of iterations executed in this epoch (`b.N`).
    pub iterations: u64,
    /// Mean nanoseconds per operation.
    pub ns_per_op: f64,
    /// Optional decimal throughput in megabytes per second (`1e6` bytes/sec).
    pub mb_per_sec: Option<f64>,
    /// Optional heap bytes allocated per operation (`B/op`).
    pub alloc_bytes_per_op: Option<u64>,
    /// Optional heap allocations per operation (`allocs/op`).
    pub allocs_per_op: Option<u64>,
}

impl BenchRecord<'_> {
    /// Writes this benchmark measurement line in Go `benchfmt` syntax.
    pub fn write(&self, mut out: impl Write) -> io::Result<()> {
        let base = if self.name.starts_with("Benchmark") {
            self.name.to_owned()
        } else {
            format!("Benchmark{}", self.name)
        };
        let full_name = match self.cpu_procs {
            Some(procs) if procs > 1 => format!("{base}-{procs}"),
            _ => base,
        };

        write!(out, "{full_name}\t{:8}\t", self.iterations.max(1))?;
        if self.ns_per_op >= 100.0 {
            write!(out, "{:10.0} ns/op", self.ns_per_op)?;
        } else if self.ns_per_op >= 10.0 {
            write!(out, "{:10.2} ns/op", self.ns_per_op)?;
        } else if self.ns_per_op >= 1.0 {
            write!(out, "{:10.3} ns/op", self.ns_per_op)?;
        } else {
            write!(out, "{:10.4} ns/op", self.ns_per_op)?;
        }

        if let Some(mbps) = self.mb_per_sec {
            write!(out, "\t{:8.2} MB/s", mbps)?;
        }
        if let Some(bytes) = self.alloc_bytes_per_op {
            write!(out, "\t{:8} B/op", bytes)?;
        }
        if let Some(allocs) = self.allocs_per_op {
            write!(out, "\t{:8} allocs/op", allocs)?;
        }
        writeln!(out)
    }
}

/// Normalizes a Rust or Criterion benchmark identifier into a whitespace-free Go `Benchmark*`
/// suffix while preserving `/key=value` sub-benchmark dimensions for `benchstat`.
#[must_use]
pub fn sanitize_bench_name(raw: &str) -> String {
    let trimmed = raw.trim().trim_start_matches("Benchmark");
    let mut out = String::with_capacity(trimmed.len() + 1);
    let mut chars = trimmed.chars();
    if let Some(first) = chars.next() {
        out.extend(first.to_uppercase());
    }
    for ch in chars {
        match ch {
            ' ' | '\t' => out.push('_'),
            _ => out.push(ch),
        }
    }
    out
}

/// Runtime options parsed from CLI arguments for [`Suite`].
#[derive(Clone, Debug)]
pub struct BenchConfig {
    /// Substring filter applied to benchmark names (`-bench` / `--filter`).
    pub filter: Option<String>,
    /// Target duration or iteration count per epoch (`-benchtime`).
    pub benchtime: BenchTime,
    /// Number of independent epochs to execute (`-count`).
    pub count: usize,
    /// Reported CPU suffix (`-cpu`), defaulting to available parallelism.
    pub cpu_procs: usize,
    /// Force allocation reporting on all benchmarks (`-benchmem`).
    pub benchmem: bool,
    /// Override `pkg:` header line (`--pkg`).
    pub pkg_override: Option<String>,
    /// List benchmark names without running them (`--list`).
    pub list_only: bool,
}

impl Default for BenchConfig {
    fn default() -> Self {
        let cpu_procs = thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        Self {
            filter: None,
            benchtime: BenchTime::default(),
            count: 1,
            cpu_procs,
            benchmem: false,
            pkg_override: None,
            list_only: false,
        }
    }
}

impl BenchConfig {
    /// Parses Go-style (`-bench`, `-benchtime`, `-count`, `-cpu`, `-benchmem`) and Cargo-style
    /// (`--bench`, `--list`, positional filter) CLI flags.
    pub fn from_args<I, S>(args: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut cfg = Self::default();
        let mut iter = args.into_iter();
        while let Some(raw) = iter.next() {
            let arg = raw.as_ref();
            if arg == "--bench" || arg == "-test.run" {
                continue;
            }
            if arg == "--list" {
                cfg.list_only = true;
                continue;
            }
            if arg == "-benchmem" || arg == "--benchmem" || arg == "-test.benchmem" {
                cfg.benchmem = true;
                continue;
            }
            if let Some(val) = split_flag(arg, &["-bench", "--filter", "-test.bench"], &mut iter)? {
                if val != "." && val != "^" {
                    cfg.filter = Some(
                        val.trim_start_matches('^')
                            .trim_end_matches('$')
                            .trim_start_matches("Benchmark")
                            .to_owned(),
                    );
                }
                continue;
            }
            if let Some(val) = split_flag(
                arg,
                &["-benchtime", "--benchtime", "-test.benchtime"],
                &mut iter,
            )? {
                cfg.benchtime = BenchTime::parse(&val)?;
                continue;
            }
            if let Some(val) = split_flag(arg, &["-count", "--count", "-test.count"], &mut iter)? {
                cfg.count = val
                    .parse::<usize>()
                    .map_err(|_| format!("invalid -count {val:?}"))?
                    .max(1);
                continue;
            }
            if let Some(val) = split_flag(arg, &["-cpu", "--cpu", "-test.cpu"], &mut iter)? {
                cfg.cpu_procs = val
                    .parse::<usize>()
                    .map_err(|_| format!("invalid -cpu {val:?}"))?
                    .max(1);
                continue;
            }
            if let Some(val) = split_flag(arg, &["-pkg", "--pkg"], &mut iter)? {
                cfg.pkg_override = Some(val);
                continue;
            }
            if !arg.starts_with('-') && cfg.filter.is_none() {
                cfg.filter = Some(arg.trim_start_matches("Benchmark").to_owned());
            }
        }
        Ok(cfg)
    }
}

fn split_flag<I, S>(arg: &str, names: &[&str], iter: &mut I) -> Result<Option<String>, String>
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    for &name in names {
        if arg == name {
            let val = iter
                .next()
                .ok_or_else(|| format!("missing argument for {name}"))?;
            return Ok(Some(val.as_ref().to_owned()));
        }
        if let Some(rest) = arg.strip_prefix(name).and_then(|s| s.strip_prefix('=')) {
            return Ok(Some(rest.to_owned()));
        }
    }
    Ok(None)
}

/// Per-epoch state passed to a benchmark closure, mirroring Go's `*testing.B`.
pub struct Bencher<'a> {
    target_n: u64,
    remaining: u64,
    started: bool,
    start_instant: Option<Instant>,
    elapsed: Duration,
    bytes_per_op: Option<u64>,
    report_allocs: bool,
    alloc_tracker: Option<&'a dyn AllocTracker>,
    start_alloc: AllocSnapshot,
    alloc_delta: AllocSnapshot,
}

impl<'a> Bencher<'a> {
    fn new(
        target_n: u64,
        default_benchmem: bool,
        alloc_tracker: Option<&'a dyn AllocTracker>,
    ) -> Self {
        Self {
            target_n,
            remaining: target_n,
            started: false,
            start_instant: None,
            elapsed: Duration::ZERO,
            bytes_per_op: None,
            report_allocs: default_benchmem,
            alloc_tracker,
            start_alloc: AllocSnapshot::default(),
            alloc_delta: AllocSnapshot::default(),
        }
    }

    /// Returns the calibrated target iteration count (`b.N` in Go) for the current epoch.
    #[must_use]
    pub const fn iterations(&self) -> u64 {
        self.target_n
    }

    /// Sets the number of bytes processed in a single operation so `MB/s` is reported.
    pub fn set_bytes(&mut self, bytes: u64) {
        self.bytes_per_op = Some(bytes);
    }

    /// Enables `B/op` and `allocs/op` reporting when the [`Suite`] has an [`AllocTracker`].
    pub fn report_allocs(&mut self) {
        self.report_allocs = true;
    }

    /// Resets the elapsed timer and allocation baseline, excluding any prior setup work.
    pub fn reset_timer(&mut self) {
        if let Some(tracker) = self.alloc_tracker {
            self.start_alloc = tracker.snapshot();
        }
        self.elapsed = Duration::ZERO;
        if self.start_instant.is_some() {
            self.start_instant = Some(Instant::now());
        }
    }

    /// Go 1.24 `b.Loop()` equivalent: starts timing and allocation tracking on the first call,
    /// runs `N` iterations, and stops timing + snapshots allocations as soon as the loop finishes.
    #[inline]
    pub fn keep_running(&mut self) -> bool {
        if !self.started {
            self.started = true;
            if let Some(tracker) = self.alloc_tracker {
                self.start_alloc = tracker.snapshot();
            }
            self.start_instant = Some(Instant::now());
        }
        if self.remaining == 0 {
            if let Some(started_at) = self.start_instant.take() {
                self.elapsed += started_at.elapsed();
            }
            if let Some(tracker) = self.alloc_tracker {
                self.alloc_delta = self.start_alloc.delta_to(tracker.snapshot());
            }
            return false;
        }
        self.remaining -= 1;
        true
    }

    /// Executes `f` for the calibrated number of iterations, preventing return-value dead-code
    /// elimination with [`std::hint::black_box`].
    pub fn iter<R>(&mut self, mut f: impl FnMut() -> R) {
        while self.keep_running() {
            black_box(f());
        }
    }
}

type BenchFn<'a> = Box<dyn FnMut(&mut Bencher<'_>) + 'a>;

/// Collection of benchmarks configured to emit Go `benchfmt` output.
pub struct Suite<'a> {
    header: BenchHeader,
    config: BenchConfig,
    alloc_tracker: Option<&'a dyn AllocTracker>,
    benches: Vec<(String, BenchFn<'a>)>,
}

impl<'a> Suite<'a> {
    /// Creates a [`Suite`] configured from `std::env::args().skip(1)`.
    #[must_use]
    pub fn from_env(default_pkg: impl Into<String>) -> Self {
        let cfg = BenchConfig::from_args(std::env::args().skip(1)).unwrap_or_else(|err| {
            eprintln!("benchfmt argument error: {err}");
            std::process::exit(2);
        });
        Self::with_config(default_pkg, cfg)
    }

    /// Creates a [`Suite`] with an explicit [`BenchConfig`].
    #[must_use]
    pub fn with_config(default_pkg: impl Into<String>, config: BenchConfig) -> Self {
        let pkg = config
            .pkg_override
            .clone()
            .unwrap_or_else(|| default_pkg.into());
        Self {
            header: BenchHeader::detect(pkg),
            config,
            alloc_tracker: None,
            benches: Vec::new(),
        }
    }

    /// Attaches a global [`AllocTracker`] (such as [`CountingAlloc`]) so `B/op` and `allocs/op`
    /// can be measured.
    #[must_use]
    pub fn with_alloc(mut self, tracker: &'a dyn AllocTracker) -> Self {
        self.alloc_tracker = Some(tracker);
        self
    }

    /// Adds a custom `key: value` metadata header line (for example `("impl", "rust")`).
    #[must_use]
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.header.extra.push((key.into(), value.into()));
        self
    }

    /// Registers a benchmark function under `name`.
    #[must_use]
    pub fn bench(mut self, name: impl Into<String>, f: impl FnMut(&mut Bencher<'_>) + 'a) -> Self {
        self.benches
            .push((sanitize_bench_name(&name.into()), Box::new(f)));
        self
    }

    /// Runs all matching benchmarks and writes `benchfmt` output to stdout.
    pub fn run(&mut self) {
        let stdout = io::stdout();
        let mut handle = stdout.lock();
        self.run_to(&mut handle)
            .expect("failed writing benchmark output to stdout");
    }

    /// Runs all matching benchmarks and writes `benchfmt` output to `out`.
    pub fn run_to(&mut self, mut out: impl Write) -> io::Result<()> {
        if self.config.list_only {
            for (name, _) in &self.benches {
                writeln!(out, "Benchmark{name}: benchmark")?;
            }
            return Ok(());
        }

        self.header.write(&mut out)?;
        out.flush()?;

        for (name, func) in &mut self.benches {
            if let Some(filter) = &self.config.filter
                && !name.contains(filter.as_str())
            {
                continue;
            }
            for _ in 0..self.config.count {
                let rec = run_single_epoch(
                    name,
                    self.config.benchtime,
                    self.config.cpu_procs,
                    self.config.benchmem,
                    self.alloc_tracker,
                    func,
                );
                rec.write(&mut out)?;
                out.flush()?;
            }
        }

        writeln!(out, "PASS")?;
        out.flush()
    }
}

fn run_single_epoch<'a>(
    name: &'a str,
    benchtime: BenchTime,
    cpu_procs: usize,
    default_benchmem: bool,
    alloc_tracker: Option<&dyn AllocTracker>,
    func: &mut BenchFn<'_>,
) -> BenchRecord<'a> {
    let mut n = match benchtime {
        BenchTime::Iterations(fixed) => fixed,
        BenchTime::Duration(_) => 1,
    };

    loop {
        let mut bencher = Bencher::new(n, default_benchmem, alloc_tracker);
        func(&mut bencher);
        // Finalize timing if the benchmark body exited early without exhausting `keep_running`.
        if let Some(started_at) = bencher.start_instant.take() {
            bencher.elapsed += started_at.elapsed();
            if let Some(tracker) = bencher.alloc_tracker {
                bencher.alloc_delta = bencher.start_alloc.delta_to(tracker.snapshot());
            }
        }

        let done = match benchtime {
            BenchTime::Iterations(_) => true,
            BenchTime::Duration(target) => bencher.elapsed >= target || n >= MAX_ITERATIONS,
        };

        if done {
            let elapsed_ns = bencher.elapsed.as_nanos() as f64;
            let ns_per_op = elapsed_ns / (n as f64);
            let mb_per_sec = bencher.bytes_per_op.and_then(|bytes| {
                if elapsed_ns > 0.0 {
                    Some((bytes as f64 * n as f64 * 1_000.0) / elapsed_ns)
                } else {
                    None
                }
            });
            let (alloc_bytes_per_op, allocs_per_op) =
                if bencher.report_allocs && alloc_tracker.is_some() {
                    (
                        Some(bencher.alloc_delta.bytes / n),
                        Some(bencher.alloc_delta.allocs / n),
                    )
                } else {
                    (None, None)
                };
            return BenchRecord {
                name,
                cpu_procs: Some(cpu_procs),
                iterations: n,
                ns_per_op,
                mb_per_sec,
                alloc_bytes_per_op,
                allocs_per_op,
            };
        }

        let target_ns = match benchtime {
            BenchTime::Duration(d) => d.as_nanos(),
            BenchTime::Iterations(_) => unreachable!(),
        };
        let elapsed_ns = bencher.elapsed.as_nanos();
        let next = if let Some(predicted) = target_ns
            .saturating_mul(u128::from(n))
            .checked_div(elapsed_ns)
        {
            let predicted = predicted as u64;
            // Grow by 20% above the linear prediction, capped at 100x per step, matching Go.
            let padded = predicted.saturating_add(predicted / 5);
            padded.min(n.saturating_mul(100)).max(n + 1)
        } else {
            n.saturating_mul(100)
        };
        n = next.min(MAX_ITERATIONS);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[global_allocator]
    static TEST_ALLOC: CountingAlloc = CountingAlloc::new();

    #[test]
    fn counting_alloc_and_suite_emit_valid_benchfmt() {
        let cfg = BenchConfig {
            filter: Some("AllocVec".to_owned()),
            benchtime: BenchTime::Iterations(50),
            count: 2,
            cpu_procs: 2,
            benchmem: false,
            pkg_override: Some("example/pkg".to_owned()),
            list_only: false,
        };
        let mut out = Vec::new();
        Suite::with_config("ignored", cfg)
            .with_alloc(&TEST_ALLOC)
            .bench("AllocVec/case=1kb", |b| {
                b.set_bytes(1024);
                b.report_allocs();
                while b.keep_running() {
                    let v = vec![1u8; 1024];
                    black_box(v);
                }
            })
            .bench("SkippedBench", |b| {
                b.iter(|| 1 + 1);
            })
            .run_to(&mut out)
            .unwrap();

        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("pkg: example/pkg\n"), "{text}");
        assert!(
            text.contains("BenchmarkAllocVec/case=1kb-2\t      50\t"),
            "{text}"
        );
        assert!(text.contains("1024 B/op\t       1 allocs/op"), "{text}");
        assert!(!text.contains("SkippedBench"), "{text}");
        assert!(text.ends_with("PASS\n"), "{text}");
    }

    #[test]
    fn converts_criterion_directory_to_benchfmt() {
        let tmp = std::env::temp_dir().join(format!("benchfmt_test_{}", std::process::id()));
        let new_dir = tmp.join("ParsePromText/rust/new");
        fs::create_dir_all(&new_dir).unwrap();
        fs::write(
            new_dir.join("benchmark.json"),
            r#"{"group_id":"ParsePromText","function_id":"parser=rust","value_str":null,"throughput":{"Bytes":1000},"full_id":"ParsePromText/parser=rust","directory_name":"ParsePromText/rust","title":"ParsePromText/parser=rust"}"#,
        )
        .unwrap();
        fs::write(
            new_dir.join("sample.json"),
            r#"{"sampling_mode":"Linear","iters":[10.0,20.0,30.0,40.0],"times":[1000.0,2000.0,3000.0,4000.0]}"#,
        )
        .unwrap();

        let opts = ConvertOptions {
            header: BenchHeader {
                goos: "linux".to_owned(),
                goarch: "amd64".to_owned(),
                pkg: "expfmt".to_owned(),
                cpu: None,
                extra: Vec::new(),
            },
            cpu_procs: Some(2),
            criterion_epochs: 2,
            criterion_baseline: "new".to_owned(),
        };
        let mut out = Vec::new();
        let count = convert_criterion_dir(&tmp, &opts, &mut out).unwrap();
        fs::remove_dir_all(&tmp).unwrap();

        assert_eq!(count, 2);
        let text = String::from_utf8(out).unwrap();
        assert!(
            text.contains(
                "BenchmarkParsePromText/parser=rust-2\t      30\t       100 ns/op\t10000.00 MB/s"
            ),
            "{text}"
        );
    }

    #[test]
    fn converts_libtest_output_to_benchfmt() {
        let sample = "running 1 test\ntest parse::promtext ... bench:   1,250.50 ns/iter (+/- 14.00) = 152.10 MB/s\n";
        let mut out = Vec::new();
        let count = convert_libtest_output(sample, &ConvertOptions::default(), &mut out).unwrap();
        assert_eq!(count, 1);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("BenchmarkParse::promtext"), "{text}");
        assert!(text.contains("1250 ns/op"), "{text}");
        assert!(text.contains("152.10 MB/s"), "{text}");
    }
}
