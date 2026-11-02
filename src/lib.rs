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
#[cfg(feature = "pprof")]
mod pprof_proto;
#[cfg(feature = "pprof")]
mod pprof_support;

use std::fs;
use std::hint::black_box;
use std::io::{self, Write};
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

pub use alloc::{AllocSnapshot, AllocTracker, CountingAlloc};
pub use convert::{ConvertOptions, convert_criterion_dir, convert_libtest_output};

const MAX_ITERATIONS: u64 = 1_000_000_000;

/// Default Poisson mean byte interval between heap profile samples (`512 KiB`), matching Go's
/// `runtime.MemProfileRate`.
pub const DEFAULT_MEM_PROFILE_RATE: usize = 512 * 1024;

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
    /// Write a pprof CPU profile to this path (`-cpuprofile`).
    pub cpu_profile: Option<PathBuf>,
    /// Write a pprof heap allocation profile to this path (`-memprofile`).
    pub mem_profile: Option<PathBuf>,
    /// Mean bytes allocated per heap sample (`-memprofilerate`; `1` samples every allocation, `0`
    /// disables sampling).
    pub mem_profile_rate: usize,
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
            cpu_profile: None,
            mem_profile: None,
            mem_profile_rate: DEFAULT_MEM_PROFILE_RATE,
        }
    }
}

impl BenchConfig {
    /// Parses Go-style (`-bench`, `-benchtime`, `-count`, `-cpu`, `-benchmem`, `-cpuprofile`,
    /// `-memprofile`, `-memprofilerate`) and Cargo-style (`--bench`, `--list`, positional filter)
    /// CLI flags.
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
            if let Some(val) = split_flag(
                arg,
                &["-cpuprofile", "--cpuprofile", "-test.cpuprofile"],
                &mut iter,
            )? {
                cfg.cpu_profile = Some(PathBuf::from(val));
                continue;
            }
            if let Some(val) = split_flag(
                arg,
                &["-memprofile", "--memprofile", "-test.memprofile"],
                &mut iter,
            )? {
                cfg.mem_profile = Some(PathBuf::from(val));
                continue;
            }
            if let Some(val) = split_flag(
                arg,
                &[
                    "-memprofilerate",
                    "--memprofilerate",
                    "-test.memprofilerate",
                ],
                &mut iter,
            )? {
                cfg.mem_profile_rate = val
                    .parse::<usize>()
                    .map_err(|_| format!("invalid -memprofilerate {val:?}"))?;
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

/// Scoped profiler notified around the timed portion of each benchmark run.
///
/// [`Bencher`] invokes [`Profiler::start_bench`] before recording its allocation baseline and
/// start timestamp, and invokes [`Profiler::stop_bench`] after recording elapsed time and
/// allocation deltas so profiler work does not skew `ns/op`, `B/op`, or `allocs/op`.
pub trait Profiler {
    /// Starts profiling for a timed `keep_running` window of `bench_name`.
    fn start_bench(&mut self, bench_name: &str) -> io::Result<()>;

    /// Stops profiling for the current timed window of `bench_name`.
    fn stop_bench(&mut self, bench_name: &str) -> io::Result<()>;

    /// Flushes any accumulated profile output once all benchmarks finish.
    fn finish(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Per-epoch state passed to a benchmark closure, mirroring Go's `*testing.B`.
pub struct Bencher<'a> {
    bench_name: &'a str,
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
    profiler: Option<&'a mut dyn Profiler>,
    profiler_running: bool,
    profiler_err: Option<io::Error>,
}

impl<'a> Bencher<'a> {
    fn new(
        bench_name: &'a str,
        target_n: u64,
        default_benchmem: bool,
        alloc_tracker: Option<&'a dyn AllocTracker>,
        profiler: Option<&'a mut dyn Profiler>,
    ) -> Self {
        Self {
            bench_name,
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
            profiler,
            profiler_running: false,
            profiler_err: None,
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
        if self.profiler_running
            && let Some(prof) = self.profiler.as_deref_mut()
            && let Err(err) = prof
                .stop_bench(self.bench_name)
                .and_then(|()| prof.start_bench(self.bench_name))
        {
            self.profiler_running = false;
            if self.profiler_err.is_none() {
                self.profiler_err = Some(err);
            }
        }
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
            // NOTE: Start the profiler before capturing allocation and wall-clock baselines so any
            // profiler initialization cost stays outside the measured benchmark window.
            if let Some(prof) = self.profiler.as_deref_mut() {
                if let Err(err) = prof.start_bench(self.bench_name) {
                    self.profiler_err = Some(err);
                    return false;
                }
                self.profiler_running = true;
            }
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
            // NOTE: Stop the profiler only after wall-clock and allocation deltas are recorded.
            if self.profiler_running {
                self.profiler_running = false;
                if let Some(prof) = self.profiler.as_deref_mut()
                    && let Err(err) = prof.stop_bench(self.bench_name)
                    && self.profiler_err.is_none()
                {
                    self.profiler_err = Some(err);
                }
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
    profiler: Option<&'a mut dyn Profiler>,
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
            profiler: None,
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

    /// Attaches a custom [`Profiler`] scoped to the timed portion of each benchmark run.
    #[must_use]
    pub fn with_profiler(mut self, profiler: &'a mut dyn Profiler) -> Self {
        self.profiler = Some(profiler);
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
        if let Err(err) = self.run_to(&mut handle) {
            eprintln!("benchfmt error: {err}");
            std::process::exit(1);
        }
    }

    /// Runs all matching benchmarks and writes `benchfmt` output to `out`.
    pub fn run_to(&mut self, mut out: impl Write) -> io::Result<()> {
        if self.config.list_only {
            for (name, _) in &self.benches {
                writeln!(out, "Benchmark{name}: benchmark")?;
            }
            return Ok(());
        }

        #[cfg(not(feature = "pprof"))]
        if self.config.cpu_profile.is_some() || self.config.mem_profile.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "-cpuprofile and -memprofile require enabling the `pprof` Cargo feature on `benchfmt`",
            ));
        }

        #[cfg(feature = "pprof")]
        let mut builtin_pprof = pprof_support::PprofSession::new(&self.config)?;

        self.header.write(&mut out)?;
        out.flush()?;

        for (name, func) in &mut self.benches {
            if let Some(filter) = &self.config.filter
                && !name.contains(filter.as_str())
            {
                continue;
            }
            for _ in 0..self.config.count {
                #[cfg(feature = "pprof")]
                let rec = {
                    let has_any = builtin_pprof.is_some() || self.profiler.is_some();
                    let mut combined = pprof_support::CombinedProfiler {
                        builtin: builtin_pprof.as_mut(),
                        custom: self
                            .profiler
                            .as_mut()
                            .map(|p| &mut **p as &mut dyn Profiler),
                    };
                    let prof_arg: Option<&mut dyn Profiler> =
                        if has_any { Some(&mut combined) } else { None };
                    run_single_epoch(
                        name,
                        self.config.benchtime,
                        self.config.cpu_procs,
                        self.config.benchmem,
                        self.alloc_tracker,
                        prof_arg,
                        func,
                    )?
                };
                #[cfg(not(feature = "pprof"))]
                let rec = run_single_epoch(
                    name,
                    self.config.benchtime,
                    self.config.cpu_procs,
                    self.config.benchmem,
                    self.alloc_tracker,
                    self.profiler
                        .as_mut()
                        .map(|p| &mut **p as &mut dyn Profiler),
                    func,
                )?;
                rec.write(&mut out)?;
                out.flush()?;
            }
        }

        #[cfg(feature = "pprof")]
        if builtin_pprof.is_some() || self.profiler.is_some() {
            let mut combined = pprof_support::CombinedProfiler {
                builtin: builtin_pprof.as_mut(),
                custom: self
                    .profiler
                    .as_mut()
                    .map(|p| &mut **p as &mut dyn Profiler),
            };
            combined.finish()?;
        }
        #[cfg(not(feature = "pprof"))]
        if let Some(prof) = self.profiler.as_deref_mut() {
            prof.finish()?;
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
    mut profiler: Option<&mut dyn Profiler>,
    func: &mut BenchFn<'_>,
) -> io::Result<BenchRecord<'a>> {
    let mut n = match benchtime {
        BenchTime::Iterations(fixed) => fixed,
        BenchTime::Duration(_) => 1,
    };

    loop {
        let (elapsed, bytes_per_op, report_allocs, alloc_delta) = {
            let mut bencher = Bencher::new(
                name,
                n,
                default_benchmem,
                alloc_tracker,
                profiler.as_mut().map(|p| &mut **p as &mut dyn Profiler),
            );
            func(&mut bencher);
            // Finalize timing if the benchmark body exited early without exhausting `keep_running`.
            if let Some(started_at) = bencher.start_instant.take() {
                bencher.elapsed += started_at.elapsed();
                if let Some(tracker) = bencher.alloc_tracker {
                    bencher.alloc_delta = bencher.start_alloc.delta_to(tracker.snapshot());
                }
            }
            if bencher.profiler_running {
                bencher.profiler_running = false;
                if let Some(prof) = bencher.profiler.as_deref_mut()
                    && let Err(err) = prof.stop_bench(name)
                    && bencher.profiler_err.is_none()
                {
                    bencher.profiler_err = Some(err);
                }
            }
            if let Some(err) = bencher.profiler_err {
                return Err(err);
            }
            (
                bencher.elapsed,
                bencher.bytes_per_op,
                bencher.report_allocs,
                bencher.alloc_delta,
            )
        };

        let done = match benchtime {
            BenchTime::Iterations(_) => true,
            BenchTime::Duration(target) => elapsed >= target || n >= MAX_ITERATIONS,
        };

        if done {
            let elapsed_ns = elapsed.as_nanos() as f64;
            let ns_per_op = elapsed_ns / (n as f64);
            let mb_per_sec = bytes_per_op.and_then(|bytes| {
                if elapsed_ns > 0.0 {
                    Some((bytes as f64 * n as f64 * 1_000.0) / elapsed_ns)
                } else {
                    None
                }
            });
            let (alloc_bytes_per_op, allocs_per_op) = if report_allocs && alloc_tracker.is_some() {
                (Some(alloc_delta.bytes / n), Some(alloc_delta.allocs / n))
            } else {
                (None, None)
            };
            return Ok(BenchRecord {
                name,
                cpu_procs: Some(cpu_procs),
                iterations: n,
                ns_per_op,
                mb_per_sec,
                alloc_bytes_per_op,
                allocs_per_op,
            });
        }

        let target_ns = match benchtime {
            BenchTime::Duration(d) => d.as_nanos(),
            BenchTime::Iterations(_) => unreachable!(),
        };
        let elapsed_ns = elapsed.as_nanos();
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
            ..BenchConfig::default()
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

    #[test]
    fn parses_profile_cli_flags() {
        let cfg = BenchConfig::from_args([
            "-cpuprofile",
            "cpu.pb.gz",
            "-test.memprofile=mem.pb.gz",
            "--memprofilerate",
            "4096",
        ])
        .unwrap();
        assert_eq!(
            cfg.cpu_profile.as_deref(),
            Some(std::path::Path::new("cpu.pb.gz"))
        );
        assert_eq!(
            cfg.mem_profile.as_deref(),
            Some(std::path::Path::new("mem.pb.gz"))
        );
        assert_eq!(cfg.mem_profile_rate, 4096);
    }

    #[test]
    fn custom_profiler_is_scoped_outside_alloc_and_timing_baselines() {
        #[derive(Default)]
        struct RecordingProfiler {
            events: Vec<String>,
            scratch: Vec<Vec<u8>>,
        }

        impl Profiler for RecordingProfiler {
            fn start_bench(&mut self, bench_name: &str) -> io::Result<()> {
                self.scratch.push(vec![0u8; 8192]);
                self.events.push(format!("start:{bench_name}"));
                Ok(())
            }

            fn stop_bench(&mut self, bench_name: &str) -> io::Result<()> {
                self.scratch.push(vec![0u8; 8192]);
                self.events.push(format!("stop:{bench_name}"));
                Ok(())
            }

            fn finish(&mut self) -> io::Result<()> {
                self.events.push("finish".to_owned());
                Ok(())
            }
        }

        let cfg = BenchConfig {
            benchtime: BenchTime::Iterations(10),
            count: 2,
            cpu_procs: 1,
            benchmem: true,
            ..BenchConfig::default()
        };
        let mut profiler = RecordingProfiler::default();
        let mut out = Vec::new();
        Suite::with_config("pkg", cfg)
            .with_alloc(&TEST_ALLOC)
            .with_profiler(&mut profiler)
            .bench("ScopedWork", |b| {
                // Allocate 4 KiB outside `keep_running`; neither this nor the profiler's 8 KiB
                // vectors should appear in `B/op`.
                let setup = vec![7u8; 4096];
                while b.keep_running() {
                    let work = vec![setup[0]; 256];
                    black_box(work);
                }
            })
            .run_to(&mut out)
            .unwrap();

        assert_eq!(
            profiler.events,
            vec![
                "start:ScopedWork",
                "stop:ScopedWork",
                "start:ScopedWork",
                "stop:ScopedWork",
                "finish",
            ]
        );
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("     256 B/op\t       1 allocs/op"), "{text}");
    }

    #[cfg(not(feature = "pprof"))]
    #[test]
    fn rejects_profile_flags_when_pprof_feature_is_disabled() {
        let cfg = BenchConfig {
            benchtime: BenchTime::Iterations(1),
            mem_profile: Some(PathBuf::from("mem.pb.gz")),
            ..BenchConfig::default()
        };
        let mut out = Vec::new();
        let err = Suite::with_config("pkg", cfg)
            .bench("Noop", |b| b.iter(|| 1))
            .run_to(&mut out)
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Unsupported);
    }

    #[cfg(feature = "pprof")]
    #[inline(never)]
    fn unmeasured_setup_alloc_helper(size: usize) -> Vec<u8> {
        vec![9u8; size]
    }

    #[cfg(feature = "pprof")]
    #[inline(never)]
    fn tracked_heap_alloc_helper(size: usize) -> Vec<u8> {
        vec![42u8; size]
    }

    #[cfg(feature = "pprof")]
    #[inline(never)]
    fn cpu_burn_helper(rounds: u64) -> u64 {
        let mut acc = 1u64;
        for i in 0..rounds {
            acc = acc.wrapping_mul(6364136223846793005).wrapping_add(i | 1);
            black_box(acc);
        }
        acc
    }

    #[cfg(feature = "pprof")]
    #[test]
    fn writes_scoped_mem_and_cpu_pprof_profiles() {
        use std::io::Read;

        let dir = std::env::temp_dir().join(format!("benchfmt_pprof_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let mem_path = dir.join("mem.pb.gz");
        let cpu_path = dir.join("cpu.pb.gz");

        let cfg = BenchConfig {
            benchtime: BenchTime::Iterations(40),
            count: 1,
            cpu_procs: 1,
            benchmem: true,
            #[cfg(unix)]
            cpu_profile: Some(cpu_path.clone()),
            mem_profile: Some(mem_path.clone()),
            // Sample every allocation so deterministic assertions hold on small iteration counts.
            mem_profile_rate: 1,
            ..BenchConfig::default()
        };

        let mut out = Vec::new();
        Suite::with_config("pprof_test", cfg)
            .with_alloc(&TEST_ALLOC)
            .bench("BurnAndAlloc", |b| {
                // Unmeasured setup allocation must not appear in `B/op` or `mem.pb.gz`.
                let unmeasured_setup = unmeasured_setup_alloc_helper(65_536);
                black_box(&unmeasured_setup);
                while b.keep_running() {
                    black_box(tracked_heap_alloc_helper(2048));
                    black_box(cpu_burn_helper(250_000));
                }
            })
            .run_to(&mut out)
            .unwrap();

        let text = String::from_utf8(out).unwrap();
        assert!(
            text.contains("    2048 B/op\t       1 allocs/op"),
            "profiler internal allocations leaked into benchfmt output:\n{text}"
        );

        let mem_gz = fs::read(&mem_path).unwrap();
        assert_eq!(&mem_gz[..2], &[0x1f, 0x8b], "expected gzip magic header");
        let mut mem_raw = Vec::new();
        flate2::read::GzDecoder::new(&mem_gz[..])
            .read_to_end(&mut mem_raw)
            .unwrap();
        let mem_lossy = String::from_utf8_lossy(&mem_raw);
        assert!(mem_lossy.contains("alloc_space"), "{mem_lossy}");
        assert!(
            mem_lossy.contains("tracked_heap_alloc_helper"),
            "expected symbolicated helper in memprofile"
        );
        assert!(
            !mem_lossy.contains("unmeasured_setup_alloc_helper"),
            "setup allocations outside keep_running leaked into memprofile"
        );

        #[cfg(unix)]
        {
            let cpu_gz = fs::read(&cpu_path).unwrap();
            assert_eq!(&cpu_gz[..2], &[0x1f, 0x8b], "expected gzip magic header");
            let mut cpu_raw = Vec::new();
            flate2::read::GzDecoder::new(&cpu_gz[..])
                .read_to_end(&mut cpu_raw)
                .unwrap();
            let cpu_lossy = String::from_utf8_lossy(&cpu_raw);
            assert!(cpu_lossy.contains("nanoseconds"), "{cpu_lossy}");
            assert!(
                cpu_lossy.contains("cpu_burn_helper"),
                "expected symbolicated helper in cpuprofile"
            );
        }

        if let Ok(go_ver) = std::process::Command::new("go").arg("version").output()
            && go_ver.status.success()
        {
            let mem_top = std::process::Command::new("go")
                .args(["tool", "pprof", "-top", mem_path.to_str().unwrap()])
                .output()
                .unwrap();
            let mem_stdout = String::from_utf8_lossy(&mem_top.stdout);
            let mem_stderr = String::from_utf8_lossy(&mem_top.stderr);
            assert!(
                mem_top.status.success(),
                "go tool pprof failed on memprofile:\nstdout:\n{mem_stdout}\nstderr:\n{mem_stderr}"
            );
            assert!(
                mem_stdout.contains("tracked_heap_alloc_helper"),
                "go tool pprof -top missing tracked_heap_alloc_helper:\n{mem_stdout}"
            );

            #[cfg(unix)]
            {
                let cpu_top = std::process::Command::new("go")
                    .args(["tool", "pprof", "-top", cpu_path.to_str().unwrap()])
                    .output()
                    .unwrap();
                let cpu_stdout = String::from_utf8_lossy(&cpu_top.stdout);
                let cpu_stderr = String::from_utf8_lossy(&cpu_top.stderr);
                assert!(
                    cpu_top.status.success(),
                    "go tool pprof failed on cpuprofile:\nstdout:\n{cpu_stdout}\nstderr:\n{cpu_stderr}"
                );
                assert!(
                    cpu_stdout.contains("cpu_burn_helper"),
                    "go tool pprof -top missing cpu_burn_helper:\n{cpu_stdout}"
                );
            }
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(feature = "pprof")]
    #[test]
    fn poisson_memprofile_sampling_scales_allocations() {
        use std::io::Read;

        let dir =
            std::env::temp_dir().join(format!("benchfmt_pprof_poisson_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let mem_path = dir.join("poisson_mem.pb.gz");

        let cfg = BenchConfig {
            benchtime: BenchTime::Iterations(1_000),
            count: 1,
            cpu_procs: 1,
            benchmem: true,
            mem_profile: Some(mem_path.clone()),
            mem_profile_rate: 16 * 1024,
            ..BenchConfig::default()
        };

        let mut out = Vec::new();
        Suite::with_config("pprof_poisson", cfg)
            .with_alloc(&TEST_ALLOC)
            .bench("PoissonAlloc", |b| {
                while b.keep_running() {
                    black_box(tracked_heap_alloc_helper(1024));
                }
            })
            .run_to(&mut out)
            .unwrap();

        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("    1024 B/op\t       1 allocs/op"), "{text}");

        let mem_gz = fs::read(&mem_path).unwrap();
        let mut mem_raw = Vec::new();
        flate2::read::GzDecoder::new(&mem_gz[..])
            .read_to_end(&mut mem_raw)
            .unwrap();
        assert!(
            String::from_utf8_lossy(&mem_raw).contains("tracked_heap_alloc_helper"),
            "expected sampled stack in Poisson memprofile"
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
