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

//! Scoped CPU (`-cpuprofile`) and sampled heap (`-memprofile`) pprof collectors.

use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime};

use crate::pprof_proto::{ProfileBuilder, ResolvedSymbol};
use crate::{BenchConfig, Profiler};

const MAX_HEAP_STACK_DEPTH: usize = 96;
#[cfg(unix)]
const CPU_SAMPLE_FREQUENCY_HZ: i32 = 99;

/// Serializes active pprof sessions within a process because `SIGPROF` timers and global heap
/// sampling state are process-wide resources.
static SESSION_LOCK: Mutex<()> = Mutex::new(());

/// Monotonic generation bumped on every [`PprofSession::new`] so per-thread sampling state resets
/// cleanly when `-memprofilerate` changes across runs.
static SESSION_GEN: AtomicU64 = AtomicU64::new(1);

/// Active Poisson mean byte interval (`0` means heap sampling is disabled).
static HEAP_SAMPLE_RATE: AtomicUsize = AtomicUsize::new(0);
static RNG_SEED_COUNTER: AtomicU64 = AtomicU64::new(0x9e37_79b9_7f4a_7c15);

#[derive(Default)]
struct HeapSampleValue {
    // Accumulate exact unrounded Poisson estimates and round once when emitting `profile.proto` so
    // half-integer scale factors (`1.5x`) do not suffer systematic per-sample rounding bias.
    alloc_objects: f64,
    alloc_space: f64,
    symbol_addrs: Box<[usize]>,
}

static HEAP_SAMPLES: Mutex<Option<HashMap<Box<[usize]>, HeapSampleValue>>> = Mutex::new(None);

#[derive(Clone, Copy)]
struct ThreadHeapState {
    in_sample: bool,
    initialized: bool,
    session_gen: u64,
    rate: usize,
    bytes_until_sample: u64,
    rng: u64,
}

thread_local! {
    // `const` initialization avoids any heap allocation when `CountingAlloc` inspects TLS.
    static THREAD_HEAP_STATE: Cell<ThreadHeapState> = const {
        Cell::new(ThreadHeapState {
            in_sample: false,
            initialized: false,
            session_gen: 0,
            rate: 0,
            bytes_until_sample: 0,
            rng: 0,
        })
    };
}

/// Returns `true` when the calling thread is currently capturing a heap sample so
/// [`crate::CountingAlloc`] skips counting and recursive sampling for internal profiler work.
#[inline]
pub(crate) fn is_in_heap_sample() -> bool {
    THREAD_HEAP_STATE
        .try_with(|cell| cell.get().in_sample)
        .unwrap_or(false)
}

/// Samples heap allocations on the calling thread using Go-style Poisson byte sampling when a
/// `-memprofile` window is active.
#[inline]
pub(crate) fn maybe_record_heap_alloc(bytes: usize) {
    let rate = HEAP_SAMPLE_RATE.load(Ordering::Relaxed);
    if rate == 0 || bytes == 0 {
        return;
    }
    slow_check_heap_sample(bytes, rate);
}

fn slow_check_heap_sample(bytes: usize, rate: usize) {
    // Synchronize with `HEAP_SAMPLE_RATE` `Release` stores so `SESSION_GEN` is never stale on
    // weakly-ordered architectures while keeping the disabled fast path barrier-free.
    std::sync::atomic::fence(Ordering::Acquire);
    let current_gen = SESSION_GEN.load(Ordering::Relaxed);
    let should_sample = THREAD_HEAP_STATE
        .try_with(|cell| {
            let mut st = cell.get();
            if st.in_sample {
                return false;
            }
            if !st.initialized {
                let seed = RNG_SEED_COUNTER
                    .fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::Relaxed)
                    .max(1);
                st.rng = splitmix64(seed);
                st.initialized = true;
            }
            if st.session_gen != current_gen || st.rate != rate {
                st.session_gen = current_gen;
                st.rate = rate;
                st.bytes_until_sample = if rate <= 1 {
                    0
                } else {
                    next_exponential_bytes(&mut st.rng, rate)
                };
            }

            let size = bytes as u64;
            if rate > 1 && size < st.bytes_until_sample {
                st.bytes_until_sample -= size;
                cell.set(st);
                return false;
            }

            st.bytes_until_sample = if rate <= 1 {
                0
            } else {
                next_exponential_bytes(&mut st.rng, rate)
            };
            st.in_sample = true;
            cell.set(st);
            true
        })
        .unwrap_or(false);

    if should_sample {
        let _reset = ResetInSampleGuard;
        capture_heap_sample(bytes, rate);
    }
}

struct ResetInSampleGuard;

impl Drop for ResetInSampleGuard {
    fn drop(&mut self) {
        let _ = THREAD_HEAP_STATE.try_with(|cell| {
            let mut st = cell.get();
            st.in_sample = false;
            cell.set(st);
        });
    }
}

#[inline(never)]
fn capture_heap_sample(bytes: usize, rate: usize) {
    let (scaled_objects, scaled_space) = scale_heap_sample(bytes, rate);

    let mut ips = [0usize; MAX_HEAP_STACK_DEPTH];
    let mut sym_addrs = [0usize; MAX_HEAP_STACK_DEPTH];
    let mut len = 0usize;
    backtrace::trace(|frame| {
        if len >= MAX_HEAP_STACK_DEPTH {
            return false;
        }
        ips[len] = frame.ip() as usize;
        sym_addrs[len] = frame.symbol_address() as usize;
        len += 1;
        true
    });
    if len == 0 {
        return;
    }

    // `try_lock` keeps sampling non-blocking and deadlock-free inside `GlobalAlloc`.
    if let Ok(mut guard) = HEAP_SAMPLES.try_lock()
        && let Some(map) = guard.as_mut()
    {
        if let Some(entry) = map.get_mut(&ips[..len]) {
            entry.alloc_objects += scaled_objects;
            entry.alloc_space += scaled_space;
        } else {
            map.insert(
                ips[..len].into(),
                HeapSampleValue {
                    alloc_objects: scaled_objects,
                    alloc_space: scaled_space,
                    symbol_addrs: sym_addrs[..len].into(),
                },
            );
        }
    }
}

/// Scales a sampled allocation of `bytes` under Poisson byte rate `rate` using Go's unbiased
/// `1 / (1 - exp(-bytes / rate))` estimator (`runtime/pprof` `scaleHeapSample`).
fn scale_heap_sample(bytes: usize, rate: usize) -> (f64, f64) {
    if rate <= 1 {
        return (1.0, bytes as f64);
    }
    let ratio = (bytes as f64) / (rate as f64);
    let prob = -(-ratio).exp_m1();
    let scale = if prob > 0.0 { 1.0 / prob } else { 1.0 };
    (scale.max(1.0), ((bytes as f64) * scale).max(bytes as f64))
}

fn next_exponential_bytes(rng: &mut u64, rate: usize) -> u64 {
    let bits = xorshift64_star(rng);
    // Uniform `(0, 1]` with 53 bits of significand precision.
    let u = (((bits >> 11) as f64) + 1.0) * (1.0 / ((1u64 << 53) as f64));
    ((-u.ln()) * (rate as f64)).ceil().max(1.0) as u64
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (x ^ (x >> 31)).max(1)
}

fn xorshift64_star(state: &mut u64) -> u64 {
    let mut x = (*state).max(1);
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

/// Built-in [`Profiler`] session managing `-cpuprofile` and `-memprofile` across a [`crate::Suite`].
pub(crate) struct PprofSession {
    #[cfg(unix)]
    cpu: Option<CpuCollector>,
    mem: Option<MemCollector>,
    // Keep `_lock` as the last field so `cpu` (`ProfilerGuard`) and `mem` are always dropped
    // before `SESSION_LOCK` is released.
    _lock: MutexGuard<'static, ()>,
}

impl PprofSession {
    pub(crate) fn new(cfg: &BenchConfig) -> io::Result<Option<Self>> {
        if cfg.cpu_profile.is_none() && cfg.mem_profile.is_none() {
            return Ok(None);
        }

        #[cfg(not(unix))]
        if cfg.cpu_profile.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "-cpuprofile is only supported on Unix targets; -memprofile is supported on all targets",
            ));
        }

        let lock = SESSION_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        SESSION_GEN.fetch_add(1, Ordering::Relaxed);

        #[cfg(unix)]
        let cpu = cfg
            .cpu_profile
            .as_ref()
            .map(|path| CpuCollector::new(path.clone()));

        let mem = if let Some(path) = &cfg.mem_profile {
            let mut guard = HEAP_SAMPLES
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *guard = Some(HashMap::new());
            Some(MemCollector {
                path: path.clone(),
                rate: cfg.mem_profile_rate,
                started_at: SystemTime::now(),
                active_since: None,
                total_duration: Duration::ZERO,
            })
        } else {
            None
        };

        Ok(Some(Self {
            #[cfg(unix)]
            cpu,
            mem,
            _lock: lock,
        }))
    }
}

impl Drop for PprofSession {
    fn drop(&mut self) {
        HEAP_SAMPLE_RATE.store(0, Ordering::Release);
        #[cfg(unix)]
        {
            self.cpu = None;
        }
        self.mem = None;
        if let Ok(mut guard) = HEAP_SAMPLES.lock() {
            *guard = None;
        }
    }
}

impl Profiler for PprofSession {
    fn start_bench(&mut self, _bench_name: &str) -> io::Result<()> {
        // NOTE: Start the CPU profiler before enabling heap sampling so `pprof`'s internal
        // ring-buffer allocations never pollute `-memprofile`.
        #[cfg(unix)]
        if let Some(cpu) = &mut self.cpu {
            cpu.start_window()?;
        }
        if let Some(mem) = &mut self.mem {
            mem.active_since = Some(Instant::now());
            HEAP_SAMPLE_RATE.store(mem.rate, Ordering::Release);
        }
        Ok(())
    }

    fn stop_bench(&mut self, _bench_name: &str) -> io::Result<()> {
        HEAP_SAMPLE_RATE.store(0, Ordering::Release);
        if let Some(mem) = &mut self.mem
            && let Some(since) = mem.active_since.take()
        {
            mem.total_duration += since.elapsed();
        }
        #[cfg(unix)]
        if let Some(cpu) = &mut self.cpu {
            cpu.stop_window()?;
        }
        Ok(())
    }

    fn finish(&mut self) -> io::Result<()> {
        HEAP_SAMPLE_RATE.store(0, Ordering::Release);
        #[cfg(unix)]
        if let Some(cpu) = &mut self.cpu {
            cpu.write_profile()?;
        }
        if let Some(mem) = &mut self.mem {
            mem.write_profile()?;
        }
        Ok(())
    }
}

/// Chains two [`Profiler`]s so a user-supplied `Suite::with_profiler` still runs alongside
/// `-cpuprofile` / `-memprofile`.
pub(crate) struct CombinedProfiler<'a> {
    pub(crate) builtin: Option<&'a mut PprofSession>,
    pub(crate) custom: Option<&'a mut dyn Profiler>,
}

impl Profiler for CombinedProfiler<'_> {
    fn start_bench(&mut self, bench_name: &str) -> io::Result<()> {
        // Start `custom` before `builtin` so any allocations inside `custom.start_bench` happen
        // before `HEAP_SAMPLE_RATE` is armed, and roll back `custom` if `builtin` fails.
        if let Some(c) = self.custom.as_deref_mut() {
            c.start_bench(bench_name)?;
        }
        if let Some(b) = self.builtin.as_deref_mut()
            && let Err(err) = b.start_bench(bench_name)
        {
            if let Some(c) = self.custom.as_deref_mut() {
                let _ = c.stop_bench(bench_name);
            }
            return Err(err);
        }
        Ok(())
    }

    fn stop_bench(&mut self, bench_name: &str) -> io::Result<()> {
        // Disarm `builtin` first so `custom.stop_bench` allocations are not sampled into
        // `-memprofile`.
        let r1 = if let Some(b) = self.builtin.as_deref_mut() {
            b.stop_bench(bench_name)
        } else {
            Ok(())
        };
        let r2 = if let Some(c) = self.custom.as_deref_mut() {
            c.stop_bench(bench_name)
        } else {
            Ok(())
        };
        r1.and(r2)
    }

    fn finish(&mut self) -> io::Result<()> {
        let r1 = if let Some(c) = self.custom.as_deref_mut() {
            c.finish()
        } else {
            Ok(())
        };
        let r2 = if let Some(b) = self.builtin.as_deref_mut() {
            b.finish()
        } else {
            Ok(())
        };
        r1.and(r2)
    }
}

struct MemCollector {
    path: PathBuf,
    rate: usize,
    started_at: SystemTime,
    active_since: Option<Instant>,
    total_duration: Duration,
}

impl MemCollector {
    fn write_profile(&mut self) -> io::Result<()> {
        let _suppress = SuppressHeapGuard::enter();
        let samples = HEAP_SAMPLES
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            .unwrap_or_default();

        let mut builder = ProfileBuilder::new(
            &[("alloc_objects", "count"), ("alloc_space", "bytes")],
            ("space", "bytes"),
            self.rate as i64,
            "alloc_space",
            self.started_at,
            self.total_duration,
        );

        let mut resolved_cache: HashMap<usize, Vec<ResolvedSymbol>> = HashMap::new();

        for (ips, val) in samples {
            let mut frame_symbols: Vec<(u64, Vec<ResolvedSymbol>)> = Vec::with_capacity(ips.len());
            for (idx, &ip) in ips.iter().enumerate() {
                if ip == 0 {
                    continue;
                }
                // NOTE: `backtrace::resolve` internally subtracts 1 from return addresses, so pass
                // the raw `ip` for symbolication while storing `ip - 1` as `Location.address`.
                let syms = resolved_cache
                    .entry(ip)
                    .or_insert_with(|| resolve_return_ip(ip, val.symbol_addrs.get(idx).copied()))
                    .clone();
                frame_symbols.push((ip.saturating_sub(1) as u64, syms));
            }

            strip_internal_heap_frames(&mut frame_symbols);

            let mut loc_ids = Vec::with_capacity(frame_symbols.len());
            for (addr, syms) in &frame_symbols {
                loc_ids.push(builder.intern_location(*addr, syms));
            }
            let objects = (val.alloc_objects.round() as i64).max(1);
            let space = (val.alloc_space.round() as i64).max(1);
            builder.add_sample(loc_ids, vec![objects, space], &[]);
        }

        builder.write_gzipped(&self.path)
    }
}

/// Strips internal unwinder, `pprof_support`, `CountingAlloc`, and `alloc::alloc` wrapper frames
/// from the leaf of a heap sample while preserving user functions into which `alloc::alloc::*`
/// was inlined (such as `Box::new` under `--release` with `debug = 1`).
fn strip_internal_heap_frames(frame_symbols: &mut Vec<(u64, Vec<ResolvedSymbol>)>) {
    let mut idx = 0usize;
    let mut skip = 0usize;
    while idx < frame_symbols.len() {
        let syms = &frame_symbols[idx].1;
        if syms.is_empty() {
            idx += 1;
        } else if syms.iter().all(is_internal_heap_frame) {
            idx += 1;
            skip = idx;
        } else {
            if syms.iter().any(is_internal_heap_frame) {
                skip = idx;
            }
            break;
        }
    }
    if skip > 0 && skip < frame_symbols.len() {
        frame_symbols.drain(..skip);
    }
    if let Some((_, first_syms)) = frame_symbols.first_mut() {
        let inline_skip = first_syms
            .iter()
            .position(|s| !is_internal_heap_frame(s))
            .unwrap_or(0);
        if inline_skip > 0 && inline_skip < first_syms.len() {
            first_syms.drain(..inline_skip);
        }
    }
}

fn resolve_return_ip(return_ip: usize, fallback_sym_addr: Option<usize>) -> Vec<ResolvedSymbol> {
    let mut out = Vec::new();
    let collect = |symbol: &backtrace::Symbol, dst: &mut Vec<ResolvedSymbol>| {
        let (name, system_name) = match symbol.name() {
            Some(sym_name) => (
                format!("{sym_name:#}"),
                String::from_utf8_lossy(sym_name.as_bytes()).into_owned(),
            ),
            None => (String::new(), String::new()),
        };
        let filename = symbol
            .filename()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
        let lineno = symbol.lineno().unwrap_or(0);
        if !name.is_empty() || !filename.is_empty() {
            dst.push(ResolvedSymbol {
                name,
                system_name,
                filename,
                lineno,
            });
        }
    };

    backtrace::resolve(return_ip as *mut c_void, |sym| collect(sym, &mut out));
    if out.is_empty()
        && let Some(sym_addr) = fallback_sym_addr
        && sym_addr != 0
    {
        // `backtrace::resolve` subtracts 1 internally, so pass `sym_addr + 1` to land on the
        // function's entry byte rather than the last byte of the preceding function.
        backtrace::resolve(sym_addr.saturating_add(1) as *mut c_void, |sym| {
            collect(sym, &mut out);
        });
    }
    out
}

fn is_internal_heap_frame(sym: &ResolvedSymbol) -> bool {
    let n = sym.name.as_str();
    // Strip leading `<` / `&` so Rust v0-mangled inherent and trait impls such as
    // `<alloc::alloc::Global>::alloc_impl_runtime` and `<std::alloc::System>::alloc_impl` match
    // alongside free functions like `alloc::alloc::alloc`.
    let norm = n.trim_start_matches(['<', '&']);
    let trimmed = n.trim_start_matches('_');
    norm.starts_with("backtrace::")
        || norm.starts_with("benchfmt::pprof_support::")
        || norm.starts_with("benchfmt::alloc::")
        || n.starts_with("__rustc::__rust_")
        || trimmed.starts_with("rust_alloc")
        || trimmed.starts_with("rust_realloc")
        || trimmed.starts_with("rg_alloc")
        || trimmed.starts_with("rg_realloc")
        || norm.starts_with("alloc::alloc::")
        || norm.starts_with("core::alloc::")
        || norm.starts_with("std::alloc::")
}

#[cfg(unix)]
struct CpuCollector {
    path: PathBuf,
    builder: pprof::ProfilerGuardBuilder,
    active: Option<(pprof::ProfilerGuard<'static>, Instant)>,
    reports: Vec<pprof::UnresolvedReport>,
    started_at: SystemTime,
    total_duration: Duration,
}

#[cfg(unix)]
impl CpuCollector {
    fn new(path: PathBuf) -> Self {
        let builder = pprof::ProfilerGuardBuilder::default().frequency(CPU_SAMPLE_FREQUENCY_HZ);
        #[cfg(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64",
            target_arch = "loongarch64"
        ))]
        let builder = builder.blocklist(&[
            "libc",
            "libgcc",
            "pthread",
            "vdso",
            "ld-linux",
            "ld-musl",
            "ld.so",
            "libsystem",
            "libdyld",
            "dyld",
            "libunwind",
        ]);

        Self {
            path,
            builder,
            active: None,
            reports: Vec::new(),
            started_at: SystemTime::now(),
            total_duration: Duration::ZERO,
        }
    }

    fn start_window(&mut self) -> io::Result<()> {
        let _suppress = SuppressHeapGuard::enter();
        let guard = self
            .builder
            .clone()
            .build()
            .map_err(|e| io::Error::other(format!("failed starting CPU profiler: {e}")))?;
        self.active = Some((guard, Instant::now()));
        Ok(())
    }

    fn stop_window(&mut self) -> io::Result<()> {
        let Some((guard, started)) = self.active.take() else {
            return Ok(());
        };
        self.total_duration += started.elapsed();
        // NOTE: Harvest unresolved frames cheaply between calibration steps and keep `in_sample`
        // set while dropping `guard` so internal allocations stay hidden from `CountingAlloc`.
        let _suppress = SuppressHeapGuard::enter();
        let report = guard
            .report()
            .build_unresolved()
            .map_err(|e| io::Error::other(format!("failed harvesting CPU samples: {e}")))?;
        self.reports.push(report);
        drop(guard);
        Ok(())
    }

    fn write_profile(&mut self) -> io::Result<()> {
        let _suppress = SuppressHeapGuard::enter();
        // Aggregate raw `UnresolvedFrames` across all calibration steps and epochs before
        // symbolizing, preserving runtime instruction pointers (`frame.ip()`) that
        // `pprof::Frames::from` discards when compiled with `-C debuginfo=1` (line tables only).
        let mut raw_samples = HashMap::new();
        for report in self.reports.drain(..) {
            for (unresolved, count) in report.data {
                if count > 0 {
                    *raw_samples.entry(unresolved).or_insert(0i64) += count as i64;
                }
            }
        }

        let period_ns = 1_000_000_000i64 / i64::from(CPU_SAMPLE_FREQUENCY_HZ);
        let mut builder = ProfileBuilder::new(
            &[("samples", "count"), ("cpu", "nanoseconds")],
            ("cpu", "nanoseconds"),
            period_ns,
            "cpu",
            self.started_at,
            self.total_duration,
        );

        let mut resolved_cache: HashMap<usize, Vec<ResolvedSymbol>> = HashMap::new();

        for (unresolved, count) in raw_samples {
            if count <= 0 {
                continue;
            }
            let mut frame_symbols = Vec::with_capacity(unresolved.frames.len());
            let mut frame_iter = unresolved.frames.iter();
            let mut next_is_interrupted_pc = false;
            while let Some(frame) = frame_iter.next() {
                #[cfg(all(
                    target_os = "macos",
                    any(
                        target_arch = "x86_64",
                        target_arch = "aarch64",
                        target_arch = "riscv64",
                        target_arch = "loongarch64"
                    )
                ))]
                let (ip, sym_addr) = (frame.ip, 0usize);
                #[cfg(not(all(
                    target_os = "macos",
                    any(
                        target_arch = "x86_64",
                        target_arch = "aarch64",
                        target_arch = "riscv64",
                        target_arch = "loongarch64"
                    )
                )))]
                let (ip, sym_addr) = (frame.ip() as usize, frame.symbol_address() as usize);
                if ip == 0 {
                    continue;
                }
                // The single frame unwound across the OS signal trampoline (`__restore_rt`) holds
                // the exact interrupted `PC` rather than a post-`call` return address; add 1 before
                // `backtrace::resolve` (which subtracts 1 internally) and store `ip` directly.
                let exact_pc = std::mem::take(&mut next_is_interrupted_pc);
                let lookup_ip = if exact_pc { ip.saturating_add(1) } else { ip };
                let syms = resolved_cache
                    .entry(lookup_ip)
                    .or_insert_with(|| {
                        resolve_return_ip(lookup_ip, (sym_addr != 0).then_some(sym_addr))
                    })
                    .clone();

                // Strip `pprof`'s `SIGPROF` handler, any unwinder frames captured above it, and
                // the OS signal return trampoline frame immediately below it (`__restore_rt`).
                if syms.iter().any(|sym| {
                    matches!(
                        sym.name.as_str(),
                        "perf_signal_handler" | "_perf_signal_handler"
                    ) || matches!(
                        sym.system_name.as_str(),
                        "perf_signal_handler" | "_perf_signal_handler"
                    )
                }) {
                    frame_symbols.clear();
                    frame_iter.next();
                    next_is_interrupted_pc = true;
                    continue;
                }

                let loc_addr = if exact_pc {
                    ip as u64
                } else {
                    ip.saturating_sub(1) as u64
                };
                frame_symbols.push((loc_addr, syms));
            }

            let mut loc_ids = Vec::with_capacity(frame_symbols.len());
            for (addr, syms) in &frame_symbols {
                loc_ids.push(builder.intern_location(*addr, syms));
            }

            let name_len = unresolved
                .thread_name_length
                .min(unresolved.thread_name.len());
            let thread_name = String::from_utf8_lossy(&unresolved.thread_name[..name_len]);
            let thread_label = if !thread_name.is_empty() {
                thread_name.into_owned()
            } else {
                unresolved.thread_id.to_string()
            };
            builder.add_sample(
                loc_ids,
                vec![count, count.saturating_mul(period_ns)],
                &[("thread", thread_label.as_str())],
            );
        }

        builder.write_gzipped(&self.path)
    }
}

struct SuppressHeapGuard {
    prev: bool,
}

impl SuppressHeapGuard {
    fn enter() -> Self {
        let prev = THREAD_HEAP_STATE
            .try_with(|cell| {
                let mut st = cell.get();
                let was = st.in_sample;
                st.in_sample = true;
                cell.set(st);
                was
            })
            .unwrap_or(false);
        Self { prev }
    }
}

impl Drop for SuppressHeapGuard {
    fn drop(&mut self) {
        let prev = self.prev;
        let _ = THREAD_HEAP_STATE.try_with(|cell| {
            let mut st = cell.get();
            st.in_sample = prev;
            cell.set(st);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_heap_sample_preserves_fractional_expectations() {
        assert_eq!(scale_heap_sample(2048, 1), (1.0, 2048.0));

        // At `bytes = 576_000` and `rate = 524_288`, `scale` is ~1.49997; keeping `f64` avoids
        // rounding every individual sample down to `1` (-33% bias).
        let (objs, space) = scale_heap_sample(576_000, 524_288);
        assert!((objs - 1.5).abs() < 0.01, "objs={objs}");
        assert!((space - 576_000.0 * objs).abs() < 1.0, "space={space}");
    }

    #[test]
    fn strip_internal_heap_frames_keeps_user_frame_with_inlined_alloc() {
        let mk = |name: &str| ResolvedSymbol {
            name: name.to_owned(),
            system_name: name.to_owned(),
            filename: "src/lib.rs".to_owned(),
            lineno: 1,
        };
        let mut frames = vec![
            // Symbolicated leaf unwinder frame followed by an unsymbolicated `_Unwind_Backtrace`
            // frame in stripped `libgcc_s`.
            (0x0fff, vec![mk("backtrace::backtrace::libunwind::trace")]),
            (0x1000, Vec::new()),
            // Internal wrapper frames.
            (
                0x1001,
                vec![mk("benchfmt::pprof_support::capture_heap_sample")],
            ),
            (
                0x1002,
                vec![mk(
                    "<benchfmt::alloc::CountingAlloc as core::alloc::global::GlobalAlloc>::alloc",
                )],
            ),
            // User physical frame where `<alloc::alloc::Global>::alloc_impl_runtime` and
            // `alloc::alloc::alloc` were inlined via `Box::new`.
            (
                0x2000,
                vec![
                    mk("<std::alloc::System>::alloc_impl"),
                    mk("<alloc::alloc::Global>::alloc_impl_runtime"),
                    mk("alloc::alloc::alloc"),
                    mk("alloc::alloc::exchange_malloc"),
                    mk("alloc::boxed::Box<T>::new"),
                    mk("my_crate::my_box_alloc_fn"),
                ],
            ),
            (0x3000, vec![mk("my_crate::caller")]),
        ];

        strip_internal_heap_frames(&mut frames);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].0, 0x2000);
        assert_eq!(frames[0].1.len(), 2);
        assert_eq!(frames[0].1[0].name, "alloc::boxed::Box<T>::new");
        assert_eq!(frames[0].1[1].name, "my_crate::my_box_alloc_fn");

        // Unsymbolicated leaf frame directly preceding a mixed inline user frame is drained.
        let mut unsym_before_mixed = vec![
            (0x1000, Vec::new()),
            (
                0x2000,
                vec![
                    mk("<alloc::alloc::Global>::alloc_impl_runtime"),
                    mk("my_crate::f"),
                ],
            ),
        ];
        strip_internal_heap_frames(&mut unsym_before_mixed);
        assert_eq!(unsym_before_mixed.len(), 1);
        assert_eq!(unsym_before_mixed[0].0, 0x2000);
        assert_eq!(unsym_before_mixed[0].1.len(), 1);
        assert_eq!(unsym_before_mixed[0].1[0].name, "my_crate::f");

        // Leading unsymbolicated user frames are preserved when not followed by internal frames.
        let mut user_unsymbolicated = vec![(0x4000, Vec::new()), (0x5000, vec![mk("my_crate::f")])];
        strip_internal_heap_frames(&mut user_unsymbolicated);
        assert_eq!(user_unsymbolicated.len(), 2);

        // All-internal stacks are never drained to empty.
        let mut all_internal = vec![(
            0x1001,
            vec![mk("benchfmt::pprof_support::capture_heap_sample")],
        )];
        strip_internal_heap_frames(&mut all_internal);
        assert_eq!(all_internal.len(), 1);
    }

    #[inline(never)]
    fn interrupted_entry_probe(x: u64) -> u64 {
        std::hint::black_box(x).wrapping_add(1)
    }

    #[test]
    fn exact_pc_plus_one_resolves_function_entry_byte() {
        assert_eq!(interrupted_entry_probe(41), 42);
        let entry_pc = interrupted_entry_probe as *const () as usize;
        let syms = resolve_return_ip(entry_pc.saturating_add(1), None);
        assert!(
            syms.iter()
                .any(|s| s.name.contains("interrupted_entry_probe")),
            "expected function entry byte to resolve to interrupted_entry_probe, got {syms:?}"
        );
    }
}
