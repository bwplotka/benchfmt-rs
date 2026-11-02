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

/// Active Poisson mean byte interval (`0` means heap sampling is disabled).
static HEAP_SAMPLE_RATE: AtomicUsize = AtomicUsize::new(0);
static RNG_SEED_COUNTER: AtomicU64 = AtomicU64::new(0x9e37_79b9_7f4a_7c15);

#[derive(Default)]
struct HeapSampleValue {
    alloc_objects: i64,
    alloc_space: i64,
    symbol_addrs: Box<[usize]>,
}

static HEAP_SAMPLES: Mutex<Option<HashMap<Box<[usize]>, HeapSampleValue>>> = Mutex::new(None);

#[derive(Clone, Copy)]
struct ThreadHeapState {
    in_sample: bool,
    initialized: bool,
    bytes_until_sample: u64,
    rng: u64,
}

thread_local! {
    // `const` initialization avoids any heap allocation when `CountingAlloc` inspects TLS.
    static THREAD_HEAP_STATE: Cell<ThreadHeapState> = const {
        Cell::new(ThreadHeapState {
            in_sample: false,
            initialized: false,
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
                st.bytes_until_sample = if rate <= 1 {
                    0
                } else {
                    next_exponential_bytes(&mut st.rng, rate)
                };
                st.initialized = true;
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
            entry.alloc_objects = entry.alloc_objects.saturating_add(scaled_objects);
            entry.alloc_space = entry.alloc_space.saturating_add(scaled_space);
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
fn scale_heap_sample(bytes: usize, rate: usize) -> (i64, i64) {
    if rate <= 1 {
        return (1, bytes as i64);
    }
    let ratio = (bytes as f64) / (rate as f64);
    let prob = -(-ratio).exp_m1();
    let scale = if prob > 0.0 { 1.0 / prob } else { 1.0 };
    let objects = scale.round() as i64;
    let space = ((bytes as f64) * scale).round() as i64;
    (objects.max(1), space.max(bytes as i64))
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
    _lock: MutexGuard<'static, ()>,
    #[cfg(unix)]
    cpu: Option<CpuCollector>,
    mem: Option<MemCollector>,
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
            _lock: lock,
            #[cfg(unix)]
            cpu,
            mem,
        }))
    }
}

impl Drop for PprofSession {
    fn drop(&mut self) {
        HEAP_SAMPLE_RATE.store(0, Ordering::Relaxed);
        if let Ok(mut guard) = HEAP_SAMPLES.lock() {
            *guard = None;
        }
    }
}

impl Profiler for PprofSession {
    fn start_bench(&mut self, _bench_name: &str) -> io::Result<()> {
        // NOTE: Start the CPU profiler before enabling heap sampling so `pprof`'s internal DWARF
        // and ring-buffer allocations never pollute `-memprofile`.
        #[cfg(unix)]
        if let Some(cpu) = &mut self.cpu {
            cpu.start_window()?;
        }
        if let Some(mem) = &mut self.mem {
            mem.active_since = Some(Instant::now());
            HEAP_SAMPLE_RATE.store(mem.rate, Ordering::Relaxed);
        }
        Ok(())
    }

    fn stop_bench(&mut self, _bench_name: &str) -> io::Result<()> {
        HEAP_SAMPLE_RATE.store(0, Ordering::Relaxed);
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
        HEAP_SAMPLE_RATE.store(0, Ordering::Relaxed);
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
        if let Some(b) = self.builtin.as_deref_mut() {
            b.start_bench(bench_name)?;
        }
        if let Some(c) = self.custom.as_deref_mut() {
            c.start_bench(bench_name)?;
        }
        Ok(())
    }

    fn stop_bench(&mut self, bench_name: &str) -> io::Result<()> {
        let r1 = if let Some(c) = self.custom.as_deref_mut() {
            c.stop_bench(bench_name)
        } else {
            Ok(())
        };
        let r2 = if let Some(b) = self.builtin.as_deref_mut() {
            b.stop_bench(bench_name)
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
                // All frames captured inside `capture_heap_sample` are return addresses, so
                // `ip - 1` points inside the calling instruction for accurate line resolution.
                let lookup_ip = ip.saturating_sub(1);
                let syms = resolved_cache
                    .entry(lookup_ip)
                    .or_insert_with(|| resolve_ip(lookup_ip, val.symbol_addrs.get(idx).copied()))
                    .clone();
                frame_symbols.push((lookup_ip as u64, syms));
            }

            // Strip internal allocator-wrapper and backtrace capture frames from the leaf so the
            // top of `go tool pprof` points at the benchmark's allocation site.
            let skip = frame_symbols
                .iter()
                .position(|(_, syms)| !syms.iter().any(is_internal_heap_frame))
                .unwrap_or(0);
            let trimmed = &frame_symbols[skip..];

            let mut loc_ids = Vec::with_capacity(trimmed.len());
            for (addr, syms) in trimmed {
                loc_ids.push(builder.intern_location(*addr, syms));
            }
            builder.add_sample(loc_ids, vec![val.alloc_objects, val.alloc_space], &[]);
        }

        builder.write_gzipped(&self.path)
    }
}

fn resolve_ip(lookup_ip: usize, fallback_sym_addr: Option<usize>) -> Vec<ResolvedSymbol> {
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

    backtrace::resolve(lookup_ip as *mut c_void, |sym| collect(sym, &mut out));
    if out.is_empty()
        && let Some(sym_addr) = fallback_sym_addr
        && sym_addr != 0
    {
        backtrace::resolve(sym_addr as *mut c_void, |sym| collect(sym, &mut out));
    }
    out
}

fn is_internal_heap_frame(sym: &ResolvedSymbol) -> bool {
    let n = sym.name.as_str();
    n.starts_with("backtrace::")
        || n.starts_with("benchfmt::pprof_support::")
        || n.starts_with("benchfmt::alloc::")
        || n.starts_with("<benchfmt::alloc::")
        || n.starts_with("__rustc::__rust_")
        || n.starts_with("alloc::alloc::")
}

#[cfg(unix)]
struct CpuCollector {
    path: PathBuf,
    builder: pprof::ProfilerGuardBuilder,
    active: Option<(pprof::ProfilerGuard<'static>, Instant)>,
    samples: HashMap<pprof::Frames, i64>,
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
        let builder = builder.blocklist(&["libc", "libgcc", "pthread", "vdso"]);

        Self {
            path,
            builder,
            active: None,
            samples: HashMap::new(),
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
        // NOTE: Keep `in_sample` set while building the report and dropping `guard` so any
        // internal allocations stay hidden from `CountingAlloc`.
        let _suppress = SuppressHeapGuard::enter();
        let report = guard
            .report()
            .build()
            .map_err(|e| io::Error::other(format!("failed harvesting CPU samples: {e}")))?;
        for (resolved, count) in report.data {
            if count > 0 {
                *self.samples.entry(resolved).or_insert(0) += count as i64;
            }
        }
        drop(guard);
        Ok(())
    }

    fn write_profile(&mut self) -> io::Result<()> {
        let _suppress = SuppressHeapGuard::enter();
        let period_ns = 1_000_000_000i64 / i64::from(CPU_SAMPLE_FREQUENCY_HZ);
        let mut builder = ProfileBuilder::new(
            &[("samples", "count"), ("cpu", "nanoseconds")],
            ("cpu", "nanoseconds"),
            period_ns,
            "cpu",
            self.started_at,
            self.total_duration,
        );

        for (frames, count) in &self.samples {
            if *count <= 0 {
                continue;
            }
            let mut loc_ids = Vec::with_capacity(frames.frames.len());
            for inline_syms in &frames.frames {
                let addr = inline_syms
                    .last()
                    .and_then(|s| s.addr)
                    .map(|p| p as usize as u64)
                    .unwrap_or(0);
                let syms: Vec<ResolvedSymbol> = inline_syms
                    .iter()
                    .map(|s| ResolvedSymbol {
                        name: s.name(),
                        system_name: s.sys_name().into_owned(),
                        filename: s
                            .filename
                            .as_ref()
                            .map(|p| p.to_string_lossy().into_owned())
                            .unwrap_or_default(),
                        lineno: s.lineno(),
                    })
                    .collect();
                loc_ids.push(builder.intern_location(addr, &syms));
            }
            let thread_label = frames.thread_name_or_id();
            builder.add_sample(
                loc_ids,
                vec![*count, count.saturating_mul(period_ns)],
                &[("thread", thread_label.as_str())],
            );
        }

        builder.write_gzipped(&self.path)
    }
}

#[cfg(unix)]
struct SuppressHeapGuard {
    prev: bool,
}

#[cfg(unix)]
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

#[cfg(unix)]
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
