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

//! End-to-end integration tests for scoped CPU and sampled heap pprof profiling.
//!
//! Kept in a dedicated integration test binary and serialized via `PPROF_TEST_LOCK` because
//! `-memprofile` and `-cpuprofile` intentionally sample across all active threads in the process
//! during `Bencher::keep_running`.

use std::fs;
use std::hint::black_box;
use std::io::{self, Read};
use std::sync::Mutex;
#[cfg(unix)]
use std::time::{Duration, Instant};

use benchfmt::{BenchConfig, BenchTime, CountingAlloc, Profiler, Suite};

#[global_allocator]
static TEST_ALLOC: CountingAlloc = CountingAlloc::new();

static PPROF_TEST_LOCK: Mutex<()> = Mutex::new(());

#[inline(never)]
fn unmeasured_setup_alloc_helper(size: usize) -> Vec<u8> {
    vec![9u8; size]
}

#[inline(never)]
fn tracked_heap_alloc_helper(size: usize) -> Vec<u8> {
    vec![42u8; size]
}

#[inline(never)]
fn tracked_box_alloc_helper(seed: u8) -> Box<[u8; 256]> {
    // Under `--release` with `-C debuginfo=1`, `Box::new` inlines `alloc::alloc::alloc`
    // directly into this function's physical stack frame.
    Box::new([seed; 256])
}

#[cfg(unix)]
#[inline(never)]
fn cpu_burn_leaf(rounds: u64) -> u64 {
    let mut acc = 1u64;
    for i in 0..rounds {
        acc = black_box(acc.wrapping_mul(6364136223846793005).wrapping_add(i | 1));
    }
    acc
}

#[cfg(unix)]
#[inline(never)]
fn cpu_burn_inner(rounds: u64) -> u64 {
    // Non-leaf wrapper so macOS frame-pointer unwinding (`fp -> lr`) records a return address
    // inside `cpu_burn_helper` while `cpu_burn_leaf` executes.
    let mut acc = 0u64;
    for _ in 0..2 {
        acc ^= black_box(cpu_burn_leaf(rounds));
    }
    acc
}

#[cfg(unix)]
#[inline(never)]
fn cpu_burn_helper(rounds: u64) -> u64 {
    let mut acc = black_box(cpu_burn_inner(rounds));
    for i in 0..rounds {
        acc = black_box(acc.wrapping_mul(6364136223846793005).wrapping_add(i | 1));
    }
    acc
}

/// Minimal protobuf wire reader extracting `(values[0], values[1])` across all `Profile.sample`
/// messages (field 2, packed `repeated int64 value` at field 2) for deterministic scaling tests.
fn sum_pprof_sample_pairs(mut buf: &[u8]) -> (i64, i64) {
    fn read_varint(b: &mut &[u8]) -> u64 {
        let mut out = 0u64;
        let mut shift = 0u32;
        while let Some((&byte, rest)) = b.split_first() {
            *b = rest;
            out |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        out
    }
    fn skip_field(b: &mut &[u8], wire: u64) {
        match wire {
            0 => {
                let _ = read_varint(b);
            }
            1 => *b = &b[8.min(b.len())..],
            2 => {
                let len = read_varint(b) as usize;
                *b = &b[len.min(b.len())..];
            }
            5 => *b = &b[4.min(b.len())..],
            _ => *b = &[],
        }
    }

    let mut total_v0 = 0i64;
    let mut total_v1 = 0i64;
    while !buf.is_empty() {
        let tag = read_varint(&mut buf);
        let (field, wire) = (tag >> 3, tag & 7);
        if field == 2 && wire == 2 {
            let len = read_varint(&mut buf) as usize;
            let mut sample = &buf[..len];
            buf = &buf[len..];
            let mut vals = Vec::new();
            while !sample.is_empty() {
                let stag = read_varint(&mut sample);
                let (sfield, swire) = (stag >> 3, stag & 7);
                if sfield == 2 && swire == 2 {
                    let vlen = read_varint(&mut sample) as usize;
                    let mut packed = &sample[..vlen];
                    sample = &sample[vlen..];
                    while !packed.is_empty() {
                        vals.push(read_varint(&mut packed) as i64);
                    }
                } else if sfield == 2 && swire == 0 {
                    vals.push(read_varint(&mut sample) as i64);
                } else {
                    skip_field(&mut sample, swire);
                }
            }
            if vals.len() >= 2 {
                total_v0 += vals[0];
                total_v1 += vals[1];
            }
        } else {
            skip_field(&mut buf, wire);
        }
    }
    (total_v0, total_v1)
}

#[test]
fn writes_scoped_mem_and_cpu_pprof_profiles() {
    let _serial = PPROF_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    #[derive(Default)]
    struct CustomCompanionProfiler {
        events: Vec<String>,
        internal_bufs: Vec<Vec<u8>>,
    }
    impl Profiler for CustomCompanionProfiler {
        fn start_bench(&mut self, bench_name: &str) -> io::Result<()> {
            self.internal_bufs.push(vec![1u8; 4096]);
            self.events.push(format!("start:{bench_name}"));
            Ok(())
        }
        fn stop_bench(&mut self, bench_name: &str) -> io::Result<()> {
            self.internal_bufs.push(vec![2u8; 4096]);
            self.events.push(format!("stop:{bench_name}"));
            Ok(())
        }
        fn finish(&mut self) -> io::Result<()> {
            self.events.push("finish".to_owned());
            Ok(())
        }
    }

    let dir = std::env::temp_dir().join(format!("benchfmt_pprof_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let mem_path = dir.join("mem.pb.gz");
    #[cfg(unix)]
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

    let mut companion = CustomCompanionProfiler::default();
    let mut out = Vec::new();
    Suite::with_config("pprof_test", cfg)
        .with_alloc(&TEST_ALLOC)
        .with_profiler(&mut companion)
        .bench("BurnAndAlloc", |b| {
            // Unmeasured setup allocation must not appear in `B/op` or `mem.pb.gz`.
            let unmeasured_setup = unmeasured_setup_alloc_helper(65_536);
            black_box(&unmeasured_setup);
            while b.keep_running() {
                black_box(tracked_heap_alloc_helper(2048));
                black_box(tracked_box_alloc_helper(7));
                #[cfg(unix)]
                {
                    // Burn both fixed CPU work and ~4ms wall-clock per iteration (~160ms across
                    // 40 iterations) so 99 Hz `ITIMER_PROF` reliably captures samples even under
                    // `--release` or brief VM preemption.
                    black_box(cpu_burn_helper(10_000));
                    let until = Instant::now() + Duration::from_millis(4);
                    while Instant::now() < until {
                        black_box(cpu_burn_helper(2_000));
                    }
                }
            }
        })
        .run_to(&mut out)
        .unwrap();

    assert_eq!(
        companion.events,
        vec!["start:BurnAndAlloc", "stop:BurnAndAlloc", "finish"]
    );
    let text = String::from_utf8(out).unwrap();
    assert!(
        text.contains("    2304 B/op\t       2 allocs/op"),
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
        "expected symbolicated Vec helper in memprofile"
    );
    assert!(
        mem_lossy.contains("tracked_box_alloc_helper"),
        "expected inlined Box::new helper in memprofile"
    );
    assert!(
        !mem_lossy.contains("unmeasured_setup_alloc_helper"),
        "setup allocations outside keep_running leaked into memprofile"
    );
    assert!(
        !mem_lossy.contains("capture_heap_sample") && !mem_lossy.contains("alloc_impl"),
        "internal heap sampler or Global/System allocator frames leaked into memprofile"
    );
    let (mem_objs, mem_bytes) = sum_pprof_sample_pairs(&mem_raw);
    assert_eq!(
        (mem_objs, mem_bytes),
        (80, 40 * (2048 + 256)),
        "companion profiler or setup allocations leaked into memprofile totals"
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
        assert!(
            !cpu_lossy.contains("perf_signal_handler"),
            "SIGPROF signal handler frame leaked into cpuprofile"
        );
    }

    #[cfg(not(unix))]
    {
        let err = Suite::with_config(
            "pprof_unsupported_cpu",
            BenchConfig {
                benchtime: BenchTime::Iterations(1),
                cpu_profile: Some(dir.join("cpu.pb.gz")),
                ..BenchConfig::default()
            },
        )
        .bench("Noop", |b| b.iter(|| 1))
        .run_to(Vec::new())
        .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Unsupported);
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
        assert!(
            mem_stdout.contains("tracked_box_alloc_helper"),
            "go tool pprof -top missing tracked_box_alloc_helper:\n{mem_stdout}"
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

            #[cfg(target_os = "linux")]
            {
                let cpu_disasm = std::process::Command::new("go")
                    .args([
                        "tool",
                        "pprof",
                        "-disasm",
                        "cpu_burn_helper",
                        cpu_path.to_str().unwrap(),
                    ])
                    .output()
                    .unwrap();
                let disasm_stdout = String::from_utf8_lossy(&cpu_disasm.stdout);
                let disasm_stderr = String::from_utf8_lossy(&cpu_disasm.stderr);
                assert!(
                    cpu_disasm.status.success() && disasm_stdout.contains("cpu_burn_helper"),
                    "go tool pprof -disasm failed (PIE SVMA->AVMA mapping mismatch):\nstdout:\n{disasm_stdout}\nstderr:\n{disasm_stderr}"
                );
            }
        }
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn poisson_memprofile_sampling_scales_allocations_and_resets_across_rates() {
    let _serial = PPROF_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let dir = std::env::temp_dir().join(format!("benchfmt_pprof_poisson_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let huge_rate_path = dir.join("huge_rate_mem.pb.gz");
    let mem_path = dir.join("poisson_mem.pb.gz");

    // Run a huge `mem_profile_rate` first on this thread so TLS `bytes_until_sample` would be
    // megabytes high if `PprofSession` failed to reset per-thread sampling state across runs.
    Suite::with_config(
        "pprof_huge_rate",
        BenchConfig {
            benchtime: BenchTime::Iterations(4),
            mem_profile: Some(huge_rate_path),
            mem_profile_rate: 64 * 1024 * 1024,
            ..BenchConfig::default()
        },
    )
    .with_alloc(&TEST_ALLOC)
    .bench("WarmTls", |b| {
        while b.keep_running() {
            black_box(tracked_heap_alloc_helper(256));
        }
    })
    .run_to(Vec::new())
    .unwrap();

    let cfg = BenchConfig {
        benchtime: BenchTime::Iterations(4_000),
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
    // True allocations are 4,000 objects and 4,096,000 bytes (~242 raw samples scaled ~16.5x).
    let (scaled_objs, scaled_bytes) = sum_pprof_sample_pairs(&mem_raw);
    assert!(
        (2_000..=6_000).contains(&scaled_objs),
        "expected scaled alloc_objects near 4000, got {scaled_objs}"
    );
    assert!(
        (2_000_000..=6_000_000).contains(&scaled_bytes),
        "expected scaled alloc_space near 4096000, got {scaled_bytes}"
    );
    let _ = fs::remove_dir_all(&dir);
}
