# benchfmt

> NOTE(@bwplotka): 👋 I'm coming from the Go ecosystem and I can't live without Go's [`benchstat`-- it's too amazing](https://www.bwplotka.dev/2024/go-microbenchmarks-benchstat/). This tooling brings [`golang.org/x/perf/benchfmt`](https://pkg.go.dev/golang.org/x/perf/benchfmt) experience to Rust!

Zero-dependency Rust benchmark runner, heap allocation counter, and Criterion/`libtest` converter emitting Go's [Benchmark Data Format](https://go.googlesource.com/proposal/+/master/design/14313-benchmark-format.md) for direct analysis with [`benchstat`](https://pkg.go.dev/golang.org/x/perf/cmd/benchstat).

## Why this exists

Comparing Rust changes across commits or benchmarking Rust implementations side by side against Go code hits three practical gaps:

1. `benchstat` expects Go's `benchfmt` line format with `ns/op`, decimal `MB/s`, `B/op`, `allocs/op`, independent `-count` epochs for Mann-Whitney U tests, and `/key=value` sub-benchmark dimensions.
2. Criterion measures wall-clock time and throughput, leaving out per-operation heap bytes and allocation counts.
3. Go's `testing.B` calibrates an iteration count `N` for a fixed `-benchtime` such as `2s` and repeats `-count` independent epochs, whereas Criterion defaults to 100 linear-ramp samples.

## When to use what

- Use `Suite` and `CountingAlloc` in a `harness = false` benchmark binary when writing new microbenchmarks, catching `B/op` and `allocs/op` regressions, comparing Rust and Go implementations on the same machine, or keeping `cargo bench` compile times near zero.
- Use the `benchfmt criterion` CLI when a repository already has Criterion benchmarks and you want `benchstat` tables across commits or `/key=value` dimensions without touching benchmark code, accepting that Criterion omits heap allocations and that the converter approximates epochs by bucketing Criterion's sample array.
- Use `benchfmt libtest` for quick one-off `benchfmt` formatting of `cargo +nightly bench` output. Because `libtest` prints a single summary line per benchmark, one run does not give `benchstat` multiple epochs for p-values unless you concatenate repeated runs.
- Use standalone Criterion when you want built-in HTML and SVG violin plots, async executor adapters, or local baseline comparisons without installing Go's `benchstat`.

## Suite vs. Criterion

| Aspect | `benchfmt::Suite` | `benchfmt criterion` converter | Standalone Criterion |
| --- | --- | --- | --- |
| External Rust crates | `0`, `std` only | `0` in CLI, Criterion in `dev-dependencies` | ~40+ crates |
| Sampling model | `-count` independent calibrated epochs for fixed `-benchtime`, e.g. `6 x 2s` | 100 linear-ramp samples bucketed into `--epochs`, `6` by default | 100 linear-ramp or flat samples |
| Fit for `benchstat` Mann-Whitney U | Exact, each epoch is an independent steady-state run | Approximate, early buckets average shorter cold runs and later buckets average longer warm runs | N/A, uses Criterion's own t-test and bootstrap |
| Heap allocations | `B/op` and `allocs/op` via `CountingAlloc` | Not recorded | Not recorded |
| Throughput unit | Decimal `MB/s` at `1e6` bytes/s, matching Go `testing.B` | Converts `Throughput::Bytes` to decimal `MB/s` | Decimal `MB/s` or binary `MiB/s` |
| Multi-run comparison | `benchstat -col /dim` across arbitrary files, commits, and Go vs. Rust | `benchstat -col /dim` via `full_id` | Pairwise `base` vs. `new` directory |

## Runner usage

In `Cargo.toml`:

```toml
[dev-dependencies]
benchfmt = { git = "https://github.com/bwplotka/benchfmt-rs" }

[[bench]]
name = "decode"
harness = false
```

In `benches/decode.rs`:

```rust
use std::hint::black_box;
use benchfmt::{CountingAlloc, Suite};

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc::new();

fn main() {
    let data = vec![b'a'; 64 * 1024];
    Suite::from_env("my_crate")
        .with_alloc(&ALLOC)
        .bench("Decode/parser=rust", |b| {
            b.set_bytes(data.len() as u64);
            b.report_allocs();
            while b.keep_running() {
                black_box(data.len());
            }
        })
        .run();
}
```

Run with Go-compatible flags and compare with `benchstat`:

```bash
cargo bench --bench decode -- -bench '^BenchmarkDecode' -benchtime 2s -count 6 -cpu 2 -benchmem | tee rust.txt
benchstat -col /parser -ignore pkg go.txt rust.txt
```

## CLI converter usage

```bash
cargo install --git https://github.com/bwplotka/benchfmt-rs

# Run Criterion benchmarks as usual, then emit 6-epoch benchfmt lines:
cargo bench
benchfmt criterion target/criterion --pkg expfmt --cpu 2 > rust.txt
benchstat old.txt rust.txt
```

## Architecture

[`Cargo.toml`](https://github.com/bwplotka/benchfmt-rs/blob/main/Cargo.toml#L12-L23) declares zero external dependencies. The crate implements CLI flag parsing, duration parsing, benchmark iteration calibration, thread-local allocation tracking, `benchfmt` formatting, and a ~250-line recursive-descent JSON reader in [`src/json.rs`](https://github.com/bwplotka/benchfmt-rs/blob/main/src/json.rs#L59-L253) from scratch on `std` alone. It delegates statistical comparison and `benchfmt` parsing to Go's upstream `benchstat` binary rather than reimplementing them.

### How `Suite` runs a benchmark

1. [`Suite::from_env`](https://github.com/bwplotka/benchfmt-rs/blob/main/src/lib.rs#L501-L507) parses Go flags such as `-bench`, `-benchtime`, `-count`, `-cpu`, and `-benchmem` alongside Cargo's `--bench` and `--list` via [`BenchConfig::from_args`](https://github.com/bwplotka/benchfmt-rs/blob/main/src/lib.rs#L310-L372). [`BenchHeader::detect`](https://github.com/bwplotka/benchfmt-rs/blob/main/src/lib.rs#L143-L163) translates Rust `OS` and `ARCH` constants to Go's `GOOS` and `GOARCH` names, such as `macos` to `darwin` and `x86_64` to `amd64`, and reads `/proc/cpuinfo` on Linux so Go and Rust runs emit compatible metadata headers.
2. [`Suite::bench`](https://github.com/bwplotka/benchfmt-rs/blob/main/src/lib.rs#L541-L545) normalizes names through [`sanitize_bench_name`](https://github.com/bwplotka/benchfmt-rs/blob/main/src/lib.rs#L255-L269), uppercasing the first character and replacing whitespace with `_` while keeping `/key=value` dimensions intact for `benchstat -col`.
3. [`Suite::run_to`](https://github.com/bwplotka/benchfmt-rs/blob/main/src/lib.rs#L556-L589) runs `count` independent epochs per matching benchmark via [`run_single_epoch`](https://github.com/bwplotka/benchfmt-rs/blob/main/src/lib.rs#L592-L669). Fixed `-benchtime=1000x` targets run once at `n = 1000`. Duration targets start at `n = 1` and recalibrate until a single run reaches `-benchtime` or `1_000_000_000` iterations, growing `n` by `1.2 * (target_ns * n / elapsed_ns)` capped at `100x` per step to match Go's `testing.B` ramp-up.
4. Inside each run, [`Bencher::keep_running`](https://github.com/bwplotka/benchfmt-rs/blob/main/src/lib.rs#L458-L477) and [`Bencher::iter`](https://github.com/bwplotka/benchfmt-rs/blob/main/src/lib.rs#L481-L485) mirror Go 1.24's `b.Loop()`. They exclude any setup before the first `keep_running()` call, start the clock and allocation snapshot on the first iteration, and stop both on the final call when `remaining == 0` before outer cleanup runs. Callers can also reset both baselines manually with [`Bencher::reset_timer`](https://github.com/bwplotka/benchfmt-rs/blob/main/src/lib.rs#L445-L453).
5. [`CountingAlloc`](https://github.com/bwplotka/benchfmt-rs/blob/main/src/alloc.rs#L63-L142) in [`src/alloc.rs`](https://github.com/bwplotka/benchfmt-rs/blob/main/src/alloc.rs) is the only module opting into `unsafe_code`. It wraps `std::alloc::GlobalAlloc`, delegating to `std::alloc::System` by default, and bumps a `const`-initialized `thread_local!` `Cell<(u64, u64)>` on `alloc`, `alloc_zeroed`, and `realloc`. Using `const` thread-local storage avoids atomic cache-line bouncing across cores and never triggers recursive heap allocations during initialization.
6. [`BenchRecord::write`](https://github.com/bwplotka/benchfmt-rs/blob/main/src/lib.rs#L217-L249) writes `Benchmark<Name>-<cpu>\t<N>\t<ns/op>` with optional decimal `MB/s`, `B/op`, and `allocs/op`, ending the stream with `PASS`.

### How the converters work

- [`convert_criterion_dir`](https://github.com/bwplotka/benchfmt-rs/blob/main/src/convert.rs#L50-L160) walks `target/criterion/**/<baseline>/`, reads `benchmark.json` and `sample.json` through [`parse_json`](https://github.com/bwplotka/benchfmt-rs/blob/main/src/json.rs#L59-L70), and partitions Criterion's `(iters, times)` pairs into `--epochs` contiguous buckets, `6` by default, emitting `sum_ns / sum_iters` per bucket.
- [`convert_libtest_output`](https://github.com/bwplotka/benchfmt-rs/blob/main/src/convert.rs#L193-L241) parses `cargo +nightly bench` text lines into single-epoch [`BenchRecord`](https://github.com/bwplotka/benchfmt-rs/blob/main/src/lib.rs#L198-L250) entries.

### Scope compared to Go's `testing.B`

- `-bench` strips leading `^`, trailing `$`, and `Benchmark`, then matches names with `str::contains` rather than full regular expressions.
- `-cpu N` sets the `-N` suffix on `BenchmarkFoo-N` lines rather than sweeping a comma-separated thread list or tuning a runtime scheduler.
- `CountingAlloc` tracks allocations on the calling thread only, not across background worker threads.
- Benchmarks register flat names on `Suite::bench` such as `"Decode/case=1kb"` rather than nesting `b.Run` sub-closures, and `Bencher` does not yet expose `StopTimer` / `StartTimer` or custom `ReportMetric` units.

## License

Apache-2.0
