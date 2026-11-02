# benchfmt

> [!INFO]
> I'm coming from the Go ecosystem and I can't live without Go's [`benchstat`-- it's too amazing](https://www.bwplotka.dev/2024/go-microbenchmarks-benchstat/). This tooling brings [`golang.org/x/perf/benchfmt`](https://pkg.go.dev/golang.org/x/perf/benchfmt) experience to Rust!

Zero-dependency Rust benchmark runner, heap allocation counter, and Criterion/`libtest` converter emitting Go's [Benchmark Data Format](https://go.googlesource.com/proposal/+/master/design/14313-benchmark-format.md) for direct analysis with [`benchstat`](https://pkg.go.dev/golang.org/x/perf/cmd/benchstat).

## Why this exists

Comparing Rust changes across commits or benchmarking Rust implementations side by side against Go code hits three practical gaps:

1. `benchstat` expects Go's `benchfmt` line format with `ns/op`, decimal `MB/s`, `B/op`, `allocs/op`, independent `-count` epochs for Mann-Whitney U tests, and `/key=value` sub-benchmark dimensions.
2. Criterion measures wall-clock time and throughput, leaving out per-operation heap bytes and allocation counts.
3. Go's `testing.B` calibrates an iteration count `N` for a fixed `-benchtime` such as `2s` and repeats `-count` independent epochs, whereas Criterion defaults to 100 linear-ramp samples.

`benchfmt` covers both new and existing suites:
- `Suite` and `CountingAlloc` run `harness = false` benchmarks using Go's `testing.B` calibration loop, CLI flags (`-bench`, `-benchtime`, `-count`, `-cpu`, `-benchmem`), and thread-local `B/op` and `allocs/op` counters.
- The `benchfmt` CLI converts existing `target/criterion` directories or `cargo +nightly bench` output into `benchfmt` files without modifying benchmark code.

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

## License

Apache-2.0
