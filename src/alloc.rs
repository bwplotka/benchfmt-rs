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

//! Global allocator wrapper for reporting `B/op` and `allocs/op` in benchmarks.

#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    // `const` initialization guarantees TLS access never triggers a recursive heap allocation.
    static THREAD_ALLOC_COUNTS: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
}

#[inline]
fn record_thread_alloc(bytes: usize) {
    #[cfg(feature = "pprof")]
    if crate::pprof_support::is_in_heap_sample() {
        return;
    }
    let _ = THREAD_ALLOC_COUNTS.try_with(|cell| {
        let (a, b) = cell.get();
        cell.set((a.wrapping_add(1), b.wrapping_add(bytes as u64)));
    });
    #[cfg(feature = "pprof")]
    crate::pprof_support::maybe_record_heap_alloc(bytes);
}

/// Point-in-time snapshot of cumulative heap allocations on the current thread.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AllocSnapshot {
    /// Total number of heap allocation calls observed so far on the calling thread.
    pub allocs: u64,
    /// Total number of heap bytes requested so far on the calling thread.
    pub bytes: u64,
}

impl AllocSnapshot {
    /// Computes the difference between `after` and `self` using wrapping subtraction.
    #[must_use]
    pub fn delta_to(self, after: Self) -> Self {
        Self {
            allocs: after.allocs.wrapping_sub(self.allocs),
            bytes: after.bytes.wrapping_sub(self.bytes),
        }
    }
}

/// Trait implemented by allocation trackers attached to a [`crate::Suite`].
pub trait AllocTracker: Sync {
    /// Captures the current cumulative allocation counters for the calling thread.
    fn snapshot(&self) -> AllocSnapshot;
}

/// Thin [`GlobalAlloc`] wrapper that counts heap allocations and allocated bytes per thread
/// without atomic cache-line contention or cross-thread noise, matching Go's
/// `testing.B.ReportAllocs` (`B/op` and `allocs/op`) for single-threaded benchmarks.
pub struct CountingAlloc<A = System> {
    inner: A,
}

impl CountingAlloc<System> {
    /// Creates a [`CountingAlloc`] wrapping [`std::alloc::System`].
    #[must_use]
    pub const fn new() -> Self {
        Self::with_allocator(System)
    }
}

impl Default for CountingAlloc<System> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A> CountingAlloc<A> {
    /// Creates a [`CountingAlloc`] wrapping a custom [`GlobalAlloc`] implementation.
    #[must_use]
    pub const fn with_allocator(inner: A) -> Self {
        Self { inner }
    }

    /// Captures the current cumulative allocation counters on the calling thread.
    #[must_use]
    pub fn snapshot(&self) -> AllocSnapshot {
        THREAD_ALLOC_COUNTS
            .try_with(|cell| {
                let (allocs, bytes) = cell.get();
                AllocSnapshot { allocs, bytes }
            })
            .unwrap_or_default()
    }
}

impl<A: Sync> AllocTracker for CountingAlloc<A> {
    fn snapshot(&self) -> AllocSnapshot {
        Self::snapshot(self)
    }
}

// SAFETY: Every allocation call delegates directly to `self.inner` with identical layout and
// pointer arguments, only updating `const`-initialized thread-local counters on success.
unsafe impl<A: GlobalAlloc> GlobalAlloc for CountingAlloc<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: Upheld by the caller of `GlobalAlloc::alloc`.
        let ptr = unsafe { self.inner.alloc(layout) };
        if !ptr.is_null() {
            record_thread_alloc(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: Upheld by the caller of `GlobalAlloc::dealloc`.
        unsafe { self.inner.dealloc(ptr, layout) };
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: Upheld by the caller of `GlobalAlloc::alloc_zeroed`.
        // Overridden explicitly so inner allocators that delegate `alloc_zeroed` to `alloc`
        // do not double-count.
        let ptr = unsafe { self.inner.alloc_zeroed(layout) };
        if !ptr.is_null() {
            record_thread_alloc(layout.size());
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: Upheld by the caller of `GlobalAlloc::realloc`.
        let new_ptr = unsafe { self.inner.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            record_thread_alloc(new_size);
        }
        new_ptr
    }
}
