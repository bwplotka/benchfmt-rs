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

//! Minimal `perftools.profiles.Profile` (`profile.proto`) wire encoder and `/proc/self/maps`
//! loader so pprof profiles can be emitted without `prost` or `protoc` build scripts.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::time::{Duration, SystemTime};

use flate2::Compression;
use flate2::write::GzEncoder;

/// Resolved symbol line within a single instruction-pointer location.
#[derive(Clone, Debug)]
pub(crate) struct ResolvedSymbol {
    pub(crate) name: String,
    pub(crate) system_name: String,
    pub(crate) filename: String,
    pub(crate) lineno: u32,
}

#[derive(Clone, Debug)]
struct MappingEntry {
    id: u64,
    memory_start: u64,
    memory_limit: u64,
    file_offset: u64,
    filename: i64,
    has_functions: bool,
    has_filenames: bool,
    has_line_numbers: bool,
    has_inline_frames: bool,
}

#[derive(Clone, Debug)]
struct FunctionEntry {
    id: u64,
    name: i64,
    system_name: i64,
    filename: i64,
}

#[derive(Clone, Debug)]
struct LocationEntry {
    id: u64,
    mapping_id: u64,
    address: u64,
    lines: Vec<(u64, i64)>,
}

#[derive(Clone, Debug)]
struct SampleEntry {
    location_ids: Vec<u64>,
    values: Vec<i64>,
    labels: Vec<(i64, i64)>,
}

/// Builder for a single `perftools.profiles.Profile` message.
pub(crate) struct ProfileBuilder {
    strings: Vec<String>,
    string_idx: HashMap<String, i64>,
    sample_types: Vec<(i64, i64)>,
    period_type: (i64, i64),
    period: i64,
    default_sample_type: i64,
    time_nanos: i64,
    duration_nanos: i64,
    mappings: Vec<MappingEntry>,
    functions: Vec<FunctionEntry>,
    function_idx: HashMap<(i64, i64, i64), u64>,
    locations: Vec<LocationEntry>,
    location_idx: HashMap<(u64, Vec<(u64, i64)>), u64>,
    samples: Vec<SampleEntry>,
}

impl ProfileBuilder {
    pub(crate) fn new(
        sample_types: &[(&str, &str)],
        period_type: (&str, &str),
        period: i64,
        default_sample_type: &str,
        started_at: SystemTime,
        duration: Duration,
    ) -> Self {
        let mut builder = Self {
            // `profile.proto` requires `string_table[0]` to be `""`.
            strings: vec![String::new()],
            string_idx: HashMap::from([(String::new(), 0)]),
            sample_types: Vec::with_capacity(sample_types.len()),
            period_type: (0, 0),
            period,
            default_sample_type: 0,
            time_nanos: started_at
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as i64,
            duration_nanos: duration.as_nanos() as i64,
            mappings: Vec::new(),
            functions: Vec::new(),
            function_idx: HashMap::new(),
            locations: Vec::new(),
            location_idx: HashMap::new(),
            samples: Vec::new(),
        };
        for &(ty, unit) in sample_types {
            let ty_id = builder.intern(ty);
            let unit_id = builder.intern(unit);
            builder.sample_types.push((ty_id, unit_id));
        }
        builder.period_type = (builder.intern(period_type.0), builder.intern(period_type.1));
        builder.default_sample_type = builder.intern(default_sample_type);
        builder.load_proc_maps();
        builder
    }

    pub(crate) fn intern(&mut self, s: &str) -> i64 {
        if let Some(&idx) = self.string_idx.get(s) {
            return idx;
        }
        let idx = self.strings.len() as i64;
        let owned = s.to_owned();
        self.strings.push(owned.clone());
        self.string_idx.insert(owned, idx);
        idx
    }

    pub(crate) fn intern_location(&mut self, address: u64, symbols: &[ResolvedSymbol]) -> u64 {
        let mut lines = Vec::with_capacity(symbols.len());
        let mut saw_fn = false;
        let mut saw_file = false;
        let mut saw_line = false;

        for sym in symbols {
            let name_id = self.intern(&sym.name);
            let sys_id = self.intern(&sym.system_name);
            let file_id = self.intern(&sym.filename);
            saw_fn |= name_id != 0;
            saw_file |= file_id != 0;
            saw_line |= sym.lineno > 0;
            let fn_id = *self
                .function_idx
                .entry((name_id, sys_id, file_id))
                .or_insert_with(|| {
                    let id = (self.functions.len() + 1) as u64;
                    self.functions.push(FunctionEntry {
                        id,
                        name: name_id,
                        system_name: sys_id,
                        filename: file_id,
                    });
                    id
                });
            lines.push((fn_id, i64::from(sym.lineno)));
        }

        let key = (address, lines);
        if let Some(&id) = self.location_idx.get(&key) {
            return id;
        }

        let mapping_id =
            self.find_mapping_and_record_caps(address, saw_fn, saw_file, saw_line, key.1.len() > 1);
        let id = (self.locations.len() + 1) as u64;
        self.locations.push(LocationEntry {
            id,
            mapping_id,
            address,
            lines: key.1.clone(),
        });
        self.location_idx.insert(key, id);
        id
    }

    pub(crate) fn add_sample(
        &mut self,
        location_ids: Vec<u64>,
        values: Vec<i64>,
        str_labels: &[(&str, &str)],
    ) {
        if location_ids.is_empty() {
            return;
        }
        let mut labels = Vec::with_capacity(str_labels.len());
        for &(k, v) in str_labels {
            if !v.is_empty() {
                labels.push((self.intern(k), self.intern(v)));
            }
        }
        self.samples.push(SampleEntry {
            location_ids,
            values,
            labels,
        });
    }

    pub(crate) fn write_gzipped(&self, path: &Path) -> io::Result<()> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        let file = File::create(path)?;
        let mut gz = GzEncoder::new(BufWriter::new(file), Compression::default());
        let encoded = self.encode_proto();
        gz.write_all(&encoded)?;
        let mut writer = gz.finish()?;
        writer.flush()
    }

    fn load_proc_maps(&mut self) {
        let Ok(maps) = fs::read_to_string("/proc/self/maps") else {
            return;
        };
        let exe_path = std::env::current_exe()
            .ok()
            .and_then(|p| fs::canonicalize(&p).ok().or(Some(p)))
            .map(|p| p.to_string_lossy().into_owned());

        let mut parsed = Vec::new();
        for line in maps.lines() {
            let mut parts = line.split_whitespace();
            let Some(range) = parts.next() else { continue };
            let Some(perms) = parts.next() else { continue };
            if !perms.contains('x') {
                continue;
            }
            let Some(offset_hex) = parts.next() else {
                continue;
            };
            let _dev = parts.next();
            let _inode = parts.next();
            let pathname = parts.collect::<Vec<_>>().join(" ");
            if !pathname.starts_with('/') {
                continue;
            }
            let Some((start_hex, end_hex)) = range.split_once('-') else {
                continue;
            };
            let (Ok(start), Ok(limit), Ok(offset)) = (
                u64::from_str_radix(start_hex, 16),
                u64::from_str_radix(end_hex, 16),
                u64::from_str_radix(offset_hex, 16),
            ) else {
                continue;
            };
            let is_main = exe_path.as_deref() == Some(pathname.as_str());
            parsed.push((is_main, start, limit, offset, pathname));
        }

        // `profile.proto` places the main executable at `mapping[0]`.
        parsed.sort_by_key(|(is_main, start, ..)| (!*is_main, *start));
        for (idx, (_, start, limit, offset, pathname)) in parsed.into_iter().enumerate() {
            let filename = self.intern(&pathname);
            self.mappings.push(MappingEntry {
                id: (idx + 1) as u64,
                memory_start: start,
                memory_limit: limit,
                file_offset: offset,
                filename,
                has_functions: false,
                has_filenames: false,
                has_line_numbers: false,
                has_inline_frames: false,
            });
        }
    }

    fn find_mapping_and_record_caps(
        &mut self,
        address: u64,
        has_fn: bool,
        has_file: bool,
        has_line: bool,
        has_inline: bool,
    ) -> u64 {
        if address == 0 {
            return 0;
        }
        for m in &mut self.mappings {
            if address >= m.memory_start && address < m.memory_limit {
                m.has_functions |= has_fn;
                m.has_filenames |= has_file;
                m.has_line_numbers |= has_line;
                m.has_inline_frames |= has_inline;
                return m.id;
            }
        }
        0
    }

    fn encode_proto(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let mut scratch = Vec::new();

        for &(ty, unit) in &self.sample_types {
            scratch.clear();
            encode_int64_field(&mut scratch, 1, ty);
            encode_int64_field(&mut scratch, 2, unit);
            encode_bytes_field(&mut out, 1, &scratch);
        }

        for sample in &self.samples {
            scratch.clear();
            encode_packed_u64_field(&mut scratch, 1, &sample.location_ids);
            encode_packed_i64_field(&mut scratch, 2, &sample.values);
            let mut label_buf = Vec::new();
            for &(k, s) in &sample.labels {
                label_buf.clear();
                encode_int64_field(&mut label_buf, 1, k);
                encode_int64_field(&mut label_buf, 2, s);
                encode_bytes_field(&mut scratch, 3, &label_buf);
            }
            encode_bytes_field(&mut out, 2, &scratch);
        }

        for m in &self.mappings {
            scratch.clear();
            encode_uint64_field(&mut scratch, 1, m.id);
            encode_uint64_field(&mut scratch, 2, m.memory_start);
            encode_uint64_field(&mut scratch, 3, m.memory_limit);
            encode_uint64_field(&mut scratch, 4, m.file_offset);
            encode_int64_field(&mut scratch, 5, m.filename);
            encode_bool_field(&mut scratch, 7, m.has_functions);
            encode_bool_field(&mut scratch, 8, m.has_filenames);
            encode_bool_field(&mut scratch, 9, m.has_line_numbers);
            encode_bool_field(&mut scratch, 10, m.has_inline_frames);
            encode_bytes_field(&mut out, 3, &scratch);
        }

        for loc in &self.locations {
            scratch.clear();
            encode_uint64_field(&mut scratch, 1, loc.id);
            encode_uint64_field(&mut scratch, 2, loc.mapping_id);
            encode_uint64_field(&mut scratch, 3, loc.address);
            let mut line_buf = Vec::new();
            for &(fn_id, lineno) in &loc.lines {
                line_buf.clear();
                encode_uint64_field(&mut line_buf, 1, fn_id);
                encode_int64_field(&mut line_buf, 2, lineno);
                encode_bytes_field(&mut scratch, 4, &line_buf);
            }
            encode_bytes_field(&mut out, 4, &scratch);
        }

        for func in &self.functions {
            scratch.clear();
            encode_uint64_field(&mut scratch, 1, func.id);
            encode_int64_field(&mut scratch, 2, func.name);
            encode_int64_field(&mut scratch, 3, func.system_name);
            encode_int64_field(&mut scratch, 4, func.filename);
            encode_bytes_field(&mut out, 5, &scratch);
        }

        for s in &self.strings {
            // `string_table` is a `repeated string`, so element 0 (`""`) must still be emitted.
            encode_bytes_field(&mut out, 6, s.as_bytes());
        }

        encode_int64_field(&mut out, 9, self.time_nanos);
        encode_int64_field(&mut out, 10, self.duration_nanos);

        scratch.clear();
        encode_int64_field(&mut scratch, 1, self.period_type.0);
        encode_int64_field(&mut scratch, 2, self.period_type.1);
        encode_bytes_field(&mut out, 11, &scratch);

        encode_int64_field(&mut out, 12, self.period);
        encode_int64_field(&mut out, 14, self.default_sample_type);
        out
    }
}

#[inline]
fn encode_varint(out: &mut Vec<u8>, mut val: u64) {
    while val >= 0x80 {
        out.push((val as u8) | 0x80);
        val >>= 7;
    }
    out.push(val as u8);
}

#[inline]
fn encode_uint64_field(out: &mut Vec<u8>, field: u32, val: u64) {
    if val != 0 {
        encode_varint(out, u64::from(field << 3));
        encode_varint(out, val);
    }
}

#[inline]
fn encode_int64_field(out: &mut Vec<u8>, field: u32, val: i64) {
    if val != 0 {
        encode_varint(out, u64::from(field << 3));
        encode_varint(out, val as u64);
    }
}

#[inline]
fn encode_bool_field(out: &mut Vec<u8>, field: u32, val: bool) {
    if val {
        encode_varint(out, u64::from(field << 3));
        out.push(1);
    }
}

#[inline]
fn encode_bytes_field(out: &mut Vec<u8>, field: u32, bytes: &[u8]) {
    encode_varint(out, u64::from((field << 3) | 2));
    encode_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

fn encode_packed_u64_field(out: &mut Vec<u8>, field: u32, vals: &[u64]) {
    if vals.is_empty() {
        return;
    }
    let mut payload = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        encode_varint(&mut payload, v);
    }
    encode_bytes_field(out, field, &payload);
}

fn encode_packed_i64_field(out: &mut Vec<u8>, field: u32, vals: &[i64]) {
    if vals.is_empty() {
        return;
    }
    let mut payload = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        encode_varint(&mut payload, v as u64);
    }
    encode_bytes_field(out, field, &payload);
}
