// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Native Delta Lake integration via `delta-kernel-rs`.
//!
//! Compiled only when the `delta` cargo feature is enabled. Kernel runs on the driver to perform
//! log replay and file enumeration; data reads/writes are routed through Comet's existing Parquet
//! machinery. See the branch plan for the full phase breakdown.
//!
//! Phase 1 (this commit) opens a table snapshot through the kernel default engine, proving end to
//! end that kernel log replay links and runs against Comet's arrow-58 dependency tree. Subsequent
//! commits add scan-file enumeration, a protobuf `DeltaScan`, the native scan operator, and the
//! write path.

use std::collections::HashMap;

use delta_kernel::engine::default::storage::store_from_url_opts;
use delta_kernel::engine::default::DefaultEngineBuilder;
use delta_kernel::scan::state::ScanFile;
use delta_kernel::{DeltaResult, Snapshot};
use url::Url;

/// A single Delta data file to scan, produced by kernel log replay on the driver. This is the
/// minimal per-file information Comet needs to later build a `PartitionedFile` for its native
/// Parquet reader. Partition values, deletion vectors, and column-mapping transforms are carried
/// in follow-up phases.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeltaScanFile {
    /// File path relative to the table root.
    pub path: String,
    /// File size in bytes (from the Delta log; lets Comet skip a HEAD/stat call).
    pub size: i64,
    /// Whether the kernel attached a logical-to-physical transform (column mapping / partition
    /// value injection) that must be applied after reading. Handled in a later phase.
    pub has_transform: bool,
}

/// Open the latest snapshot of the Delta table at `table_uri` and return a short human-readable
/// summary (version + logical schema). Phase 1 scaffolding: it exercises kernel log replay via the
/// default (object-store-backed) engine. Cloud credential wiring, file/predicate serialization,
/// and execution through Comet's `ParquetSource` land in later commits.
pub fn snapshot_summary(table_uri: &str) -> DeltaResult<String> {
    let (snapshot, _engine) = open_snapshot(table_uri)?;
    Ok(format!(
        "version={:?} schema={:?}",
        snapshot.version(),
        snapshot.schema()
    ))
}

/// Enumerate the data files that make up the latest snapshot of the Delta table at `table_uri`.
/// This performs kernel log replay (including checkpoint + JSON commit reconciliation) and returns
/// one entry per live data file. Phase 2: no predicate pushdown yet, so every live file is
/// returned.
pub fn list_scan_files(table_uri: &str) -> DeltaResult<Vec<DeltaScanFile>> {
    let (snapshot, engine) = open_snapshot(table_uri)?;
    let scan = snapshot.scan_builder().build()?;
    let mut files: Vec<DeltaScanFile> = Vec::new();
    for batch in scan.scan_metadata(&engine)? {
        files = batch?.visit_scan_files(files, visit_scan_file)?;
    }
    Ok(files)
}

/// Callback invoked by the kernel for each live data file during log replay.
fn visit_scan_file(files: &mut Vec<DeltaScanFile>, scan_file: ScanFile) {
    files.push(DeltaScanFile {
        path: scan_file.path.to_string(),
        size: scan_file.size,
        has_transform: scan_file.transform.is_some(),
    });
}

/// Build a kernel default engine for `table_uri` and load its latest snapshot. The engine is
/// returned alongside the snapshot because scan execution needs it. Credential wiring (S3/Azure/
/// GCS) is intentionally deferred; this uses anonymous/default object-store configuration.
fn open_snapshot(
    table_uri: &str,
) -> DeltaResult<(
    delta_kernel::SnapshotRef,
    delta_kernel::engine::default::DefaultEngine<
        delta_kernel::engine::default::executor::tokio::TokioBackgroundExecutor,
    >,
)> {
    let url = Url::parse(table_uri)
        .map_err(|e| delta_kernel::Error::Generic(format!("invalid table URI {table_uri}: {e}")))?;
    let store = store_from_url_opts(&url, HashMap::<String, String>::new())?;
    let engine = DefaultEngineBuilder::new(store).build();
    let snapshot = Snapshot::builder_for(url).build(&engine)?;
    Ok((snapshot, engine))
}

#[cfg(test)]
mod tests {
    use super::{list_scan_files, snapshot_summary};

    #[test]
    fn opening_a_missing_table_returns_err() {
        // No fixture is created here; the goal is to prove the kernel read path links and is
        // callable. A missing table must surface as an error rather than panic.
        assert!(snapshot_summary("file:///definitely/not/a/delta/table").is_err());
        assert!(list_scan_files("file:///definitely/not/a/delta/table").is_err());
    }

    /// Happy-path check against a real Delta table. Skipped unless `COMET_DELTA_TEST_TABLE` points
    /// at a table URI (e.g. `file:///tmp/my-delta`), so CI stays self-contained. Run locally with:
    /// `COMET_DELTA_TEST_TABLE=file:///path cargo test -p datafusion-comet --features delta`.
    #[test]
    fn lists_files_for_real_table_when_env_set() {
        let Ok(uri) = std::env::var("COMET_DELTA_TEST_TABLE") else {
            eprintln!("skipping: set COMET_DELTA_TEST_TABLE to run");
            return;
        };
        let summary = snapshot_summary(&uri).expect("snapshot_summary");
        let files = list_scan_files(&uri).expect("list_scan_files");
        eprintln!("{summary}");
        eprintln!("scan files ({}): {:#?}", files.len(), files);
        assert!(!files.is_empty(), "expected at least one data file");
    }
}
