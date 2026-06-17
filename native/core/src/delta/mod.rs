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
use delta_kernel::{DeltaResult, Snapshot};
use url::Url;

/// Open the latest snapshot of the Delta table at `table_uri` and return a short human-readable
/// summary (version + logical schema). Phase 1 scaffolding: it exercises kernel log replay via the
/// default (object-store-backed) engine. Cloud credential wiring, file/predicate serialization,
/// and execution through Comet's `ParquetSource` land in later commits.
pub fn snapshot_summary(table_uri: &str) -> DeltaResult<String> {
    let url = Url::parse(table_uri)
        .map_err(|e| delta_kernel::Error::Generic(format!("invalid table URI {table_uri}: {e}")))?;
    let store = store_from_url_opts(&url, HashMap::<String, String>::new())?;
    let engine = DefaultEngineBuilder::new(store).build();
    let snapshot = Snapshot::builder_for(url).build(&engine)?;
    Ok(format!(
        "version={:?} schema={:?}",
        snapshot.version(),
        snapshot.schema()
    ))
}

#[cfg(test)]
mod tests {
    use super::snapshot_summary;

    #[test]
    fn opening_a_missing_table_returns_err() {
        // No fixture is created here; the goal is to prove the kernel read path links and is
        // callable. A missing table must surface as an error rather than panic.
        let result = snapshot_summary("file:///definitely/not/a/delta/table");
        assert!(result.is_err(), "expected an error for a missing table");
    }
}
