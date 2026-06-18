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
//! Compiled only when the `delta` cargo feature is enabled. The kernel performs Delta-specific
//! work (log replay, file enumeration, deletion vectors, column mapping, partition values, and
//! transactional commits); Comet only sees plain Apache Arrow `RecordBatch`es on the way in and
//! out. This keeps the integration minimal — there is no reimplementation of the Delta protocol.
//!
//! Implemented here:
//! - [`snapshot_summary`] / [`list_scan_files`] — log replay + file enumeration.
//! - [`scan_to_batches`] — full read (deletion vectors and column mapping applied by the kernel).
//! - [`create_table`] / [`append`] — transactional write (unpartitioned).

use std::collections::HashMap;
use std::sync::Arc;

use delta_kernel::arrow::array::{BooleanArray, RecordBatch};
use delta_kernel::arrow::compute::filter_record_batch;
use delta_kernel::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::engine::arrow_conversion::{TryFromArrow, TryIntoArrow};
use delta_kernel::engine::arrow_data::{ArrowEngineData, EngineDataArrowExt};
use delta_kernel::engine::default::executor::tokio::TokioBackgroundExecutor;
use delta_kernel::engine::default::storage::store_from_url_opts;
use delta_kernel::engine::default::{DefaultEngine, DefaultEngineBuilder};
use delta_kernel::scan::state::{transform_to_logical, ScanFile};
use delta_kernel::schema::{SchemaRef, StructType};
use delta_kernel::transaction::create_table::create_table as kernel_create_table;
use delta_kernel::transaction::{CommitResult, RetryableTransaction};
use delta_kernel::{DeltaResult, Engine, Error, FileMeta, Snapshot, SnapshotRef};
use itertools::Itertools;
use url::Url;

mod predicate;
mod scan_exec;
pub use scan_exec::DeltaScanExec;

/// The kernel default engine specialized for Comet (tokio-backed object-store IO).
type KernelEngine = DefaultEngine<TokioBackgroundExecutor>;

/// A single Delta data file to scan, produced by kernel log replay on the driver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeltaScanFile {
    /// File path relative to the table root.
    pub path: String,
    /// File size in bytes (from the Delta log; lets Comet skip a HEAD/stat call).
    pub size: i64,
    /// Whether the kernel attached a logical-to-physical transform (column mapping / partition
    /// value injection) that must be applied after reading.
    pub has_transform: bool,
}

fn build_engine(table_uri: &str) -> DeltaResult<(Url, KernelEngine)> {
    let url = Url::parse(table_uri)
        .map_err(|e| Error::Generic(format!("invalid table URI {table_uri}: {e}")))?;
    let store = store_from_url_opts(&url, HashMap::<String, String>::new())?;
    let engine = DefaultEngineBuilder::new(store).build();
    Ok((url, engine))
}

fn open_snapshot(table_uri: &str) -> DeltaResult<(SnapshotRef, KernelEngine)> {
    let (url, engine) = build_engine(table_uri)?;
    let snapshot = Snapshot::builder_for(url).build(&engine)?;
    Ok((snapshot, engine))
}

/// Open a snapshot of the Delta table at `table_uri`, pinned to `version` when `Some` (else
/// latest). Used so every split-read partition reads the exact same snapshot.
fn open_snapshot_at(
    table_uri: &str,
    version: Option<u64>,
) -> DeltaResult<(SnapshotRef, KernelEngine)> {
    let (url, engine) = build_engine(table_uri)?;
    let mut builder = Snapshot::builder_for(url);
    if let Some(v) = version {
        builder = builder.at_version(v);
    }
    let snapshot = builder.build(&engine)?;
    Ok((snapshot, engine))
}

/// Return the latest snapshot's version and live data-file count for the table at `table_uri`. The
/// driver uses this to pin the version and choose a partition count for split-parallel reads.
pub fn snapshot_info(table_uri: &str) -> DeltaResult<(u64, u64)> {
    let (snapshot, engine) = open_snapshot(table_uri)?;
    let version = snapshot.version();
    let scan = snapshot.scan_builder().build()?;
    let mut count: u64 = 0;
    for batch in scan.scan_metadata(&engine)? {
        let files = batch?.visit_scan_files(Vec::<()>::new(), |acc, _f| acc.push(()))?;
        count += files.len() as u64;
    }
    Ok((version, count))
}

/// Open the latest snapshot of the Delta table at `table_uri` and return a short human-readable
/// summary (version + logical schema).
pub fn snapshot_summary(table_uri: &str) -> DeltaResult<String> {
    let (snapshot, _engine) = open_snapshot(table_uri)?;
    Ok(format!(
        "version={:?} schema={:?}",
        snapshot.version(),
        snapshot.schema()
    ))
}

/// Enumerate the data files that make up the latest snapshot of the Delta table at `table_uri`.
/// Performs kernel log replay; every live data file is returned (no predicate pushdown yet).
pub fn list_scan_files(table_uri: &str) -> DeltaResult<Vec<DeltaScanFile>> {
    let (snapshot, engine) = open_snapshot(table_uri)?;
    let scan = snapshot.scan_builder().build()?;
    let mut files: Vec<DeltaScanFile> = Vec::new();
    for batch in scan.scan_metadata(&engine)? {
        files = batch?.visit_scan_files(files, visit_scan_file)?;
    }
    Ok(files)
}

fn visit_scan_file(files: &mut Vec<DeltaScanFile>, scan_file: ScanFile) {
    files.push(DeltaScanFile {
        path: scan_file.path.to_string(),
        size: scan_file.size,
        has_transform: scan_file.transform.is_some(),
    });
}

/// Build the projected logical read schema for `columns` (selected in the given order), or `None`
/// to read the table's full schema. Errors if a requested column is not present in the snapshot's
/// logical schema.
fn projected_read_schema(
    snapshot: &Snapshot,
    columns: Option<&[String]>,
) -> DeltaResult<Option<SchemaRef>> {
    let Some(columns) = columns else {
        return Ok(None);
    };
    let full = snapshot.schema();
    let mut fields = Vec::with_capacity(columns.len());
    for name in columns {
        let field = full
            .field(name)
            .ok_or_else(|| Error::generic(format!("column `{name}` not found in Delta schema")))?;
        fields.push(field.clone());
    }
    Ok(Some(Arc::new(StructType::try_new(fields)?)))
}

/// Read the latest snapshot of the Delta table at `table_uri` into Arrow `RecordBatch`es. When
/// `columns` is `Some`, only those columns are read, in the given order (column pruning); `None`
/// reads the full schema. When `predicate` is `Some`, it is pushed into the kernel for best-effort
/// file-level data skipping (a Filter above the scan still enforces exact correctness). The kernel
/// applies deletion vectors and column-mapping transforms and injects partition values, so the
/// returned batches are logically correct, Spark-compatible Arrow data.
pub fn scan_to_batches(
    table_uri: &str,
    columns: Option<&[String]>,
    predicate: Option<delta_kernel::expressions::PredicateRef>,
) -> DeltaResult<Vec<RecordBatch>> {
    let (snapshot, engine) = open_snapshot(table_uri)?;
    let read_schema = projected_read_schema(&snapshot, columns)?;
    let scan = snapshot
        .scan_builder()
        .with_schema_opt(read_schema)
        .with_predicate(predicate)
        .build()?;
    let batches: Vec<RecordBatch> = scan
        .execute(Arc::new(engine))?
        .map(EngineDataArrowExt::try_into_record_batch)
        .try_collect()?;
    Ok(batches)
}

/// Split-parallel read: read only the data files assigned to partition `partition_index` of
/// `num_partitions`, from the snapshot pinned at `version`. Files are enumerated, sorted by path
/// for a deterministic cross-partition assignment, then partitioned round-robin by index. This
/// replicates the kernel's per-file read loop (deletion vectors via a keep-mask filter, plus the
/// logical transform for column mapping / partition values) using only public kernel APIs, so the
/// output is identical to `scan_to_batches` for the assigned subset. Used by `DeltaScanExec` when
/// the JVM schedules more than one partition, so the read parallelizes across Spark cores.
pub fn scan_to_batches_split(
    table_uri: &str,
    columns: Option<&[String]>,
    predicate: Option<delta_kernel::expressions::PredicateRef>,
    version: Option<u64>,
    partition_index: usize,
    num_partitions: usize,
) -> DeltaResult<Vec<RecordBatch>> {
    let num_partitions = num_partitions.max(1);
    let (snapshot, engine) = open_snapshot_at(table_uri, version)?;
    let read_schema = projected_read_schema(&snapshot, columns)?;
    let scan = snapshot
        .scan_builder()
        .with_schema_opt(read_schema)
        .with_predicate(predicate)
        .build()?;

    // Enumerate all live scan files, then sort by path so every partition derives the identical
    // ordering independently (the round-robin assignment below depends on a stable order).
    let mut scan_files: Vec<ScanFile> = Vec::new();
    for batch in scan.scan_metadata(&engine)? {
        scan_files = batch?.visit_scan_files(scan_files, |acc, f| acc.push(f))?;
    }
    scan_files.sort_by(|a, b| a.path.cmp(&b.path));

    let table_root = scan.snapshot().table_root().clone();
    let physical_schema = Arc::clone(scan.physical_schema());
    let logical_schema = Arc::clone(scan.logical_schema());

    let mut batches: Vec<RecordBatch> = Vec::new();
    for scan_file in scan_files
        .into_iter()
        .enumerate()
        .filter(|(i, _)| i % num_partitions == partition_index)
        .map(|(_, f)| f)
    {
        let file_path = table_root.join(&scan_file.path)?;
        // Deletion-vector keep-mask for the whole file (true = keep). None when the file has no DV.
        let mut keep_mask: Option<Vec<bool>> = scan_file
            .dv_info
            .get_selection_vector(&engine, &table_root)?;
        let meta = FileMeta {
            last_modified: 0,
            size: scan_file.size as u64,
            location: file_path,
        };
        let read_iter = engine.parquet_handler().read_parquet_files(
            &[meta],
            Arc::clone(&physical_schema),
            None,
        )?;
        for read_result in read_iter {
            let logical = transform_to_logical(
                &engine,
                read_result?,
                &physical_schema,
                &logical_schema,
                scan_file.transform.clone(),
            )?;
            let batch = logical.try_into_record_batch()?;
            let n = batch.num_rows();
            // Consume the leading `n` entries of the keep mask for this batch; the remainder
            // applies to subsequent batches of the same file. Rows past the mask are kept.
            let masked = match keep_mask.as_mut() {
                Some(mask) => {
                    let rest = if n < mask.len() {
                        Some(mask.split_off(n))
                    } else {
                        None
                    };
                    let mut this = std::mem::take(mask);
                    keep_mask = rest;
                    if this.len() < n {
                        this.resize(n, true);
                    }
                    Some(filter_record_batch(&batch, &BooleanArray::from(this))?)
                }
                None => None,
            };
            batches.push(masked.unwrap_or(batch));
        }
    }
    Ok(batches)
}

/// Read the latest snapshot's logical schema as an Arrow schema, projected to `columns` when
/// `Some`. Used by `DeltaScanExec` to report its output schema without reading any data.
pub fn snapshot_arrow_schema(
    table_uri: &str,
    columns: Option<&[String]>,
) -> DeltaResult<ArrowSchemaRef> {
    let (snapshot, _engine) = open_snapshot(table_uri)?;
    let logical = match projected_read_schema(&snapshot, columns)? {
        Some(schema) => schema,
        None => snapshot.schema(),
    };
    let arrow_schema: delta_kernel::arrow::datatypes::Schema = logical.as_ref().try_into_arrow()?;
    Ok(Arc::new(arrow_schema))
}

/// Create an empty Delta table with the given logical schema at `table_uri`. Errors if the table
/// already exists.
pub fn create_table(table_uri: &str, schema: SchemaRef) -> DeltaResult<()> {
    let (url, engine) = build_engine(table_uri)?;
    let _committed = kernel_create_table(url.as_str(), schema, "datafusion-comet/delta")
        .build(&engine, Box::new(FileSystemCommitter::new()))?
        .commit(&engine)?;
    Ok(())
}

/// Append one Arrow `RecordBatch` to the (existing, unpartitioned) Delta table at `table_uri` as a
/// single transactional commit, returning the committed version. The batch schema must match the
/// table's physical schema.
pub async fn append(table_uri: &str, data: RecordBatch) -> DeltaResult<u64> {
    let (url, engine) = build_engine(table_uri)?;
    let snapshot = Snapshot::builder_for(url).build(&engine)?;
    let mut txn = snapshot
        .transaction(Box::new(FileSystemCommitter::new()), &engine)?
        .with_operation("WRITE".to_string())
        .with_engine_info("datafusion-comet/delta")
        .with_data_change(true);

    let write_context = Arc::new(txn.unpartitioned_write_context()?);
    let engine_data = ArrowEngineData::new(data);
    let file_metadata = engine
        .write_parquet(&engine_data, write_context.as_ref())
        .await?;
    txn.add_files(file_metadata);

    commit_with_retries(txn, &engine)
}

/// Commit a prepared transaction, retrying on retryable conflicts. Returns the committed version.
fn commit_with_retries(
    mut txn: delta_kernel::transaction::Transaction,
    engine: &KernelEngine,
) -> DeltaResult<u64> {
    let mut retries = 0;
    loop {
        if retries > 5 {
            return Err(Error::generic(
                "exceeded maximum retries committing transaction",
            ));
        }
        txn = match txn.commit(engine)? {
            CommitResult::CommittedTransaction(committed) => return Ok(committed.commit_version()),
            CommitResult::ConflictedTransaction(conflicted) => {
                return Err(Error::generic(format!(
                    "transaction conflicted with version {}",
                    conflicted.conflict_version()
                )));
            }
            CommitResult::RetryableTransaction(RetryableTransaction { transaction, .. }) => {
                transaction
            }
        };
        retries += 1;
    }
}

/// Write a sequence of Arrow `RecordBatch`es to the unpartitioned Delta table at `table_uri` in a
/// single transactional commit, creating the table (with `arrow_schema` as its logical schema) if
/// it does not yet exist. Returns the committed version. The caller must restrict this to eligible
/// tables (non-partitioned, kernel-supported types); partition columns are not handled here.
pub async fn write_batches(
    table_uri: &str,
    arrow_schema: ArrowSchemaRef,
    batches: Vec<RecordBatch>,
) -> DeltaResult<u64> {
    let (url, engine) = build_engine(table_uri)?;

    // Create the table on first write (no existing snapshot).
    if Snapshot::builder_for(url.clone()).build(&engine).is_err() {
        let kernel_schema: SchemaRef = Arc::new(
            StructType::try_from_arrow(arrow_schema.as_ref())
                .map_err(|e| Error::generic(format!("arrow->delta schema conversion: {e}")))?,
        );
        let _ = kernel_create_table(url.as_str(), kernel_schema, "datafusion-comet/delta")
            .build(&engine, Box::new(FileSystemCommitter::new()))?
            .commit(&engine)?;
    }

    let snapshot = Snapshot::builder_for(url).build(&engine)?;
    let mut txn = snapshot
        .transaction(Box::new(FileSystemCommitter::new()), &engine)?
        .with_operation("WRITE".to_string())
        .with_engine_info("datafusion-comet/delta")
        .with_data_change(true);

    let write_context = Arc::new(txn.unpartitioned_write_context()?);
    for batch in batches {
        let engine_data = ArrowEngineData::new(batch);
        let file_metadata = engine
            .write_parquet(&engine_data, write_context.as_ref())
            .await?;
        txn.add_files(file_metadata);
    }

    commit_with_retries(txn, &engine)
}

/// Decode an Arrow IPC stream (schema + zero or more batches) from `ipc_bytes` and write it to the
/// Delta table at `table_uri` as a single transactional commit (creating the table if missing).
/// Returns the committed version. Synchronous: drives the kernel's async write on a fresh runtime
/// so it can be called directly from a JNI thread.
pub fn write_arrow_ipc(table_uri: &str, ipc_bytes: &[u8]) -> DeltaResult<u64> {
    let reader = arrow::ipc::reader::StreamReader::try_new(std::io::Cursor::new(ipc_bytes), None)
        .map_err(|e| Error::generic(format!("invalid arrow ipc stream: {e}")))?;
    let arrow_schema = reader.schema();
    let batches: Vec<RecordBatch> = reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| Error::generic(format!("failed to read arrow ipc batches: {e}")))?;
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| Error::generic(format!("failed to start tokio runtime: {e}")))?;
    runtime.block_on(write_batches(table_uri, arrow_schema, batches))
}

#[cfg(test)]
mod tests {
    use super::*;
    use delta_kernel::arrow::array::{Float64Array, Int64Array};
    use delta_kernel::arrow::datatypes::Schema as ArrowSchema;
    use delta_kernel::schema::{DataType, StructField, StructType};
    use std::time::Instant;
    use tempfile::TempDir;

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Runtime::new().unwrap()
    }

    /// (id: long, score: double)
    fn test_schema() -> SchemaRef {
        Arc::new(
            StructType::try_new(vec![
                StructField::nullable("id", DataType::LONG),
                StructField::nullable("score", DataType::DOUBLE),
            ])
            .unwrap(),
        )
    }

    fn make_batch(schema: &SchemaRef, start: i64, n: i64) -> RecordBatch {
        let arrow_schema: ArrowSchema = schema.as_ref().try_into_arrow().unwrap();
        let ids = Int64Array::from((start..start + n).collect::<Vec<_>>());
        let scores = Float64Array::from(
            (start..start + n)
                .map(|i| i as f64 * 1.5)
                .collect::<Vec<_>>(),
        );
        RecordBatch::try_new(
            Arc::new(arrow_schema),
            vec![Arc::new(ids), Arc::new(scores)],
        )
        .unwrap()
    }

    fn table_uri(dir: &TempDir) -> String {
        Url::from_directory_path(dir.path()).unwrap().to_string()
    }

    fn total_rows(batches: &[RecordBatch]) -> usize {
        batches.iter().map(|b| b.num_rows()).sum()
    }

    fn sorted_ids(batches: &[RecordBatch]) -> Vec<i64> {
        let mut ids: Vec<i64> = batches
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        ids.sort_unstable();
        ids
    }

    #[test]
    fn split_read_covers_all_rows_exactly_once() {
        let dir = TempDir::new().unwrap();
        let uri = table_uri(&dir);
        let schema = test_schema();
        create_table(&uri, Arc::clone(&schema)).unwrap();
        // five commits => five data files, so a 3-way split has uneven file counts per partition.
        for i in 0..5 {
            rt().block_on(append(&uri, make_batch(&schema, i * 10, 10)))
                .unwrap();
        }

        let (version, num_files) = snapshot_info(&uri).unwrap();
        assert_eq!(num_files, 5);

        let full = sorted_ids(&scan_to_batches(&uri, None, None).unwrap());
        assert_eq!(full.len(), 50);

        let num_partitions = 3;
        let mut combined: Vec<i64> = Vec::new();
        for p in 0..num_partitions {
            let part =
                scan_to_batches_split(&uri, None, None, Some(version), p, num_partitions).unwrap();
            combined.extend(sorted_ids(&part));
        }
        combined.sort_unstable();
        // Every row is read exactly once across the partitions, matching the full read.
        assert_eq!(combined, full);
    }

    #[test]
    fn split_read_with_projection_matches_full() {
        let dir = TempDir::new().unwrap();
        let uri = table_uri(&dir);
        let schema = test_schema();
        create_table(&uri, Arc::clone(&schema)).unwrap();
        for i in 0..4 {
            rt().block_on(append(&uri, make_batch(&schema, i * 8, 8)))
                .unwrap();
        }
        let (version, _) = snapshot_info(&uri).unwrap();
        let cols = vec!["id".to_string()];

        let mut total = 0usize;
        for p in 0..2 {
            let part = scan_to_batches_split(&uri, Some(&cols), None, Some(version), p, 2).unwrap();
            for b in &part {
                assert_eq!(b.num_columns(), 1);
                assert_eq!(b.schema().field(0).name(), "id");
            }
            total += total_rows(&part);
        }
        assert_eq!(total, 32);
    }

    #[test]
    fn opening_a_missing_table_returns_err() {
        assert!(snapshot_summary("file:///definitely/not/a/delta/table").is_err());
        assert!(list_scan_files("file:///definitely/not/a/delta/table").is_err());
        assert!(scan_to_batches("file:///definitely/not/a/delta/table", None, None).is_err());
    }

    #[test]
    fn create_append_read_round_trip() {
        let dir = TempDir::new().unwrap();
        let uri = table_uri(&dir);
        let schema = test_schema();

        create_table(&uri, Arc::clone(&schema)).unwrap();
        // two separate commits => two versions, two data files
        let v1 = rt()
            .block_on(append(&uri, make_batch(&schema, 0, 5)))
            .unwrap();
        let v2 = rt()
            .block_on(append(&uri, make_batch(&schema, 5, 3)))
            .unwrap();
        assert!(v2 > v1, "each append should advance the table version");

        assert_eq!(
            list_scan_files(&uri).unwrap().len(),
            2,
            "expected two files"
        );

        let batches = scan_to_batches(&uri, None, None).unwrap();
        assert_eq!(total_rows(&batches), 8);

        // verify the actual values round-trip
        let mut ids: Vec<i64> = batches
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn newly_created_table_is_empty_and_keeps_schema() {
        let dir = TempDir::new().unwrap();
        let uri = table_uri(&dir);
        create_table(&uri, test_schema()).unwrap();

        // No data files, no rows, but the schema is readable.
        assert_eq!(list_scan_files(&uri).unwrap().len(), 0);
        assert_eq!(total_rows(&scan_to_batches(&uri, None, None).unwrap()), 0);
        let summary = snapshot_summary(&uri).unwrap();
        assert!(summary.contains("version=0"), "summary was: {summary}");
        assert!(summary.contains("id") && summary.contains("score"));
    }

    #[test]
    fn write_arrow_ipc_creates_table_and_round_trips() {
        let dir = TempDir::new().unwrap();
        let uri = table_uri(&dir);
        let schema = test_schema();
        let arrow_schema: ArrowSchema = schema.as_ref().try_into_arrow().unwrap();
        let arrow_schema_ref = Arc::new(arrow_schema);

        // Two batches => two parquet files written in a single commit; the table is created on the
        // fly from the IPC stream's schema.
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut writer =
                arrow::ipc::writer::StreamWriter::try_new(&mut buf, arrow_schema_ref.as_ref())
                    .unwrap();
            writer.write(&make_batch(&schema, 0, 4)).unwrap();
            writer.write(&make_batch(&schema, 4, 3)).unwrap();
            writer.finish().unwrap();
        }

        let version = write_arrow_ipc(&uri, &buf).unwrap();
        assert_eq!(version, 1, "create (v0) then write (v1)");
        assert_eq!(list_scan_files(&uri).unwrap().len(), 2);

        let batches = scan_to_batches(&uri, None, None).unwrap();
        assert_eq!(total_rows(&batches), 7);
        let mut ids: Vec<i64> = batches
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..7).collect::<Vec<_>>());

        // A second IPC write appends to the now-existing table.
        let mut buf2: Vec<u8> = Vec::new();
        {
            let mut writer =
                arrow::ipc::writer::StreamWriter::try_new(&mut buf2, arrow_schema_ref.as_ref())
                    .unwrap();
            writer.write(&make_batch(&schema, 7, 2)).unwrap();
            writer.finish().unwrap();
        }
        let version2 = write_arrow_ipc(&uri, &buf2).unwrap();
        assert_eq!(version2, 2);
        assert_eq!(total_rows(&scan_to_batches(&uri, None, None).unwrap()), 9);
    }

    #[test]
    fn projection_reads_only_requested_columns_in_order() {
        let dir = TempDir::new().unwrap();
        let uri = table_uri(&dir);
        let schema = test_schema();
        create_table(&uri, Arc::clone(&schema)).unwrap();
        rt().block_on(append(&uri, make_batch(&schema, 0, 4)))
            .unwrap();

        // Project a single column.
        let only_score = vec!["score".to_string()];
        let batches = scan_to_batches(&uri, Some(&only_score), None).unwrap();
        assert_eq!(total_rows(&batches), 4);
        assert_eq!(batches[0].num_columns(), 1);
        assert_eq!(batches[0].schema().field(0).name(), "score");

        // Project both columns in reverse order; the output must follow the requested order.
        let reordered = vec!["score".to_string(), "id".to_string()];
        let batches = scan_to_batches(&uri, Some(&reordered), None).unwrap();
        assert_eq!(batches[0].num_columns(), 2);
        assert_eq!(batches[0].schema().field(0).name(), "score");
        assert_eq!(batches[0].schema().field(1).name(), "id");

        // The projected arrow schema matches the projected read.
        let projected_schema = snapshot_arrow_schema(&uri, Some(&only_score)).unwrap();
        assert_eq!(projected_schema.fields().len(), 1);
        assert_eq!(projected_schema.field(0).name(), "score");

        // An unknown column is an error.
        let missing = vec!["does_not_exist".to_string()];
        assert!(scan_to_batches(&uri, Some(&missing), None).is_err());
    }

    #[test]
    fn perf_scan_throughput() {
        let dir = TempDir::new().unwrap();
        let uri = table_uri(&dir);
        let schema = test_schema();
        let rows: i64 = std::env::var("COMET_DELTA_PERF_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(200_000);

        create_table(&uri, Arc::clone(&schema)).unwrap();
        let write_start = Instant::now();
        rt().block_on(append(&uri, make_batch(&schema, 0, rows)))
            .unwrap();
        let write_ms = write_start.elapsed().as_secs_f64() * 1000.0;

        // warm up, then measure the full read
        let _ = scan_to_batches(&uri, None, None).unwrap();
        let read_start = Instant::now();
        let batches = scan_to_batches(&uri, None, None).unwrap();
        let read_secs = read_start.elapsed().as_secs_f64();

        // measure a projected read of a single column (id) to show column-pruning speedup
        let one_col = vec!["id".to_string()];
        let _ = scan_to_batches(&uri, Some(&one_col), None).unwrap();
        let proj_start = Instant::now();
        let proj_batches = scan_to_batches(&uri, Some(&one_col), None).unwrap();
        let proj_secs = proj_start.elapsed().as_secs_f64();
        assert_eq!(proj_batches[0].num_columns(), 1);

        let n = total_rows(&batches);
        assert_eq!(n as i64, rows);
        // id:long + score:double = 16 logical bytes/row
        let bytes = (n as f64) * 16.0;
        let rows_per_sec = n as f64 / read_secs;
        let mb_per_sec = bytes / read_secs / (1024.0 * 1024.0);
        let proj_rows_per_sec = n as f64 / proj_secs;
        // scalastyle:off
        eprintln!("=== Native Delta (delta-kernel-rs) scan perf ===");
        eprintln!("rows               : {n}");
        eprintln!("write (1 commit)   : {write_ms:.1} ms");
        eprintln!("read all (warm)    : {:.1} ms", read_secs * 1000.0);
        eprintln!("throughput         : {rows_per_sec:.0} rows/s  ({mb_per_sec:.1} MB/s logical)");
        eprintln!("read 1 of 2 cols   : {:.1} ms", proj_secs * 1000.0);
        eprintln!("projected thruput  : {proj_rows_per_sec:.0} rows/s");
        // scalastyle:on
    }
}
