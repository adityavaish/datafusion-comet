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

//! DataFusion `ExecutionPlan` that reads a Delta table natively via delta-kernel-rs.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::{DataFusionError, Result as DFResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use datafusion_comet_proto::spark_expression;
use delta_kernel::expressions::PredicateRef;
use futures::TryStreamExt;

use super::predicate::translate_filters;
use super::{scan_to_batches, scan_to_batches_split, snapshot_arrow_schema};

/// Leaf operator that reads a Delta table through the kernel default engine and emits the
/// resulting Arrow batches. The kernel applies deletion vectors, column mapping, and partition
/// values, so the output is Spark-correct. When `projection` is `Some`, only those columns are
/// read, in the given order (column pruning). Translatable data filters are pushed into the kernel
/// for best-effort file skipping. When `num_partitions > 1`, this instance reads only the file
/// subset assigned to `partition_index` (from the pinned `version`), so the read parallelizes
/// across Spark cores; with a single partition it uses the kernel's all-in-one read.
#[derive(Debug)]
pub struct DeltaScanExec {
    table_uri: String,
    projection: Option<Vec<String>>,
    predicate: Option<PredicateRef>,
    version: Option<u64>,
    partition_index: usize,
    num_partitions: usize,
    output_schema: SchemaRef,
    plan_properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl DeltaScanExec {
    /// Build the operator, reading the table's logical schema from the Delta log. When `projection`
    /// is `Some`, the output schema is pruned to those columns, in order. `data_filters` are the
    /// scan's Spark filters; the supported subset is translated into a kernel predicate for data
    /// skipping (the rest are dropped — correctness is enforced by a Filter above the scan).
    /// `version`/`partition_index`/`num_partitions` drive split-parallel reads (one Spark task per
    /// partition, each reading a disjoint file subset of the pinned snapshot).
    pub fn try_new(
        table_uri: String,
        projection: Option<Vec<String>>,
        data_filters: &[spark_expression::Expr],
        version: Option<u64>,
        partition_index: usize,
        num_partitions: usize,
    ) -> DFResult<Self> {
        let output_schema =
            snapshot_arrow_schema(&table_uri, projection.as_deref()).map_err(|e| {
                DataFusionError::Execution(format!("delta: failed to read schema: {e}"))
            })?;
        let predicate = translate_filters(data_filters);
        let plan_properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&output_schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Ok(Self {
            table_uri,
            projection,
            predicate,
            version,
            partition_index,
            num_partitions: num_partitions.max(1),
            output_schema,
            plan_properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }
}

impl DisplayAs for DeltaScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "DeltaScanExec: table_uri={}", self.table_uri)
    }
}

impl ExecutionPlan for DeltaScanExec {
    fn name(&self) -> &str {
        "DeltaScanExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.output_schema)
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.plan_properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let table_uri = self.table_uri.clone();
        let projection = self.projection.clone();
        let predicate = self.predicate.clone();
        let version = self.version;
        let partition_index = self.partition_index;
        let num_partitions = self.num_partitions;
        let schema = Arc::clone(&self.output_schema);

        // The kernel read is synchronous and blocking; run it on a blocking worker so it does not
        // stall the tokio reactor, then stream the resulting batches. With a single partition use
        // the kernel's all-in-one read; otherwise read only this partition's file subset.
        let fut = async move {
            let batches = tokio::task::spawn_blocking(move || {
                if num_partitions > 1 {
                    scan_to_batches_split(
                        &table_uri,
                        projection.as_deref(),
                        predicate,
                        version,
                        partition_index,
                        num_partitions,
                    )
                } else {
                    scan_to_batches(&table_uri, projection.as_deref(), predicate)
                }
            })
            .await
            .map_err(|e| DataFusionError::Execution(format!("delta: scan task failed: {e}")))?
            .map_err(|e| DataFusionError::Execution(format!("delta: scan failed: {e}")))?;
            Ok::<_, DataFusionError>(futures::stream::iter(batches.into_iter().map(Ok)))
        };
        let stream = futures::stream::once(fut).try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::physical_plan::collect;
    use datafusion::prelude::SessionContext;
    use delta_kernel::arrow::array::{Float64Array, Int64Array, RecordBatch};
    use delta_kernel::arrow::datatypes::Schema as ArrowSchema;
    use delta_kernel::engine::arrow_conversion::TryIntoArrow;
    use delta_kernel::schema::{DataType, SchemaRef as KernelSchemaRef, StructField, StructType};
    use tempfile::TempDir;

    #[test]
    fn delta_scan_exec_reads_table_through_datafusion() {
        let dir = TempDir::new().unwrap();
        let uri = url::Url::from_directory_path(dir.path())
            .unwrap()
            .to_string();

        let kschema: KernelSchemaRef = Arc::new(
            StructType::try_new(vec![
                StructField::nullable("id", DataType::LONG),
                StructField::nullable("score", DataType::DOUBLE),
            ])
            .unwrap(),
        );
        super::super::create_table(&uri, Arc::clone(&kschema)).unwrap();

        let arrow_schema: ArrowSchema = kschema.as_ref().try_into_arrow().unwrap();
        let batch = RecordBatch::try_new(
            Arc::new(arrow_schema),
            vec![
                Arc::new(Int64Array::from(vec![0i64, 1, 2, 3])),
                Arc::new(Float64Array::from(vec![0.0f64, 1.5, 3.0, 4.5])),
            ],
        )
        .unwrap();

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(super::super::append(&uri, batch)).unwrap();

        let exec: Arc<dyn ExecutionPlan> =
            Arc::new(DeltaScanExec::try_new(uri, None, &[], None, 0, 1).unwrap());
        assert_eq!(exec.schema().fields().len(), 2);

        let ctx = SessionContext::new();
        let batches = rt.block_on(collect(exec, ctx.task_ctx())).unwrap();
        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(rows, 4);
    }

    #[test]
    fn delta_scan_exec_projects_columns() {
        let dir = TempDir::new().unwrap();
        let uri = url::Url::from_directory_path(dir.path())
            .unwrap()
            .to_string();

        let kschema: KernelSchemaRef = Arc::new(
            StructType::try_new(vec![
                StructField::nullable("id", DataType::LONG),
                StructField::nullable("score", DataType::DOUBLE),
            ])
            .unwrap(),
        );
        super::super::create_table(&uri, Arc::clone(&kschema)).unwrap();

        let arrow_schema: ArrowSchema = kschema.as_ref().try_into_arrow().unwrap();
        let batch = RecordBatch::try_new(
            Arc::new(arrow_schema),
            vec![
                Arc::new(Int64Array::from(vec![0i64, 1, 2, 3])),
                Arc::new(Float64Array::from(vec![0.0f64, 1.5, 3.0, 4.5])),
            ],
        )
        .unwrap();

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(super::super::append(&uri, batch)).unwrap();

        let exec: Arc<dyn ExecutionPlan> = Arc::new(
            DeltaScanExec::try_new(uri, Some(vec!["score".to_string()]), &[], None, 0, 1).unwrap(),
        );
        assert_eq!(exec.schema().fields().len(), 1);
        assert_eq!(exec.schema().field(0).name(), "score");

        let ctx = SessionContext::new();
        let batches = rt.block_on(collect(exec, ctx.task_ctx())).unwrap();
        assert_eq!(batches[0].num_columns(), 1);
        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(rows, 4);
    }
}
