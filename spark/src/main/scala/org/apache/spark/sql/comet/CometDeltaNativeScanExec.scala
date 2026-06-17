/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.spark.sql.comet

import scala.jdk.CollectionConverters._

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.{Attribute, SortOrder}
import org.apache.spark.sql.catalyst.plans.QueryPlan
import org.apache.spark.sql.catalyst.plans.physical.{Partitioning, UnknownPartitioning}
import org.apache.spark.sql.execution.FileSourceScanExec
import org.apache.spark.sql.vectorized.ColumnarBatch

import com.google.common.base.Objects

import org.apache.comet.serde.OperatorOuterClass.Operator

/**
 * Native Delta Lake scan that reads a table through delta-kernel-rs (behind the `delta` cargo
 * feature). The native `DeltaScan` operator is self-contained - it carries the table URI and the
 * required column projection, and the kernel performs log replay, column pruning, deletion-vector
 * and column-mapping application, and partition-value injection - so this leaf needs no
 * per-partition planning data and reads the table as a single partition. Partition pruning and
 * predicate pushdown are follow-ups.
 */
case class CometDeltaNativeScanExec(
    override val nativeOp: Operator,
    override val output: Seq[Attribute],
    tableUri: String,
    @transient originalPlan: FileSourceScanExec,
    override val serializedPlanOpt: SerializedPlan)
    extends CometLeafExec {

  override val supportsColumnar: Boolean = true

  override val nodeName: String = s"CometDeltaNativeScan $tableUri"

  override lazy val outputPartitioning: Partitioning = UnknownPartitioning(1)

  override lazy val outputOrdering: Seq[SortOrder] = Nil

  override def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val nativeMetrics = CometMetricNode.fromCometPlan(this)
    val serializedPlan = CometExec.serializeNativePlan(nativeOp)
    CometExecRDD(
      sparkContext,
      inputRDDs = Seq.empty,
      commonByKey = Map.empty,
      perPartitionByKey = Map.empty,
      serializedPlan = serializedPlan,
      numPartitions = 1,
      numOutputCols = output.length,
      nativeMetrics = nativeMetrics,
      subqueries = Seq.empty)
  }

  override def convertBlock(): CometDeltaNativeScanExec = {
    val newSerializedPlan = if (serializedPlanOpt.isEmpty) {
      SerializedPlan(Some(CometExec.serializeNativePlan(nativeOp)))
    } else {
      serializedPlanOpt
    }
    CometDeltaNativeScanExec(nativeOp, output, tableUri, originalPlan, newSerializedPlan)
  }

  override protected def doCanonicalize(): CometDeltaNativeScanExec = {
    CometDeltaNativeScanExec(
      nativeOp,
      output.map(QueryPlan.normalizeExpressions(_, output)),
      tableUri,
      null,
      SerializedPlan(None))
  }

  override def stringArgs: Iterator[Any] = Iterator(output, tableUri)

  override def equals(obj: Any): Boolean = obj match {
    case other: CometDeltaNativeScanExec =>
      tableUri == other.tableUri && output == other.output &&
      serializedPlanOpt == other.serializedPlanOpt
    case _ => false
  }

  override def hashCode(): Int = Objects.hashCode(tableUri, output.asJava, serializedPlanOpt)
}
