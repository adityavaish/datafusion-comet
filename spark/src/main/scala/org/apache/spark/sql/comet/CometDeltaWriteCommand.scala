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

import scala.util.control.NonFatal

import org.apache.hadoop.fs.Path
import org.apache.spark.internal.Logging
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.catalyst.plans.QueryPlan
import org.apache.spark.sql.comet.execution.arrow.CometDeltaWrite
import org.apache.spark.sql.execution.command.LeafRunnableCommand
import org.apache.spark.sql.execution.datasources.SaveIntoDataSourceCommand
import org.apache.spark.sql.types._

import org.apache.comet.{Native, NativeBase}

/**
 * A drop-in replacement for a Delta [[SaveIntoDataSourceCommand]] that writes the table natively
 * via delta-kernel-rs when the write is eligible, and otherwise (or on any native error)
 * transparently delegates to the original Delta command so a correct table is always produced.
 *
 * Eligibility is intentionally narrow and safe: only the creation of a brand-new, non-partitioned
 * table with kernel-supported primitive column types is handled natively. The query result is
 * collected to the driver and written as a single transactional commit, so this is meant for
 * small writes. Gated by `spark.comet.write.deltaNative.enabled` (default off) via
 * [[org.apache.comet.rules.CometDeltaWriteRule]].
 */
case class CometDeltaWriteCommand(original: SaveIntoDataSourceCommand)
    extends LeafRunnableCommand
    with Logging {

  override def innerChildren: Seq[QueryPlan[_]] = original.innerChildren

  override def run(session: SparkSession): Seq[Row] = {
    val handledNatively =
      try {
        CometDeltaWriteCommand.tryNativeWrite(session, original)
      } catch {
        case NonFatal(e) =>
          logWarning("Native Delta write failed; falling back to Delta's own write", e)
          false
      }
    if (handledNatively) Seq.empty else original.run(session)
  }
}

object CometDeltaWriteCommand extends Logging {

  /** Class name of Delta's V1 data source, matched reflectively to avoid a compile dependency. */
  val DELTA_DATA_SOURCE = "org.apache.spark.sql.delta.sources.DeltaDataSource"

  /** Key under which the DataFrameWriter encodes partitionBy columns into the save options. */
  private val PARTITION_COLUMNS_KEY = "__partition_columns"

  private val supportedTypes: Set[DataType] = Set(
    BooleanType,
    ByteType,
    ShortType,
    IntegerType,
    LongType,
    FloatType,
    DoubleType,
    StringType,
    DateType)

  def isDeltaDataSource(dataSource: Any): Boolean =
    dataSource.getClass.getName == DELTA_DATA_SOURCE

  /**
   * Attempt the native write. Returns true only if the table was written natively; returns false
   * (without side effects) when the write is ineligible, so the caller can fall back to Delta.
   */
  private def tryNativeWrite(session: SparkSession, cmd: SaveIntoDataSourceCommand): Boolean = {
    if (!NativeBase.isFeatureEnabled("delta")) {
      return false
    }
    val pathStr = cmd.options.get("path").orElse(cmd.options.get("PATH")).getOrElse {
      return false
    }
    if (isPartitioned(cmd.options)) {
      return false
    }
    val schema = cmd.query.schema
    if (schema.isEmpty || !schema.fields.forall(f => supportedTypes.contains(f.dataType))) {
      return false
    }

    val hadoopConf = session.sessionState.newHadoopConf()
    val rawPath = new Path(pathStr)
    val fs = rawPath.getFileSystem(hadoopConf)
    val qualified = fs.makeQualified(rawPath)
    // Only brand-new tables: appending to or overwriting an existing table (which may use Delta
    // features the kernel write does not model) falls back to Delta.
    if (fs.exists(new Path(qualified, "_delta_log"))) {
      return false
    }
    // The kernel writes into an existing directory; create the (new) table root first.
    fs.mkdirs(qualified)

    val tableUri = qualified.toUri.toString
    val rows = session.sessionState.executePlan(cmd.query).executedPlan.executeCollect()
    val timeZoneId = session.sessionState.conf.sessionLocalTimeZone
    val ipc = CometDeltaWrite.rowsToArrowIpc(rows, schema, timeZoneId)
    val version = new Native().writeDeltaTable(tableUri, ipc)
    logInfo(
      s"Comet native Delta write committed version $version to $tableUri (${rows.length} rows)")
    true
  }

  private def isPartitioned(options: Map[String, String]): Boolean =
    options.get(PARTITION_COLUMNS_KEY).exists(v => v.nonEmpty && v != "[]")
}
