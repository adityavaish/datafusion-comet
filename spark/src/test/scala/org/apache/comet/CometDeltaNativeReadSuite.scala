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

package org.apache.comet

import org.apache.spark.SparkConf
import org.apache.spark.sql.CometTestBase
import org.apache.spark.sql.comet.CometDeltaNativeScanExec
import org.apache.spark.sql.internal.SQLConf

/**
 * End-to-end tests for the native Delta Lake scan (delta-kernel-rs). These require Comet to be
 * built with the `delta` cargo feature; when it is not, the suite self-cancels (the native
 * `isFeatureEnabled("delta")` check returns false), so it is safe to run in CI against a Comet
 * built without the feature.
 */
class CometDeltaNativeReadSuite extends CometTestBase {

  override protected def sparkConf: SparkConf = {
    val conf = super.sparkConf
    conf.set("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    conf.set("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    conf
  }

  private def assumeDeltaFeature(): Unit =
    assume(isFeatureEnabled("delta"), "Comet was not built with the `delta` cargo feature")

  private def nativeScanPartitions(df: org.apache.spark.sql.DataFrame): Int =
    stripAQEPlan(df.queryExecution.executedPlan)
      .collect { case s: CometDeltaNativeScanExec =>
        s.numPartitions
      }
      .headOption
      .getOrElse(0)

  test("native split scan over a multi-file table matches Spark") {
    assumeDeltaFeature()
    withSQLConf(
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "false",
      CometConf.COMET_DELTA_NATIVE_ENABLED.key -> "true") {
      withTempPath { dir =>
        val path = dir.getCanonicalPath
        // Range-partition into 8 files so the native scan splits across multiple partitions.
        spark
          .range(0, 4000)
          .selectExpr("id", "cast(id as double) * 1.5 as score", "cast(id as string) as label")
          .repartitionByRange(8, org.apache.spark.sql.functions.col("id"))
          .write
          .format("delta")
          .save(path)

        val df = spark.read.format("delta").load(path)
        df.collect()
        val names = stripAQEPlan(df.queryExecution.executedPlan).collect { case p =>
          p.getClass.getSimpleName
        }
        assert(
          names.contains("CometDeltaNativeScanExec"),
          s"expected a native Delta scan, got: ${names.mkString(", ")}")
        val parts = nativeScanPartitions(df)
        info(s"native scan partitions: $parts")
        assert(parts > 1, s"expected a split (multi-partition) native scan, got $parts")
        checkSparkAnswer(df)
        // An aggregate over the split scan must also match Spark.
        checkSparkAnswer(df.groupBy("label").count())
      }
    }
  }

  test("native Delta scan falls back for deletion-vector (merge-on-read) tables") {
    assumeDeltaFeature()
    withSQLConf(
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "false",
      CometConf.COMET_DELTA_NATIVE_ENABLED.key -> "true") {
      withTempPath { dir =>
        val path = dir.getCanonicalPath
        spark
          .range(0, 2000)
          .selectExpr("id", "cast(id as double) as score")
          .repartitionByRange(6, org.apache.spark.sql.functions.col("id"))
          .write
          .format("delta")
          .option("delta.enableDeletionVectors", "true")
          .save(path)
        // DELETE creates deletion vectors (merge-on-read): Delta's scan reads hidden
        // __delta_internal_* columns + a Filter, so the native scan must fall back. Results,
        // produced by Spark, must still be correct.
        spark.sql(s"DELETE FROM delta.`$path` WHERE id % 7 = 0")

        val df = spark.read.format("delta").load(path)
        df.collect()
        val names = stripAQEPlan(df.queryExecution.executedPlan).collect { case p =>
          p.getClass.getSimpleName
        }
        assert(
          !names.contains("CometDeltaNativeScanExec"),
          s"native scan must fall back for a merge-on-read DV table, got: ${names.mkString(", ")}")
        checkSparkAnswer(df)
      }
    }
  }

  test("native Delta scan with a data filter matches Spark across multiple files") {
    assumeDeltaFeature()
    withSQLConf(
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "false",
      CometConf.COMET_DELTA_NATIVE_ENABLED.key -> "true") {
      withTempPath { dir =>
        val path = dir.getCanonicalPath
        // Three commits => three files with disjoint id ranges, so a kernel data-skipping bug
        // (skipping a file that holds matching rows) would drop rows and fail checkSparkAnswer.
        spark
          .range(0, 30)
          .selectExpr("id", "cast(id as double) as score")
          .write
          .format("delta")
          .mode("append")
          .save(path)
        spark
          .range(30, 60)
          .selectExpr("id", "cast(id as double) as score")
          .write
          .format("delta")
          .mode("append")
          .save(path)
        spark
          .range(60, 90)
          .selectExpr("id", "cast(id as double) as score")
          .write
          .format("delta")
          .mode("append")
          .save(path)

        val df = spark.read.format("delta").load(path).where("id >= 55 and score < 80.0")
        df.collect()
        val names = stripAQEPlan(df.queryExecution.executedPlan).collect { case p =>
          p.getClass.getSimpleName
        }
        info(s"plan nodes: ${names.mkString(", ")}")
        assert(
          names.contains("CometDeltaNativeScanExec"),
          s"expected a native Delta scan, got: ${names.mkString(", ")}")
        checkSparkAnswer(df)
      }
    }
  }

  test("native Delta scan with projection and a data filter matches Spark") {
    assumeDeltaFeature()
    withSQLConf(
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "false",
      CometConf.COMET_DELTA_NATIVE_ENABLED.key -> "true") {
      withTempPath { dir =>
        val path = dir.getCanonicalPath
        spark
          .range(0, 200)
          .selectExpr("id", "cast(id as double) * 1.5 as score", "cast(id as string) as label")
          .write
          .format("delta")
          .save(path)

        // Filter references a column not in the projection; Spark keeps it in requiredSchema.
        val df =
          spark.read.format("delta").load(path).where("score > 100.0").select("label", "id")
        df.collect()
        val names = stripAQEPlan(df.queryExecution.executedPlan).collect { case p =>
          p.getClass.getSimpleName
        }
        assert(
          names.contains("CometDeltaNativeScanExec"),
          s"expected a native Delta scan, got: ${names.mkString(", ")}")
        checkSparkAnswer(df)
      }
    }
  }

  test("native Delta scan reads a plain non-partitioned table") {
    assumeDeltaFeature()
    withSQLConf(
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "false",
      CometConf.COMET_DELTA_NATIVE_ENABLED.key -> "true") {
      withTempPath { dir =>
        val path = dir.getCanonicalPath
        spark
          .range(0, 50)
          .selectExpr("id", "cast(id as double) * 1.5 as score")
          .write
          .format("delta")
          .save(path)

        val df = spark.read.format("delta").load(path)
        df.collect()
        val names = stripAQEPlan(df.queryExecution.executedPlan).collect { case p =>
          p.getClass.getSimpleName
        }
        info(s"plan nodes: ${names.mkString(", ")}")
        assert(
          names.contains("CometDeltaNativeScanExec"),
          s"expected a native Delta scan, got: ${names.mkString(", ")}")
        checkSparkAnswer(df)
      }
    }
  }

  test("native Delta scan prunes to the projected columns") {
    assumeDeltaFeature()
    withSQLConf(
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "false",
      CometConf.COMET_DELTA_NATIVE_ENABLED.key -> "true") {
      withTempPath { dir =>
        val path = dir.getCanonicalPath
        spark
          .range(0, 50)
          .selectExpr("id", "cast(id as double) * 1.5 as score", "cast(id as string) as label")
          .write
          .format("delta")
          .save(path)

        // Select a subset of columns, reordered relative to the table schema.
        val df = spark.read.format("delta").load(path).select("label", "id")
        df.collect()
        val names = stripAQEPlan(df.queryExecution.executedPlan).collect { case p =>
          p.getClass.getSimpleName
        }
        info(s"plan nodes: ${names.mkString(", ")}")
        assert(
          names.contains("CometDeltaNativeScanExec"),
          s"expected a native Delta scan, got: ${names.mkString(", ")}")
        checkSparkAnswer(df)
      }
    }
  }

  test("native Delta scan reads a partitioned table") {
    assumeDeltaFeature()
    withSQLConf(
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "false",
      CometConf.COMET_DELTA_NATIVE_ENABLED.key -> "true") {
      withTempPath { dir =>
        val path = dir.getCanonicalPath
        spark
          .range(0, 60)
          .selectExpr(
            "id",
            "cast(id as double) * 1.5 as score",
            "cast(id % 3 as int) as part_i",
            "concat('p', cast(id % 2 as string)) as part_s")
          .write
          .format("delta")
          .partitionBy("part_i", "part_s")
          .save(path)

        // Full read of all partitions (no partition filter): partition values are injected by the
        // kernel and must match Spark, including the partition columns.
        val df = spark.read.format("delta").load(path)
        df.collect()
        val names = stripAQEPlan(df.queryExecution.executedPlan).collect { case p =>
          p.getClass.getSimpleName
        }
        info(s"plan nodes: ${names.mkString(", ")}")
        assert(
          names.contains("CometDeltaNativeScanExec"),
          s"expected a native Delta scan, got: ${names.mkString(", ")}")
        checkSparkAnswer(df)
      }
    }
  }

  test("native Delta scan falls back to Spark for a partition filter") {
    assumeDeltaFeature()
    withSQLConf(
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "false",
      CometConf.COMET_DELTA_NATIVE_ENABLED.key -> "true") {
      withTempPath { dir =>
        val path = dir.getCanonicalPath
        spark
          .range(0, 30)
          .selectExpr("id", "cast(id % 3 as int) as part_i")
          .write
          .format("delta")
          .partitionBy("part_i")
          .save(path)

        // A partition predicate is consumed by file pruning with no Filter node above, so the
        // native full-read path must not fire (it would return pruned-away rows).
        val df = spark.read.format("delta").load(path).where("part_i = 1")
        df.collect()
        val names = stripAQEPlan(df.queryExecution.executedPlan).collect { case p =>
          p.getClass.getSimpleName
        }
        assert(
          !names.contains("CometDeltaNativeScanExec"),
          s"native Delta scan must not fire with a partition filter, got: ${names.mkString(", ")}")
        checkSparkAnswer(df)
      }
    }
  }

  test("native Delta scan falls back to Spark when disabled") {
    assumeDeltaFeature()
    withSQLConf(
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "false",
      CometConf.COMET_DELTA_NATIVE_ENABLED.key -> "false") {
      withTempPath { dir =>
        val path = dir.getCanonicalPath
        spark.range(0, 10).toDF("id").write.format("delta").save(path)
        val df = spark.read.format("delta").load(path)
        df.collect()
        val names = stripAQEPlan(df.queryExecution.executedPlan).collect { case p =>
          p.getClass.getSimpleName
        }
        assert(!names.contains("CometDeltaNativeScanExec"))
        checkSparkAnswer(df)
      }
    }
  }
}
