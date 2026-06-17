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
