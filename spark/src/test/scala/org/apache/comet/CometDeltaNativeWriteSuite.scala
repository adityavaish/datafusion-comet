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

import java.io.File

import scala.io.Source

import org.apache.spark.SparkConf
import org.apache.spark.sql.CometTestBase

/**
 * End-to-end tests for the native Delta Lake write (delta-kernel-rs). Require Comet to be built
 * with the `delta` cargo feature; otherwise the suite self-cancels. Each test proves both
 * correctness (Spark's own Delta reader sees exactly the written data) and provenance (the first
 * commit's engineInfo identifies the kernel writer vs. Delta's own write).
 */
class CometDeltaNativeWriteSuite extends CometTestBase {

  override protected def sparkConf: SparkConf = {
    val conf = super.sparkConf
    conf.set("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    conf.set("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    conf
  }

  private def assumeDeltaFeature(): Unit =
    assume(isFeatureEnabled("delta"), "Comet was not built with the `delta` cargo feature")

  /** Concatenate every Delta commit (all `*.json` under `_delta_log`) to inspect provenance. */
  private def allCommits(path: String): String = {
    val logDir = new File(path, "_delta_log")
    val jsons = Option(logDir.listFiles())
      .getOrElse(Array.empty)
      .filter(_.getName.endsWith(".json"))
      .sortBy(_.getName)
    jsons
      .map { f =>
        val src = Source.fromFile(f)
        try src.getLines().mkString("\n")
        finally src.close()
      }
      .mkString("\n")
  }

  private val cometEngine = "datafusion-comet/delta"

  test("native Delta write creates a kernel-written table that Spark reads back") {
    assumeDeltaFeature()
    withSQLConf(
      CometConf.COMET_DELTA_NATIVE_WRITE_ENABLED.key -> "true",
      // Read back with Spark's own Delta reader to independently validate the written table.
      CometConf.COMET_DELTA_NATIVE_ENABLED.key -> "false") {
      withTempPath { dir =>
        val path = new File(dir, "tbl").getCanonicalPath
        val data = spark
          .range(0, 100)
          .selectExpr(
            "id",
            "cast(id * 1.5 as double) as score",
            "cast(id as string) as label",
            "cast(id % 2 = 0 as boolean) as flag")
        data.write.format("delta").save(path)

        assert(
          allCommits(path).contains(cometEngine),
          "expected the native delta-kernel writer to have committed version 0")
        checkAnswer(spark.read.format("delta").load(path), data.collect().toSeq)
      }
    }
  }

  test("native Delta write falls back to Delta for a partitioned write") {
    assumeDeltaFeature()
    withSQLConf(CometConf.COMET_DELTA_NATIVE_WRITE_ENABLED.key -> "true") {
      withTempPath { dir =>
        val path = new File(dir, "tblp").getCanonicalPath
        val data = spark.range(0, 30).selectExpr("id", "cast(id % 3 as int) as p")
        data.write.format("delta").partitionBy("p").save(path)

        assert(
          !allCommits(path).contains(cometEngine),
          "a partitioned write must fall back to Delta's own writer")
        checkAnswer(spark.read.format("delta").load(path), data.collect().toSeq)
      }
    }
  }

  test("native Delta write falls back to Delta when disabled") {
    assumeDeltaFeature()
    withSQLConf(CometConf.COMET_DELTA_NATIVE_WRITE_ENABLED.key -> "false") {
      withTempPath { dir =>
        val path = new File(dir, "tbld").getCanonicalPath
        spark.range(0, 5).toDF("id").write.format("delta").save(path)
        assert(
          !allCommits(path).contains(cometEngine),
          "with the config disabled the native writer must not run")
      }
    }
  }
}
