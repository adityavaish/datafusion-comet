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

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.CometTestBase
import org.apache.spark.sql.functions._
import org.apache.spark.sql.internal.SQLConf

/**
 * A JVM-level comparison benchmark for the native Delta Lake read and write paths
 * (delta-kernel-rs) against Spark's own paths. This is NOT a correctness gate, so it self-cancels
 * unless `COMET_DELTA_BENCH=1` is set in the environment and Comet was built with the `delta`
 * cargo feature. The class name intentionally does not end in `Suite` so it is excluded from the
 * CI suite registration check. Run it explicitly, e.g.
 * {{{
 *   COMET_DELTA_BENCH=1 ./mvnw -B test -Dplatform=linux -Darch=amd64 \
 *     -Dsuites='org.apache.comet.CometDeltaBenchmark'
 * }}}
 *
 * Read configurations compared (same physical table for all):
 *   - spark : Comet fully disabled (pure Spark + Delta).
 *   - comet-convert : Comet enabled, native Delta scan off (Spark reads Delta, Comet computes).
 *   - comet-native : Comet enabled, native Delta scan on (delta-kernel-rs reads).
 *
 * Write configurations compared (each to a fresh path):
 *   - delta : Delta's own writer.
 *   - comet-native : native delta-kernel-rs writer (small-write oriented; collects to driver).
 */
class CometDeltaBenchmark extends CometTestBase {

  override protected def sparkConf: SparkConf = {
    val conf = super.sparkConf
    conf.set("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    conf.set("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    conf
  }

  private val benchEnabled: Boolean = sys.env.get("COMET_DELTA_BENCH").contains("1")

  private val readWarmups = sys.env.get("COMET_DELTA_BENCH_WARMUP").map(_.toInt).getOrElse(3)
  private val readIters = sys.env.get("COMET_DELTA_BENCH_ITERS").map(_.toInt).getOrElse(7)
  private val writeIters = sys.env.get("COMET_DELTA_BENCH_WRITE_ITERS").map(_.toInt).getOrElse(3)

  private def assumeBench(): Unit = {
    assume(isFeatureEnabled("delta"), "Comet was not built with the `delta` cargo feature")
    assume(benchEnabled, "set COMET_DELTA_BENCH=1 to run the Delta benchmark")
  }

  private def median(xs: Seq[Double]): Double = {
    val s = xs.sorted
    s(s.length / 2)
  }

  private def timeMs(body: => Unit): Double = {
    val t0 = System.nanoTime()
    body
    (System.nanoTime() - t0) / 1e6
  }

  /**
   * Run `body` warmups times (discarded) then iters times, returning the median wall-clock ms.
   */
  private def measure(warmups: Int, iters: Int)(body: => Unit): Double = {
    (0 until warmups).foreach(_ => body)
    median((0 until iters).map(_ => timeMs(body)))
  }

  /** The three read configurations as (label, confOverrides). */
  private val readConfigs: Seq[(String, Map[String, String])] = Seq(
    ("spark", Map(CometConf.COMET_ENABLED.key -> "false")),
    (
      "comet-convert",
      Map(
        CometConf.COMET_ENABLED.key -> "true",
        CometConf.COMET_CONVERT_FROM_PARQUET_ENABLED.key -> "true",
        CometConf.COMET_DELTA_NATIVE_ENABLED.key -> "false")),
    (
      "comet-native",
      Map(
        CometConf.COMET_ENABLED.key -> "true",
        CometConf.COMET_DELTA_NATIVE_ENABLED.key -> "true")))

  private def withConfs[T](confs: Map[String, String])(f: => T): T = {
    val keys = confs.keys.toSeq
    val saved = keys.map(k =>
      k -> (if (spark.conf.getOption(k).isDefined) Some(spark.conf.get(k))
            else None))
    confs.foreach { case (k, v) => spark.conf.set(k, v) }
    try f
    finally
      saved.foreach {
        case (k, Some(v)) => spark.conf.set(k, v)
        case (k, None) => spark.conf.unset(k)
      }
  }

  private def planContainsNativeDelta(df: DataFrame): Boolean =
    stripAQEPlan(df.queryExecution.executedPlan)
      .collect { case p => p.getClass.getSimpleName }
      .contains("CometDeltaNativeScanExec")

  private def generateTable(path: String, rows: Long): Unit = {
    // Neutral table written by Delta's own writer so every read config reads identical files.
    // Range-partition by id so each file has a disjoint id range (per-file min/max stats are
    // disjoint), which makes the filter query's data skipping meaningful.
    spark
      .range(0, rows)
      .selectExpr(
        "id",
        "cast(id * 1.5 as double) as score",
        "cast(id % 1000 as int) as category",
        "concat('label_', cast(id % 100 as string)) as label")
      .repartitionByRange(8, col("id"))
      .write
      .format("delta")
      .mode("overwrite")
      .save(path)
  }

  // Each read query: (name, builder, forcesNativeEligible). All return a tiny result so the
  // measured time is read/compute bound, not driver-transfer bound.
  private def readQueries(path: String, rows: Long): Seq[(String, () => DataFrame, Boolean)] = {
    val threshold = rows / 20 // ~5% selectivity
    Seq(
      ("count(*)", () => spark.read.format("delta").load(path).agg(count(lit(1))), true),
      (
        "full-agg (all cols)",
        () =>
          spark.read
            .format("delta")
            .load(path)
            .agg(sum("id"), sum("score"), sum("category"), sum(length(col("label")))),
        true),
      (
        "projection sum(score)",
        () => spark.read.format("delta").load(path).agg(sum("score")),
        true),
      (
        "filter id<5% + agg",
        () =>
          spark.read
            .format("delta")
            .load(path)
            .where(col("id") < threshold)
            .agg(sum("score")),
        true))
  }

  test("Delta native vs Spark read/write perf comparison") {
    assumeBench()
    withSQLConf(SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "false") {
      val sb = new StringBuilder
      sb.append("\n================ Comet native Delta benchmark ================\n")
      sb.append(
        s"warmups=$readWarmups readIters=$readIters writeIters=$writeIters " +
          s"cores=${spark.sparkContext.defaultParallelism}\n")

      // ---------------- READ ----------------
      val readSizes = sys.env
        .get("COMET_DELTA_BENCH_READ_SIZES")
        .map(_.split(",").map(_.trim.toLong).toSeq)
        .getOrElse(Seq(200000L, 1000000L, 4000000L))

      withTempPath { dir =>
        readSizes.foreach { rows =>
          val path = new File(dir, s"read_$rows").getCanonicalPath
          generateTable(path, rows)

          sb.append(s"\n--- READ  rows=$rows ---\n")
          sb.append(
            f"${"query"}%-24s ${"spark"}%12s ${"comet-conv"}%12s " +
              f"${"comet-native"}%14s ${"native vs spark"}%16s ${"native vs conv"}%15s\n")

          readQueries(path, rows).foreach { case (qname, build, expectNative) =>
            // Verify the native path actually fires for the native config (else we'd be timing a
            // silent fallback and the comparison would be meaningless).
            val nativeFired = withConfs(readConfigs.last._2)(planContainsNativeDelta(build()))
            val results = readConfigs.map { case (label, confs) =>
              label -> withConfs(confs)(measure(readWarmups, readIters)(build().collect()))
            }.toMap
            val sparkMs = results("spark")
            val convMs = results("comet-convert")
            val natMs = results("comet-native")
            val nativeTag =
              if (expectNative && !nativeFired) " (FELL BACK!)"
              else if (!expectNative && !nativeFired) " (n/a, falls back)"
              else ""
            sb.append(f"$qname%-24s $sparkMs%10.1fms $convMs%10.1fms $natMs%12.1fms " +
              f"${sparkMs / natMs}%14.2fx ${convMs / natMs}%14.2fx$nativeTag\n")
          }
        }
      }

      // ---------------- WRITE ----------------
      val writeSizes = sys.env
        .get("COMET_DELTA_BENCH_WRITE_SIZES")
        .map(_.split(",").map(_.trim.toLong).toSeq)
        .getOrElse(Seq(50000L, 200000L, 1000000L))

      sb.append(s"\n--- WRITE (new non-partitioned table) ---\n")
      sb.append(f"${"rows"}%-12s ${"delta"}%14s ${"comet-native"}%16s ${"speedup"}%12s\n")

      withTempPath { dir =>
        writeSizes.foreach { rows =>
          val data = spark
            .range(0, rows)
            .selectExpr(
              "id",
              "cast(id * 1.5 as double) as score",
              "cast(id % 1000 as int) as category",
              "concat('label_', cast(id % 100 as string)) as label")
            .cache()
          data.count() // materialize cache so write timing excludes upstream compute

          var idx = 0
          def freshPath(tag: String): String = {
            idx += 1
            new File(dir, s"w_${tag}_${rows}_$idx").getCanonicalPath
          }

          val deltaMs =
            withConfs(Map(CometConf.COMET_DELTA_NATIVE_WRITE_ENABLED.key -> "false")) {
              measure(1, writeIters)(data.write.format("delta").save(freshPath("delta")))
            }
          // Confirm the native writer actually committed (engineInfo) for one sample.
          val nativeConfs = Map(
            CometConf.COMET_ENABLED.key -> "true",
            CometConf.COMET_DELTA_NATIVE_WRITE_ENABLED.key -> "true")
          val samplePath = freshPath("native_check")
          withConfs(nativeConfs)(data.write.format("delta").save(samplePath))
          val nativeFired = firstCommitMentions(samplePath, "datafusion-comet/delta")
          val natMs = withConfs(nativeConfs) {
            measure(1, writeIters)(data.write.format("delta").save(freshPath("native")))
          }
          val tag = if (nativeFired) "" else " (FELL BACK!)"
          sb.append(f"$rows%-12d $deltaMs%12.1fms $natMs%14.1fms ${deltaMs / natMs}%10.2fx$tag\n")
          data.unpersist()
        }
      }

      sb.append("==============================================================\n")
      // scalastyle:off
      println(sb.toString)
      // scalastyle:on
      info(sb.toString)
    }
  }

  private def firstCommitMentions(path: String, needle: String): Boolean = {
    val logDir = new File(path, "_delta_log")
    Option(logDir.listFiles())
      .getOrElse(Array.empty)
      .filter(_.getName.endsWith(".json"))
      .exists { f =>
        val src = scala.io.Source.fromFile(f)
        try src.getLines().exists(_.contains(needle))
        finally src.close()
      }
  }
}
