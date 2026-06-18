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

package org.apache.comet.rules

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.comet.CometDeltaWriteCommand
import org.apache.spark.sql.execution.datasources.SaveIntoDataSourceCommand

import org.apache.comet.CometConf

/**
 * Post-hoc resolution rule that rewrites an eligible Delta `SaveIntoDataSourceCommand` into a
 * [[CometDeltaWriteCommand]] so the table can be written natively via delta-kernel-rs. Runs
 * during analysis, before Spark eagerly executes the command. The actual eligibility decision
 * (and a transparent fallback to Delta) happens at execution time inside
 * [[CometDeltaWriteCommand]]; this rule only narrows to Delta save commands and is a no-op unless
 * `spark.comet.write.deltaNative.enabled` is set, so its blast radius is minimal.
 */
case class CometDeltaWriteRule(session: SparkSession) extends Rule[LogicalPlan] {

  override def apply(plan: LogicalPlan): LogicalPlan = {
    if (!CometConf.COMET_DELTA_NATIVE_WRITE_ENABLED.get()) {
      return plan
    }
    plan match {
      case s: SaveIntoDataSourceCommand
          if CometDeltaWriteCommand.isDeltaDataSource(s.dataSource) =>
        CometDeltaWriteCommand(s)
      case other => other
    }
  }
}
