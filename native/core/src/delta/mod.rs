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
//! Phase 0 (this commit) is a dependency spike: it verifies that `delta_kernel` compiles and links
//! alongside Comet's arrow-58 / DataFusion-53 dependency tree. Real scan planning lands in Phase 1.

use delta_kernel::{DeltaResult, Error};

/// Phase 0 link check: reference top-level kernel types so the crate is compiled and linked.
#[allow(dead_code)]
pub(crate) fn kernel_link_check(err: Error) -> String {
    fn _accepts_result<T>(_r: DeltaResult<T>) {}
    err.to_string()
}
