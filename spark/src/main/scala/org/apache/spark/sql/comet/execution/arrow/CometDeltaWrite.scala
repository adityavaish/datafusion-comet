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

package org.apache.spark.sql.comet.execution.arrow

import java.io.ByteArrayOutputStream
import java.nio.channels.Channels

import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.VectorSchemaRoot
import org.apache.arrow.vector.ipc.ArrowStreamWriter
import org.apache.arrow.vector.types.pojo.Schema
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.comet.util.Utils
import org.apache.spark.sql.types.StructType

/**
 * Helpers for the native Delta write path. Lives in the `arrow` package so it can use the
 * package-private [[ArrowWriter]].
 */
object CometDeltaWrite {

  /**
   * Serialize `rows` into a single-batch Arrow IPC stream (schema message followed by one record
   * batch) that the native side reads with `arrow::ipc::reader::StreamReader`. Intended for
   * driver-side writes of modest size.
   */
  def rowsToArrowIpc(
      rows: Array[InternalRow],
      schema: StructType,
      timeZoneId: String): Array[Byte] = {
    val arrowSchema: Schema = Utils.toArrowSchema(schema, timeZoneId)
    val allocator = new RootAllocator(Long.MaxValue)
    val root = VectorSchemaRoot.create(arrowSchema, allocator)
    val out = new ByteArrayOutputStream()
    val arrowWriter = ArrowWriter.create(root)
    val streamWriter = new ArrowStreamWriter(root, null, Channels.newChannel(out))
    try {
      streamWriter.start()
      var i = 0
      while (i < rows.length) {
        arrowWriter.write(rows(i))
        i += 1
      }
      arrowWriter.finish()
      streamWriter.writeBatch()
      streamWriter.end()
      out.toByteArray
    } finally {
      streamWriter.close()
      root.close()
      allocator.close()
    }
  }
}
