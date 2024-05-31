/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/experimental/shlo/ops/reverse.h"

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

absl::Status CheckParameters(const Tensor& operand,
                             absl::Span<const DimensionSize> dimensions,
                             Tensor& output) {
  if (operand.element_type() != output.element_type()) {
    return absl::FailedPreconditionError(
        "stablehlo.reverse: The element type of operand must be same as the "
        "element type of output.");
  }

  std::unordered_set<DimensionSize> seen_dimensions;
  for (DimensionSize dimension : dimensions) {
    if (dimension >= output.Rank() || dimension < 0) {
      return absl::FailedPreconditionError(
          "stablehlo.reverse: The dimensions must be in the range of the rank "
          "of output.");
    }
    if (!seen_dimensions.insert(dimension).second) {
      return absl::FailedPreconditionError(
          "stablehlo.reverse: Every dimension to be reversed must be unique.");
    }
  }

  return absl::OkStatus();
}

template <DataType storage_type>
absl::Status EvaluateImpl(ReverseOp& op, const Tensor& operand,
                          Tensor& output) {
  using StorageT = StorageType<storage_type>;

  const StorageT* operand_buffer = operand.GetDataAs<storage_type>();
  const DimensionSize operand_size = operand.NumElements();
  const DimensionSize output_size = output.NumElements();
  const size_t operand_rank = operand.Rank();
  const size_t output_rank = output.Rank();

  absl::InlinedVector<DimensionSize, kMaxNumDimensions> operand_index(
      operand_rank);
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> output_index(
      output_rank);
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> output_shape(
      output.Rank());

  absl::c_copy(output.shape(), output_shape.data());

  // Will be updated with optimized implementation
  for (size_t i = 0; i < operand_size; ++i) {
    operand.GetNdIndex(i, operand_index);
    for (size_t j = 0; j < output_rank; ++j) {
      if (absl::c_find(op.attributes.dimensions, j) !=
          op.attributes.dimensions.end()) {
        output_index[j] = output_shape[j] - operand_index[j] - 1;
      } else {
        output_index[j] = operand_index[j];
      }
    }
    output.Set<storage_type>(output_index, operand_buffer[i]);
  }

  return absl::OkStatus();
}

ReverseOp Create(ReverseOp::Attributes attributes) {
  return {.attributes = attributes};
}

absl::Status Prepare(ReverseOp& op, const Tensor& operand, Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(
      CheckParameters(operand, op.attributes.dimensions, output));

  return absl::OkStatus();
}

absl::Status Evaluate(ReverseOp& op, const Tensor& operand, Tensor& output) {
  DISPATCH_BOOL_INT_FLOAT(EvaluateImpl, output.StorageType(), op, operand,
                          output);
}
}  // namespace shlo_ref
