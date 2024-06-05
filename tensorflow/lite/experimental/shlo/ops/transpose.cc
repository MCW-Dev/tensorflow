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
#include "tensorflow/lite/experimental/shlo/ops/transpose.h"

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

absl::Status CheckParameters(const Tensor& operand,
                             absl::Span<const Axis> permutation,
                             Tensor& output) {
  if (operand.element_type() != output.element_type()) {
    return absl::FailedPreconditionError(
        "stablehlo.transpose: The element type of operand must be same as the "
        "element type of output.");
  }

  for (Axis perm : permutation) {
    if (perm < 0 || perm >= operand.Rank()) {
      return absl::FailedPreconditionError(
          "stablehlo.transpose: The permutation should be in the range of "
          "operand rank.");
    }
  }

  for (size_t i = 0; i < operand.Rank(); ++i) {
    if (output.shape().Dim(i) != operand.shape().Dim(permutation[i])) {
      return absl::FailedPreconditionError(
          "stablehlo.transpose: The output shape should be equal to the "
          "permutation of operand shape.");
    }
  }

  if (output.IsPerAxisQuantized()) {
    if (operand.quantized_per_axis_element_type().QuantizedDimension() !=
        permutation[output.quantized_per_axis_element_type()
                        .QuantizedDimension()]) {
      return absl::FailedPreconditionError(
          "stablehlo.transpose: The quantization dimension of operand should "
          "be equal to the permutation of quantization dimension of output.");
    }
  }
  return absl::OkStatus();
}

// This EvaluateImpl function will soon be replaced with an optimized version
template <DataType storage_type>
absl::Status EvaluateImpl(const Tensor& operand,
                          absl::Span<const Axis> permutation, Tensor& output) {
  using StorageT = StorageType<storage_type>;

  StorageT* output_buffer = output.GetDataAs<storage_type>();
  const DimensionSize operand_size = operand.NumElements();
  const Axis operand_rank = operand.Rank();

  absl::InlinedVector<DimensionSize, kMaxNumDimensions> operand_index(
      operand_rank);
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> output_index(
      operand_rank);

  for (DimensionSize k = 0; k < operand_size; ++k) {
    operand.GetNdIndex(k, operand_index);
    for (DimensionSize d = 0; d < operand_rank; ++d) {
      output_index[d] = operand_index[permutation[d]];
    }
    output.Set<storage_type>(output_index,
                             operand.Get<storage_type>(operand_index));
  }
  return absl::OkStatus();
}

TransposeOp Create(TransposeOp::Attributes attributes) {
  return {.attributes = attributes};
}

absl::Status Prepare(TransposeOp& op, const Tensor& operand, Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(
      CheckParameters(operand, op.attributes.permutation, output));

  return absl::OkStatus();
}

absl::Status Evaluate(TransposeOp& op, const Tensor& operand, Tensor& output) {
  DISPATCH_BOOL_INT_FLOAT(EvaluateImpl, output.StorageType(), operand,
                          op.attributes.permutation, output);
  return absl::FailedPreconditionError(
      "stablehlo.transpose: Unsupported tensor type.");
}
}  // namespace shlo_ref
