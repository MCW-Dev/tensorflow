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

#include "tensorflow/lite/experimental/shlo/ops/slice.h"

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

bool CheckRange(absl::Span<const int64_t> start_indices,
                absl::Span<const int64_t> limit_indices, const Shape& shape) {
  return absl::c_all_of(start_indices, [&start_indices, &limit_indices, &shape,
                                        index = 0](int64_t x) mutable {
    return (0 <= x && x <= limit_indices[index] &&
            limit_indices[index] <= shape.Dim(index++));
  });
}

bool IsGreaterThanZero(absl::Span<const int64_t> operand) {
  return absl::c_all_of(operand, [](int64_t x) { return x > 0; });
}

bool CheckOutputDimensions(SliceOp& op, const Shape& shape) {
  DimensionSize expected_shape;
  for (size_t i = 0; i < shape.size(); ++i) {
    expected_shape =
        std::ceil(static_cast<double>(op.attributes.limit_indices[i] -
                                      op.attributes.start_indices[i]) /
                  op.attributes.strides[i]);
    if (expected_shape != shape.Dim(i)) {
      return false;
    }
  }
  return true;
}

// Constraints Check
absl::Status CheckParameters(SliceOp& op, const Tensor& operand,
                             Tensor& output) {
  if (operand.element_type() != output.element_type()) {
    return absl::FailedPreconditionError(
        "stablehlo.slice: The element type of operand must be same as the "
        "element type of output.");
  }
  if (op.attributes.start_indices.size() !=
          op.attributes.limit_indices.size() ||
      op.attributes.limit_indices.size() != op.attributes.strides.size() ||
      op.attributes.strides.size() != operand.Rank()) {
    return absl::FailedPreconditionError(
        "stablehlo.slice: Size of start_indices, limit_indices and strides "
        "must be same as rank(operand)");
  }
  if (!CheckRange(op.attributes.start_indices, op.attributes.limit_indices,
                  operand.shape())) {
    return absl::FailedPreconditionError(
        "stablehlo.slice: Range must be 0 <= start_indices[i] <= limit_indices "
        "<= shape(operand)");
  }
  if (!IsGreaterThanZero(op.attributes.strides)) {
    return absl::FailedPreconditionError(
        "stablehlo.slice: Strides must be > 0");
  }
  if (!CheckOutputDimensions(op, output.shape())) {
    return absl::FailedPreconditionError(
        "stablehlo.slice: Output shape is not properly set");
  }
  return absl::OkStatus();
}

template <DataType storage_type>
absl::Status EvaluateImpl(SliceOp& op, const Tensor& operand, Tensor& output) {
  using StorageT = StorageType<storage_type>;
  const StorageT* operand_buffer = operand.GetDataAs<storage_type>();
  StorageT* output_buffer = output.GetDataAs<storage_type>();

  absl::InlinedVector<DimensionSize, kMaxNumDimensions> start_indices(
      kMaxNumDimensions, 0);
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> stop_indices(
      kMaxNumDimensions, 1);
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> input_dims(
      kMaxNumDimensions, 1);
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> offsets(
      kMaxNumDimensions, 0);
  size_t copy_offset = 6 - operand.Rank();

  absl::c_copy(op.attributes.start_indices,
               start_indices.begin() + copy_offset);
  absl::c_copy(output.shape(), stop_indices.begin() + copy_offset);
  absl::c_copy(operand.shape(), input_dims.begin() + copy_offset);
  absl::c_copy(op.attributes.strides, offsets.begin() + copy_offset);

  for (DimensionSize i = 0; i < stop_indices[0]; ++i) {
    for (DimensionSize j = 0; j < stop_indices[1]; ++j) {
      for (DimensionSize k = 0; k < stop_indices[2]; ++k) {
        for (DimensionSize l = 0; l < stop_indices[3]; ++l) {
          for (DimensionSize m = 0; m < stop_indices[4]; ++m) {
            for (DimensionSize n = 0; n < stop_indices[5]; ++n) {
              size_t output_index =
                  ((((i * stop_indices[1] + j) * stop_indices[2] + k) *
                        stop_indices[3] +
                    l) *
                       stop_indices[4] +
                   m) *
                      stop_indices[5] +
                  n;
              size_t input_index =
                  (((((offsets[0] * i + start_indices[0]) * input_dims[1] +
                      (offsets[1] * j + start_indices[1])) *
                         input_dims[2] +
                     (offsets[2] * k + start_indices[2])) *
                        input_dims[3] +
                    (offsets[3] * l + start_indices[3])) *
                       input_dims[4] +
                   (offsets[4] * m + start_indices[4])) *
                      input_dims[5] +
                  (offsets[5] * n + start_indices[5]);
              output_buffer[output_index] = operand_buffer[input_index];
            }
          }
        }
      }
    }
  }

  return absl::OkStatus();
}

SliceOp Create(SliceOp::Attributes attributes) {
  return {.attributes = attributes};
}

absl::Status Prepare(SliceOp& op, const Tensor& operand, Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(CheckParameters(op, operand, output));
  return absl::OkStatus();
}

absl::Status Evaluate(SliceOp& op, const Tensor& operand, Tensor& output) {
  DISPATCH_BOOL_INT_FLOAT(EvaluateImpl, output.StorageType(), op, operand,
                          output);
}
}  // namespace shlo_ref