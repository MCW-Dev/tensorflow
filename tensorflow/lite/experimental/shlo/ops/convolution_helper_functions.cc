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

#include "tensorflow/lite/experimental/shlo/ops/convolution_helper_functions.h"

#include "absl/types/span.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

bool IsUnique(const int64_t& batch_dimension, const int64_t& feature_dimension,
              absl::Span<const int64_t> operand) {
  std::unordered_set<int64_t> seen_elements;
  if (!seen_elements.insert(batch_dimension).second) {
    return false;
  }
  if (!seen_elements.insert(feature_dimension).second) {
    return false;
  }
  for (int64_t element : operand) {
    if (!seen_elements.insert(element).second) {
      return false;
    }
  }
  return true;
}

bool IsInRange(const int64_t& batch_dimension, const int64_t& feature_dimension,
               absl::Span<const int64_t> operand, size_t N) {
  auto is_in_range = [N](int64_t v) { return v >= 0 && v < N; };
  if (!is_in_range(batch_dimension) || !is_in_range(feature_dimension)) {
    return false;
  }
  return absl::c_all_of(operand, is_in_range);
}

bool IsGreaterThanZero(absl::Span<const int64_t> operand) {
  return absl::c_all_of(operand, [](int64_t x) { return x > 0; });
}

bool CheckOutputSpatial(ConvolutionOp& op, const Tensor& lhs, const Tensor& rhs,
                        const Tensor& output) {
  const int64_t* padding_buffer =
      op.attributes.padding.GetDataAs<DataType::kSI64>();
  for (size_t i = 0; i < output.Rank() - 2; ++i) {
    int64_t lhs_dim = lhs.shape().Dim(
        static_cast<Axis>(op.attributes.input_spatial_dimensions[i]));
    int64_t rhs_dim = rhs.shape().Dim(
        static_cast<Axis>(op.attributes.kernel_spatial_dimensions[i]));
    int64_t lhs_dilation = op.attributes.lhs_dilation[i];
    int64_t rhs_dilation = op.attributes.rhs_dilation[i];
    int64_t window_stride = op.attributes.window_strides[i];

    int64_t dilated_lhs_shape =
        (lhs_dim == 0) ? 0 : (lhs_dim - 1) * lhs_dilation + 1;
    int64_t padded_lhs_shape =
        dilated_lhs_shape + padding_buffer[2 * i] + padding_buffer[2 * i + 1];
    int64_t dilated_rhs_shape =
        (rhs_dim == 0) ? 0 : (rhs_dim - 1) * rhs_dilation + 1;

    bool is_empty_window =
        (padded_lhs_shape == 0 || dilated_rhs_shape > padded_lhs_shape);
    int64_t expected_output_shape =
        is_empty_window ? 0
                        : std::floor((padded_lhs_shape - dilated_rhs_shape) /
                                     window_stride) +
                              1;

    if (output.shape().Dim(
            static_cast<Axis>(op.attributes.output_spatial_dimensions[i])) !=
        expected_output_shape) {
      return false;
    }
  }

  return true;
}

}  // namespace shlo_ref