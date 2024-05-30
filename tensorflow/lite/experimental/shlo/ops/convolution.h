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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_CONVOLUTION_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_CONVOLUTION_H_

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

enum class PrecisionTypes {
  DEFAULT,
  HIGH,
  HIGHEST,
};

class ConvolutionOp {
 public:
  struct Attributes {
    absl::Span<const int64_t> window_strides;
    const Tensor padding;
    absl::Span<const int64_t> lhs_dilation;
    absl::Span<const int64_t> rhs_dilation;
    const int64_t input_batch_dimension;
    const int64_t input_feature_dimension;
    absl::Span<const int64_t> input_spatial_dimensions;
    const int64_t kernel_input_feature_dimension;
    const int64_t kernel_output_feature_dimension;
    absl::Span<const int64_t> kernel_spatial_dimensions;
    const int64_t output_batch_dimension;
    const int64_t output_feature_dimension;
    absl::Span<const int64_t> output_spatial_dimensions;
    const int64_t feature_group_count;
    const int64_t batch_group_count;
    std::array<PrecisionTypes, 2> precision_configs;
  };
  Attributes attributes;
  Tensor lhs_transposed;
  Tensor rhs_transposed;
  Tensor output_transposed;
  Tensor lhs_padded;
  std::vector<Tensor> lhs_splits;
  std::vector<Tensor> rhs_splits;
  absl::InlinedVector<Axis, kMaxNumDimensions> lhs_permutations;
  absl::InlinedVector<Axis, kMaxNumDimensions> rhs_permutations;
  absl::InlinedVector<Axis, kMaxNumDimensions> output_permutations;
  std::vector<std::byte> lhs_transposed_data;
  std::vector<std::byte> rhs_transposed_data;
  std::vector<std::byte> output_transposed_data;
  std::vector<std::byte> lhs_padded_data;
  std::vector<std::vector<std::byte>> lhs_splits_data;
  std::vector<std::vector<std::byte>> rhs_splits_data;
  absl::InlinedVector<int64_t, kMaxNumDimensions> pad_input_shape;
  absl::InlinedVector<int64_t, kMaxNumDimensions> pad_input_strides;
  absl::InlinedVector<int64_t, kMaxNumDimensions> pad_output_strides;
  int64_t pad_input_offset;
  int64_t pad_output_offset;
};

ConvolutionOp Create(const ConvolutionOp::Attributes& attributes);
absl::Status Prepare(ConvolutionOp& op, const Tensor& lhs, const Tensor& rhs,
                     Tensor& output);
absl::Status Evaluate(ConvolutionOp& op, const Tensor& lhs, const Tensor& rhs,
                      Tensor& output);

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_CONVOLUTION_H_