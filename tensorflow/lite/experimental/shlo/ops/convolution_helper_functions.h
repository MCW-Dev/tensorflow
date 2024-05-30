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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_CONVOLUTION_HELPER_FUNCTIONS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_CONVOLUTION_HELPER_FUNCTIONS_H_

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/ops/convolution.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

bool IsUnique(const int64_t& batch_dimension, const int64_t& feature_dimension,
              absl::Span<const int64_t> operand);

bool IsInRange(const int64_t& batch_dimension, const int64_t& feature_dimension,
               absl::Span<const int64_t> operand, size_t N);

bool IsGreaterThanZero(absl::Span<const int64_t> operand);

bool CheckOutputSpatial(ConvolutionOp& op, const Tensor& lhs, const Tensor& rhs,
                        const Tensor& output);

absl::InlinedVector<DimensionSize, kMaxNumDimensions> GenerateIndices(
    int i, absl::Span<const DimensionSize> temp);

// Transpose op implementation
template <DataType data_type>
inline absl::Status TransposeImpl(const Tensor& operand,
                                  absl::Span<const Axis> permutation,
                                  Tensor& output) {
  using StorageT = StorageType<data_type>;
  if (permutation.size() != operand.Rank()) {
    return absl::FailedPreconditionError(
        "Rank of output and permutation doesn't match");
  }
  const StorageT* operand_buffer = operand.GetDataAs<data_type>();
  StorageT* output_buffer = output.GetDataAs<data_type>();

  int64_t operand_product = 1, output_product = 1;
  for (int64_t i = 0; i < operand.Rank(); ++i) {
    operand_product *= operand.shape().Dim(i);
    output_product *= output.shape().Dim(i);
  }

  absl::InlinedVector<DimensionSize, kMaxNumDimensions> temp;
  for (int64_t i = 0; i < operand.Rank(); ++i) {
    temp.push_back(operand.shape().Dim(i));
  }

  for (size_t k = 0; k < operand.NumElements(); ++k) {
    absl::InlinedVector<DimensionSize, kMaxNumDimensions> operand_index =
        GenerateIndices(k, temp);
    absl::InlinedVector<DimensionSize, kMaxNumDimensions> output_index(
        output.Rank(), 0);
    for (size_t d = 0; d < output.Rank(); ++d) {
      output_index[d] = operand_index[permutation[d]];
    }

    int operand_element_index = 0, output_element_index = 0;
    int64_t temp1 = 1, temp2 = 1;
    for (int64_t i = 0; i < operand.Rank(); i++) {
      temp1 *= operand.shape().Dim(i);
      operand_element_index += operand_index[i] * (operand_product / temp1);
      temp2 *= output.shape().Dim(i);
      output_element_index += output_index[i] * (output_product / temp2);
    }
    output_buffer[output_element_index] = operand_buffer[operand_element_index];
  }
  return absl::OkStatus();
}

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_CONVOLUTION_HELPER_FUNCTIONS_H_