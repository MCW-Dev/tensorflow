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

int64_t DivNegRoundAwayOrZero(int64_t num, int64_t denum);

// Split function implementation
template <DataType storage_type>
inline void SliceImpl(const Tensor& operand, int64_t num_outputs,
                      size_t start_indices, size_t inner_dimensions_size,
                      size_t outer_dimensions_size, int64_t dimension,
                      Tensor& output) {
  using StorageT = StorageType<storage_type>;

  StorageT* output_buffer = output.GetDataAs<storage_type>();
  const StorageT* operand_buffer = operand.GetDataAs<storage_type>();

  size_t i = start_indices;
  size_t k = 0;
  const size_t size = (output.shape().Dim(dimension) * inner_dimensions_size);
  while (i < operand.NumElements()) {
    for (size_t j = 0; j < size; ++j, ++k) {
      output_buffer[k] = operand_buffer[i + j];
    }
    i += outer_dimensions_size;
  }
}

template <DataType storage_type>
inline void Split(const Tensor& operand, int64_t num_outputs, int64_t dimension,
                  std::vector<Tensor>& outputs) {
  size_t start_indices = 0;
  size_t inner_dimensions_size = 1;
  size_t outer_dimensions_size = 1;
  size_t dimension_size = operand.shape().Dim(dimension) / num_outputs;

  for (size_t i = operand.Rank() - 1; i > dimension; --i) {
    inner_dimensions_size *= operand.shape().Dim(i);
  }
  outer_dimensions_size *=
      inner_dimensions_size * operand.shape().Dim(dimension);

  for (int64_t i = 0; i < num_outputs; ++i) {
    start_indices = (i)*dimension_size * inner_dimensions_size;
    SliceImpl<storage_type>(operand, num_outputs, start_indices,
                            inner_dimensions_size, outer_dimensions_size,
                            dimension, outputs[i]);
  }
}

// Padding function implementation
template <typename StorageT>
inline void StridedCopy(const int rank, const StorageT* input,
                        const int64_t* input_shape,
                        const int64_t* input_strides, StorageT* output,
                        const int64_t* output_strides,
                        const int64_t element_size, const int depth) {
  if (depth + 1 == rank) {
    for (int64_t i = 0; i < input_shape[depth]; ++i) {
      std::memcpy(output, input, element_size);
      input += input_strides[depth];
      output += output_strides[depth];
    }
  } else {
    for (int64_t i = 0; i < input_shape[depth]; ++i) {
      StridedCopy<StorageT>(rank, input, input_shape, input_strides, output,
                            output_strides, element_size, depth + 1);
      input += input_strides[depth];
      output += output_strides[depth];
    }
  }
}

template <DataType storage_type>
inline void PaddingImpl(ConvolutionOp& op, const Tensor& operand) {
  using StorageT = StorageType<storage_type>;

  const StorageT* operand_buffer = operand.GetDataAs<storage_type>();
  StorageT* output_buffer = op.lhs_padded.GetDataAs<storage_type>();
  StridedCopy<StorageT>(operand.Rank(), operand_buffer + op.pad_input_offset,
                        op.pad_input_shape.begin(),
                        op.pad_input_strides.begin(),
                        output_buffer + op.pad_output_offset,
                        op.pad_output_strides.begin(), sizeof(StorageT),
                        /*depth=*/0);
}

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_CONVOLUTION_HELPER_FUNCTIONS_H_