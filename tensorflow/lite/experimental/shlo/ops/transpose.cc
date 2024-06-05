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

template <DataType storage_type>
absl::Status PrepareImpl(TransposeOp& op, const Tensor& operand,
                         Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(
      CheckParameters(operand, op.attributes.permutation, output));
  using StorageT = StorageType<storage_type>;
  const DimensionSize operand_size = operand.NumElements();
  const DimensionSize output_size = output.NumElements();
  const size_t permutation_size = op.attributes.permutation.size();
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> reshaped_input(
      operand.Rank());
  Shape expected_reshaped_operand_shape;
  Shape reshaped_output_shape;

  for (DimensionSize i = 0; i < operand.Rank(); ++i) {
    reshaped_input[i] = operand.shape().Dim(i);
  }
  absl::InlinedVector<Axis, kMaxNumDimensions> reshaped_permutation(
      op.attributes.permutation.begin(), op.attributes.permutation.end());
  size_t max_dimensions = reshaped_permutation.size();
  for (size_t i = 0; i < max_dimensions - 1; ++i) {
    if (reshaped_permutation[i] == reshaped_permutation[i + 1] - 1) {
      size_t c = reshaped_permutation[i + 1];
      reshaped_input[reshaped_permutation[i]] *= reshaped_input[c];
      for (size_t j = i + 1; j < max_dimensions - 1; ++j) {
        reshaped_permutation[j] = reshaped_permutation[j + 1];
      }
      for (size_t k = c; k < max_dimensions - 1; ++k) {
        reshaped_input[k] = reshaped_input[k + 1];
      }
      for (size_t j = 0; j < max_dimensions - 1; ++j) {
        if (reshaped_permutation[j] > c) {
          reshaped_permutation[j] -= 1;
        }
      }

      --max_dimensions;
      --i;
    }
  }
  reshaped_permutation.resize(max_dimensions);
  reshaped_input.resize(max_dimensions);

  absl::InlinedVector<DimensionSize, kMaxNumDimensions> reshaped_output(
      reshaped_permutation.size());
  if (reshaped_permutation.size() != permutation_size) {
    for (DimensionSize i = 0; i < reshaped_permutation.size(); ++i) {
      reshaped_output[i] = (reshaped_input[reshaped_permutation[i]]);
    }
    expected_reshaped_operand_shape = Shape(reshaped_input);
    reshaped_output_shape = Shape(reshaped_output);
  } else {
    expected_reshaped_operand_shape = operand.shape();
    reshaped_output_shape = output.shape();
  }
  op.output_reshaped_data_buffer =
      std::vector<std::byte>(output.NumElements() * sizeof(StorageT));
  Tensor operand_reshaped{
      .type = TensorType{.shape = expected_reshaped_operand_shape,
                         .element_type = storage_type},
      .data = operand.data};
  Tensor output_reshaped{.type = TensorType{.shape = reshaped_output_shape,
                                            .element_type = storage_type},
                         .data = op.output_reshaped_data_buffer.data()};

  op.operand_reshaped = std::move(operand_reshaped);
  op.output_reshaped = std::move(output_reshaped);
  op.reshaped_permutation = std::move(reshaped_permutation);
  return absl::OkStatus();
}

template <DataType storage_type>
absl::Status EvaluateImpl(TransposeOp& op, const Tensor& operand,
                          Tensor& output) {
  using StorageT = StorageType<storage_type>;
  const DimensionSize operand_size = op.operand_reshaped.NumElements();
  const StorageT* input_data = op.operand_reshaped.GetDataAs<storage_type>();
  StorageT* output_data = output.GetDataAs<storage_type>();
  StorageT* output_reshaped_data_buffer =
      op.output_reshaped.GetDataAs<storage_type>();

  if (op.operand_reshaped.Rank() == 1) {
    std::copy(input_data, input_data + operand_size, output_data);
    return absl::OkStatus();
  }

  if (op.operand_reshaped.Rank() == 2) {
    // Optimized 2D transpose
    const DimensionSize rows = op.operand_reshaped.shape().Dim(0);
    const DimensionSize cols = op.operand_reshaped.shape().Dim(1);
    const DimensionSize blockSize = 4;
    for (DimensionSize i = 0; i < rows; i += blockSize) {
      for (DimensionSize j = 0; j < cols; j += blockSize) {
        for (DimensionSize k = i; k < std::min(i + blockSize, rows); ++k) {
          for (DimensionSize l = j; l < std::min(j + blockSize, cols); ++l) {
            output_reshaped_data_buffer[l * rows + k] =
                input_data[k * cols + l];
          }
        }
      }
    }
    std::memcpy(output.data, op.output_reshaped.data, output.SizeInBytes());
    return absl::OkStatus();
  }

  if (op.operand_reshaped.Rank() == 3) {
    DimensionSize input_dim2, input_dim3;
    input_dim2 = op.operand_reshaped.shape().Dim(1);
    input_dim3 = op.operand_reshaped.shape().Dim(2);
    DimensionSize stride_dim1, stride_dim2, stride_dim3;
    if (op.reshaped_permutation[0] == 2) {
      stride_dim1 = 1;
    } else if (op.reshaped_permutation[1] == 2) {
      stride_dim2 = 1;
    } else {
      stride_dim3 = 1;
    }
    if (op.reshaped_permutation[0] == 1) {
      stride_dim1 = input_dim3;
    } else if (op.reshaped_permutation[1] == 1) {
      stride_dim2 = input_dim3;
    } else {
      stride_dim3 = input_dim3;
    }

    if (op.reshaped_permutation[0] == 0) {
      stride_dim1 = input_dim2 * input_dim3;
    } else if (op.reshaped_permutation[1] == 0) {
      stride_dim2 = input_dim2 * input_dim3;
    } else {
      stride_dim3 = input_dim2 * input_dim3;
    }
    DimensionSize output_dims[3];
    output_dims[0] =
        op.operand_reshaped.shape().Dim(op.reshaped_permutation[0]);
    output_dims[1] =
        op.operand_reshaped.shape().Dim(op.reshaped_permutation[1]);
    output_dims[2] =
        op.operand_reshaped.shape().Dim(op.reshaped_permutation[2]);
    for (DimensionSize i1 = 0; i1 < output_dims[0]; ++i1) {
      for (DimensionSize i2 = 0; i2 < output_dims[1]; ++i2) {
        for (DimensionSize i3 = 0; i3 < output_dims[2]; ++i3) {
          const DimensionSize i =
              i1 * stride_dim1 + i2 * stride_dim2 + i3 * stride_dim3;
          const DimensionSize o =
              i1 * output_dims[1] * output_dims[2] + i2 * output_dims[2] + i3;
          output_reshaped_data_buffer[o] = input_data[i];
        }
      }
    }
    for (DimensionSize i = 0; i < output.NumElements(); ++i) {
      output_data[i] = output_reshaped_data_buffer[i];
    }
    return absl::OkStatus();
  }

  // Fall back to the general implementation if not 2D/3D
  const Axis operand_rank = op.operand_reshaped.Rank();
  const Shape output_shape = op.output_reshaped.shape();
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> input_stride(
      operand_rank);
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> output_stride(
      operand_rank);
  DimensionSize stride = 1;
  DimensionSize stride_output = 1;
  for (int64_t i = static_cast<int64_t>(operand_rank) - 1; i >= 0; --i) {
    input_stride[i] = stride;
    output_stride[i] = stride_output;
    stride *= op.operand_reshaped.shape().Dim(i);
    stride_output *= op.output_reshaped.shape().Dim(i);
  }

  TransposeImpl(0, operand_rank, op.reshaped_permutation.data(), input_data,
                input_stride.data(), output_reshaped_data_buffer,
                output_stride.data(), output_shape);
  for (DimensionSize i = 0; i < output.NumElements(); ++i) {
    output_data[i] = output_reshaped_data_buffer[i];
  }
  return absl::OkStatus();
}

template <typename StorageT>
void TransposeImpl(const Axis depth, const Axis dims, const Axis* perm,
                   const StorageT* input_data,
                   const DimensionSize* input_stride, StorageT* output_data,
                   const DimensionSize* output_stride,
                   const Shape output_shape) {
  const DimensionSize dimension_size = output_shape.Dim(depth);

  if (depth == dims - 1) {
    const DimensionSize loop_stride = input_stride[perm[depth]];
    for (DimensionSize i = 0; i < dimension_size; ++i) {
      output_data[i] = *input_data;
      input_data += loop_stride;
    }
  } else {
    for (DimensionSize i = 0; i < dimension_size; ++i) {
      TransposeImpl(depth + 1, dims, perm, input_data, input_stride,
                    output_data, output_stride, output_shape);

      input_data += input_stride[perm[depth]];
      output_data += output_stride[depth];
    }
  }
}

TransposeOp Create(TransposeOp::Attributes attributes) {
  return {.attributes = attributes};
}

absl::Status Prepare(TransposeOp& op, const Tensor& operand, Tensor& output) {
  DISPATCH_BOOL_INT_FLOAT(PrepareImpl, operand.StorageType(), op, operand,
                          output);

  return absl::OkStatus();
}

absl::Status Evaluate(TransposeOp& op, const Tensor& operand, Tensor& output) {
  DISPATCH_BOOL_INT_FLOAT(EvaluateImpl, output.StorageType(), op, operand,
                          output);
  return absl::FailedPreconditionError(
      "stablehlo.transpose: Unsupported tensor type.");
}
}  // namespace shlo_ref
