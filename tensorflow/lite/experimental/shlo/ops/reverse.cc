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

  StorageT* output_buffer = output.GetDataAs<storage_type>();
  StorageT* temp_operand_buffer = op.temp_operand.GetDataAs<storage_type>();

  const DimensionSize operand_size = operand.NumElements();
  const DimensionSize output_size = output.NumElements();
  const size_t operand_rank = operand.Rank();
  const size_t output_rank = output.Rank();
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> operand_new_shape(
      op.temp_operand.Rank());

  absl::c_copy(op.temp_operand.shape(), operand_new_shape.begin());

  if (op.attributes.dimensions.size() > 0) {
    for (size_t i = 0; i < op.new_dimensions.size(); ++i) {
      DimensionSize dimension = op.new_dimensions[i];
      DimensionSize upper_size = 1;
      for (size_t j = 0; j < dimension; ++j) {
        upper_size *= operand_new_shape[j];
      }
      DimensionSize lower_size = 1;
      for (size_t j = dimension + 1; j < operand_new_shape.size(); ++j) {
        lower_size *= operand_new_shape[j];
      }

      DimensionSize current_dim_size = operand_new_shape[dimension];

      if (lower_size > 1) {
        for (size_t i = 0; i < upper_size; ++i) {
          for (size_t j = 0; j < current_dim_size; ++j) {
            StorageT* src =
                temp_operand_buffer + (i * current_dim_size + j) * lower_size;
            StorageT* dst = output_buffer + (i * current_dim_size +
                                             (current_dim_size - j - 1)) *
                                                lower_size;
            std::memcpy(dst, src, lower_size * sizeof(StorageT));
          }
        }
      } else {
        for (size_t i = 0; i < upper_size; ++i) {
          std::reverse_copy(temp_operand_buffer + i * current_dim_size,
                            temp_operand_buffer + (i + 1) * current_dim_size,
                            output_buffer + i * current_dim_size);
        }
      }

      std::memcpy(temp_operand_buffer, output_buffer,
                  operand_size * sizeof(StorageT));
    }
  } else {
    std::memcpy(output_buffer, temp_operand_buffer,
                operand_size * sizeof(StorageT));
  }

  return absl::OkStatus();
}

absl::Status ReshapeOperand(ReverseOp& op, const Tensor& operand,
                            Tensor& output) {
  const DimensionSize operand_rank = operand.Rank();
  const DimensionSize operand_size = operand.NumElements();
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> operand_new_shape;
  absl::InlinedVector<bool, kMaxNumDimensions> dimensions_map(operand_rank,
                                                              false);

  for (size_t i = 0; i < op.attributes.dimensions.size(); ++i) {
    dimensions_map[op.attributes.dimensions[i]] = true;
  }

  bool reshaped_dimension = true;
  DimensionSize reshaped_size = 1;
  size_t index_offset = 0;
  if (op.attributes.dimensions.size() > 0) {
    for (size_t i = 0; i < operand_rank; ++i) {
      if (dimensions_map[i]) {
        if (i + 1 < operand_rank && dimensions_map[i + 1]) {
          if (reshaped_dimension == true) {
            op.new_dimensions.push_back(i - index_offset);
            reshaped_dimension = false;
          }
          reshaped_size *= operand.shape().Dim(i);
          index_offset += 1;
        } else {
          reshaped_size *= operand.shape().Dim(i);
          operand_new_shape.push_back(reshaped_size);
          reshaped_size = 1;
          if (reshaped_dimension) {
            op.new_dimensions.push_back(i - index_offset);
          }
          reshaped_dimension = true;
        }
      } else {
        operand_new_shape.push_back(operand.shape().Dim(i));
      }
    }

    if (reshaped_size != 1) {
      operand_new_shape.push_back(reshaped_size);
    }
  }
  op.temp_operand_data = std::vector<std::byte>(operand.SizeInBytes());
  Tensor temp_operand{.type = TensorType{.shape = Shape(operand_new_shape),
                                         .element_type = operand.StorageType()},
                      .data = op.temp_operand_data.data()};
  op.temp_operand = std::move(temp_operand);

  std::memcpy(temp_operand.data, operand.data, operand.SizeInBytes());

  return absl::OkStatus();
}

ReverseOp Create(ReverseOp::Attributes attributes) {
  return {.attributes = attributes};
}

absl::Status Prepare(ReverseOp& op, const Tensor& operand, Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(
      CheckParameters(operand, op.attributes.dimensions, output));

  SHLO_REF_RETURN_ON_ERROR(ReshapeOperand(op, operand, output));

  return absl::OkStatus();
}

absl::Status Evaluate(ReverseOp& op, const Tensor& operand, Tensor& output) {
  DISPATCH_BOOL_INT_FLOAT(EvaluateImpl, output.StorageType(), op, operand,
                          output);
}
}  // namespace shlo_ref
