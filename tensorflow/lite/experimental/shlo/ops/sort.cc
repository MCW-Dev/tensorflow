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

#include "tensorflow/lite/experimental/shlo/ops/sort.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

absl::Status CheckParameters(const std::vector<Tensor>& inputs,
                             int64_t dimension, bool is_stable,
                             std::vector<Tensor>& outputs) {
  if (inputs.empty()) {
    return absl::InvalidArgumentError(
        "There must be at least one input tensor.");
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].tensor_element_type() != outputs[i].tensor_element_type()) {
      return absl::FailedPreconditionError(
          "Input and result tensor types must be the same.");
    }
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].shape() != outputs[i].shape()) {
      return absl::FailedPreconditionError(
          "Input and result tensor shapes must be the same.");
    }
  }

  int64_t rank = inputs[0].Rank();
  if (dimension < -rank || dimension >= rank) {
    return absl::InvalidArgumentError("Dimension is out of range.");
  }

  return absl::OkStatus();
}

template <DataType storage_type>
absl::Status PrepareImpl(sortOp& op, const std::vector<Tensor>& inputs,
                         std::vector<Tensor>& outputs) {
  SHLO_REF_RETURN_ON_ERROR(CheckParameters(inputs, op.attributes.dimension,
                                           op.attributes.is_stable, outputs));

  op.args.resize(inputs.size());
  op.indices.resize(inputs[0].shape().Dim(op.attributes.dimension));
  return absl::OkStatus();
}

template <DataType storage_type>
absl::Status EvaluateImpl(sortOp& op, std::vector<Tensor>& inputs,
                          std::vector<Tensor>& outputs) {
  using StorageT = StorageType<storage_type>;

  Axis rank = inputs[0].Rank();

  const DimensionSize operand_size = outputs[0].NumElements();
  const Axis operand_rank = outputs[0].Rank();

  absl::InlinedVector<DimensionSize, kMaxNumDimensions> operand_index(
      operand_rank);
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> output_index(
      operand_rank);
  for (int64_t k = 0; k < operand_size; ++k) {
    outputs[0].GetNdIndex(k, operand_index);
    auto result_it = operand_index;
    if (result_it[op.attributes.dimension] != 0) continue;

    absl::c_iota(op.indices, 0);

    auto comparator_wrapper = [&](int64_t lhs_handle,
                                  int64_t rhs_handle) -> bool {
      auto lhs_index = result_it;
      auto rhs_index = result_it;
      lhs_index[op.attributes.dimension] = lhs_handle;
      rhs_index[op.attributes.dimension] = rhs_handle;

      for (int i = 0; i < op.args.size() / 2; ++i) {
        op.args[2 * i] = inputs[i].Get<storage_type>(lhs_index);
        op.args[(2 * i) + 1] = inputs[i].Get<storage_type>(rhs_index);
      }
      return op.attributes.comparator(op.args);
    };

    if (op.attributes.is_stable) {
      absl::c_stable_sort(op.indices, comparator_wrapper);
    } else {
      absl::c_sort(op.indices, comparator_wrapper);
    }

    for (size_t input_handle = 0; input_handle < op.indices.size();
         ++input_handle) {
      int64_t result_handle = op.indices[input_handle];
      for (size_t i = 0; i < inputs.size(); ++i) {
        auto input_index = result_it;
        auto result_index = result_it;
        input_index[op.attributes.dimension] = input_handle;
        result_index[op.attributes.dimension] = result_handle;
        StorageT element(inputs[i].Get<storage_type>(input_index));
        outputs[i].Set<storage_type>(result_index, element);
      }
    }
  }

  return absl::OkStatus();
}

sortOp Create(sortOp::Attributes attributes) { return sortOp(attributes); }

absl::Status Prepare(sortOp& op, const std::vector<Tensor>& inputs,
                     std::vector<Tensor>& outputs) {
  DISPATCH_INT_FLOAT(PrepareImpl, inputs[0].StorageType(), op, inputs, outputs);
  return absl::OkStatus();
}

absl::Status Evaluate(sortOp& op, std::vector<Tensor>& inputs,
                      std::vector<Tensor>& outputs) {
  DISPATCH_INT_FLOAT(EvaluateImpl, inputs[0].StorageType(), op, inputs,
                     outputs);
  return absl::FailedPreconditionError("Unsupported tensor type.");
}

}  // namespace shlo_ref
