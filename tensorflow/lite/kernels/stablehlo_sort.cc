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

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "Eigen/Core"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/tensor_slice_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_sort {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

template <typename DataType>
TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node,
                      const std::vector<const TfLiteTensor*>& operands,
                      const int dimension, const bool is_stable,
                      std::vector<TfLiteTensor*>& results) {
  const TfLiteStablehloSortParams* data =
      reinterpret_cast<TfLiteStablehloSortParams*>(node->builtin_data);
  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto* subgraphs = this_subgraph->GetSubgraphs();
  Subgraph& comparator_subgraph =
      *(*subgraphs)[data->comparator_subgraph_index];

  const TfLiteTensor* operand = operands[0];
  const int rank = NumDimensions(operand);
  const int operand_size = NumElements(operand);
  const int dim_size = GetTensorShape(operand).Dims(dimension);
  std::vector<int> indices(dim_size);
  std::vector<int> current_index(rank, 0);
  for (int k = 0; k < operand_size; ++k) {
    if (current_index[dimension] != 0) {
      NextIndex(rank, GetTensorShape(operand).DimsData(), current_index.data());
      continue;
    }
    std::iota(indices.begin(), indices.end(), 0);

    auto comparator_wrapper = [&](int lhs_indices, int rhs_indices) -> bool {
      auto lhs_index = current_index;
      auto rhs_index = current_index;
      lhs_index[dimension] = lhs_indices;
      rhs_index[dimension] = rhs_indices;
      const DataType* operand_data = GetTensorData<DataType>(operand);
      int flat_lhs_index = TensorIndexToFlat(lhs_index.data(), lhs_index.size(),
                                             GetTensorShape(operand));
      int flat_rhs_index = TensorIndexToFlat(rhs_index.data(), rhs_index.size(),
                                             GetTensorShape(operand));
      for (int i = 0; i < comparator_subgraph.inputs().size(); ++i) {
        TfLiteTensor* subgraph_input =
            comparator_subgraph.tensor(comparator_subgraph.inputs()[i]);
        std::memcpy(
            subgraph_input->data.raw,
            (i % 2 == 0
                 ? &GetTensorData<DataType>(operands[i / 2])[flat_lhs_index]
                 : &GetTensorData<DataType>(operands[i / 2])[flat_rhs_index]),
            sizeof(DataType));
      }

      TF_LITE_ENSURE_OK(context, comparator_subgraph.Invoke());
      TfLiteTensor* output_tensor1 =
          comparator_subgraph.tensor(comparator_subgraph.outputs()[0]);
      const bool* output_tensor1_data = GetTensorData<bool>(output_tensor1);
      return output_tensor1_data[0];
    };

    if (is_stable) {
      std::stable_sort(indices.begin(), indices.end(), comparator_wrapper);
    } else {
      std::sort(indices.begin(), indices.end(), comparator_wrapper);
    }

    for (int operand_handle = 0; operand_handle < indices.size();
         ++operand_handle) {
      int64_t result_handle = indices[operand_handle];
      for (int i = 0; i < operands.size(); ++i) {
        auto operand_index = current_index;
        auto output_index = current_index;
        operand_index[dimension] = operand_handle;
        output_index[dimension] = result_handle;
        const DataType* operand_data = GetTensorData<DataType>(operands[i]);
        int flat_operand_index =
            TensorIndexToFlat(operand_index.data(), operand_index.size(),
                              GetTensorShape(operand));
        DataType operand_val = operand_data[flat_operand_index];
        DataType* output_data = GetTensorData<DataType>(results[i]);
        int flat_output_index =
            TensorIndexToFlat(output_index.data(), output_index.size(),
                              GetTensorShape(results[i]));
        output_data[flat_output_index] = operand_val;
      }
    }

    NextIndex(rank, GetTensorShape(operand).DimsData(), current_index.data());
  }
  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  int input_rank = input->dims->size;
  RuntimeShape input_shape = GetTensorShape(input);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  int result_rank = output->dims->size;
  RuntimeShape result_runtime_shape(result_rank, output->dims->data);
  context->ResizeTensor(context, output, TfLiteIntArrayCopy(input->dims));

  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto* subgraphs = this_subgraph->GetSubgraphs();
  const TfLiteStablehloSortParams* data =
      reinterpret_cast<TfLiteStablehloSortParams*>(node->builtin_data);
  if (data->comparator_subgraph_index >= subgraphs->size()) {
    TF_LITE_KERNEL_LOG(context,
                       "Comparator subgraph not found for stablehlo.sort.");
    return TfLiteStatus::kTfLiteError;
  }
  Subgraph* comparator_subgraph =
      (*subgraphs)[data->comparator_subgraph_index].get();
  TF_LITE_ENSURE_EQ(context, comparator_subgraph->outputs().size(), 1);
  for (int i = 0; i < node->inputs->size; ++i) {
    int input_idx = comparator_subgraph->inputs()[i];
    TfLiteTensor* comparator_subgraph_input =
        comparator_subgraph->tensor(input_idx);
    comparator_subgraph_input->params.scale = input->params.scale;
    comparator_subgraph_input->params.zero_point = input->params.zero_point;
    TF_LITE_ENSURE_OK(context, comparator_subgraph->AllocateTensors());
  }
  TF_LITE_ENSURE_MSG(context, node->inputs->size > 0,
                     "'stablehlo.sort' Input should not be empty.");

  for (int i = 0; i < node->inputs->size; ++i) {
    const TfLiteTensor* input_tensor;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, i, &input_tensor));
    TfLiteTensor* output_tensor;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &output_tensor));
    TF_LITE_ENSURE_MSG(
        context, input_tensor->type == output_tensor->type,
        "'stablehlo.sort' Input and Output tensor types must be the same.");
    TF_LITE_ENSURE_MSG(
        context, input_tensor->dims->size == output_tensor->dims->size,
        "'stablehlo.sort' Input and Output tensor shapes must be the same.");
    for (int j = 0; j < input_tensor->dims->size; ++j) {
      TF_LITE_ENSURE_MSG(
          context, input_tensor->dims->data[j] == output_tensor->dims->data[j],
          "'stablehlo.sort' Input and Output tensor shapes must be the "
          "same.");
    }
  }

  TF_LITE_ENSURE_MSG(context,
                     data->dimension >= -input->dims->size &&
                         data->dimension < input->dims->size,
                     "'stablehlo.sort' Dimension out of range.");
  return TfLiteStatus::kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  std::vector<const TfLiteTensor*> inputs;
  for (int i = 0; i < node->inputs->size; ++i) {
    const TfLiteTensor* input;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, i, &input));
    inputs.push_back(input);
  }
  std::vector<TfLiteTensor*> outputs;
  for (int i = 0; i < node->outputs->size; ++i) {
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &output));
    outputs.push_back(output);
  }
  TfLiteType data_type = inputs[0]->type;
  const TfLiteStablehloSortParams* data =
      reinterpret_cast<TfLiteStablehloSortParams*>(node->builtin_data);
  int64_t dimension = data->dimension;
  bool is_stable = data->is_stable;

  if (data_type == kTfLiteFloat32) {
    return EvalImpl<float>(context, node, inputs, dimension, is_stable,
                           outputs);
  } else if (data_type == kTfLiteFloat16) {
    return EvalImpl<Eigen::half>(context, node, inputs, dimension, is_stable,
                                 outputs);
  } else if (data_type == kTfLiteBFloat16) {
    return EvalImpl<Eigen::bfloat16>(context, node, inputs, dimension,
                                     is_stable, outputs);
  } else if (data_type == kTfLiteInt8) {
    return EvalImpl<int8_t>(context, node, inputs, dimension, is_stable,
                            outputs);
  } else if (data_type == kTfLiteInt16) {
    return EvalImpl<int16_t>(context, node, inputs, dimension, is_stable,
                             outputs);
  } else if (data_type == kTfLiteInt32) {
    return EvalImpl<int32_t>(context, node, inputs, dimension, is_stable,
                             outputs);
  } else {
    TF_LITE_KERNEL_LOG(context, "(Index Type: %s) currently not supported.\n",
                       TfLiteTypeGetName(data_type));
    return TfLiteStatus::kTfLiteError;
  }
}

}  // namespace stablehlo_sort

TfLiteRegistration* Register_STABLEHLO_SORT() {
  static TfLiteRegistration r = {nullptr, nullptr, stablehlo_sort::Prepare,
                                 stablehlo_sort::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
