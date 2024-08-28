/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
         //
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <cmath>

#include <cstdint>
#include <limits>
#include <numeric>

#include "Eigen/Core"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_reduce {

constexpr int kInputTensor = 0;
constexpr int kInitValue = 1;
constexpr int kOutputTensor = 0;
constexpr int kMaxStablehloReduceRank = 8;

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  
}

// Returns the output shape.
TfLiteStatus GetOutputShape(TfLiteContext* context,
                            const TfLiteIntArray* input_dims,
                            const int input_num_dims, const int64_t* axis,
                            const int64_t num_axis,
                            TfLiteIntArray** output_shape) {
  if (input_num_dims == 0) {
    *output_shape = TfLiteIntArrayCreate(0);
    return kTfLiteOk;
  }

  // Calculates size of reducing axis.
  int num_reduce_axis = num_axis;
  for (int i = 0; i < num_axis; ++i) {
    int current = axis[i];
    if (current < 0) {
      current += input_num_dims;
    }
    TF_LITE_ENSURE(context, current >= 0 && current < input_num_dims);
    for (int j = 0; j < i; ++j) {
      int previous = axis[j];
      if (previous < 0) {
        previous += input_num_dims;
      }
      if (current == previous) {
        --num_reduce_axis;
        break;
      }
    }

    // Determines output dimensions.
    TfLiteIntArray* output_dims =
        TfLiteIntArrayCreate(input_num_dims - num_reduce_axis);
    int num_skip_axis = 0;
    for (int idx = 0; idx < input_num_dims; ++idx) {
      bool is_axis = false;
      for (int axis_idx = 0; axis_idx < num_axis; ++axis_idx) {
        if (axis[axis_idx] == idx || axis[axis_idx] + input_num_dims == idx) {
          ++num_skip_axis;
          is_axis = true;
          break;
        }
      }
      if (!is_axis) {
        output_dims->data[idx - num_skip_axis] = input_dims->data[idx];
      }
    }
    *output_shape = output_dims;
    return kTfLiteOk;
  }
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));

  const TfLiteTensor* init_value;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInitValue, &init_value));

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));  
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  int input_rank = input->dims->size;
  int init_value_rank = init_value->dims->size;

  TF_LITE_ENSURE_MSG(context, node->inputs->size > 0,
                     "'stablehlo.reduce' Input should not be empty.");
  TF_LITE_ENSURE_MSG(context,
                     input_rank > 0 && input_rank <= kMaxStablehloReduceRank,
                     "'stablehlo.reduce' Input rank out of range.");
  TF_LITE_ENSURE_MSG(
      context,
      init_value_rank >= 0 && init_value_rank < kMaxStablehloReduceRank,
      "'stablehlo.reduce' Init Value rank out of range.");

  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto* subgraphs = this_subgraph->GetSubgraphs();

  const TfLiteStablehloReduceParams* data =
      reinterpret_cast<TfLiteStablehloReduceParams*>(node->builtin_data);
  if (data->body_subgraph_index >= subgraphs->size()) {
    TF_LITE_KERNEL_LOG(context,
                       "body subgraph not found for stablehlo.reduce.");
    return TfLiteStatus::kTfLiteError;
  }

  Subgraph* body_subgraph = (*subgraphs)[data->body_subgraph_index].get();
  TF_LITE_ENSURE_EQ(context, body_subgraph->outputs().size(), 1);

  for (int i = 0; i < node->inputs->size; ++i) {
    int input_idx = body_subgraph->inputs()[i];

    TfLiteTensor* body_subgraph_input = body_subgraph->tensor(input_idx);
    body_subgraph_input->params = input->params;
  }
  TfLiteTensor* body_subgraph_output =
      body_subgraph->tensor(body_subgraph->outputs()[0]);
  body_subgraph_output->params = output->params;

  TF_LITE_ENSURE_OK(context, body_subgraph->AllocateTensors());

  for (int idx = 0; idx < data->num_dimensions; ++idx) {
    TF_LITE_ENSURE_MSG(
        context,
        data->dimensions[idx] >= 0 && data->dimensions[idx] < input->dims->size,
        "'stablehlo.reduce' Dimension out of range.");
  }

  TfLiteIntArray* output_dims;
  TF_LITE_ENSURE_OK(
      context,
      GetOutputShape(context, input->dims, input->dims->size, data->dimensions,
                     data->num_dimensions, &output_dims));
  context->ResizeTensor(context, output, output_dims);

  return TfLiteStatus::kTfLiteOk;
}

}  // namespace stablehlo_reduce

TfLiteRegistration* Register_STABLEHLO_REDUCE() {
  static TfLiteRegistration r = {nullptr, nullptr, stablehlo_reduce::Prepare,
                                 stablehlo_reduce::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
