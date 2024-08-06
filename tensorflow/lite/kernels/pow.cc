/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <stddef.h>
#include <stdint.h>

#include <cstdint>

#include "Eigen/Core"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/dequantize.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace pow {
namespace {

// Input/output tensor index.
constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;
constexpr int kNumTempTensorsForQuantization = 3;

// Op data for pow op.
struct OpData {
  bool requires_broadcast;
  int scratch_tensor_index;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = new OpData;
  data->requires_broadcast = false;
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  TF_LITE_ENSURE_TYPES_EQ(context, input1->type, input2->type);

  const TfLiteType type = input1->type;
  if (type != kTfLiteInt32 && type != kTfLiteFloat32 &&
      type != kTfLiteFloat16 && type != kTfLiteBFloat16 &&
      type != kTfLiteInt8 && type != kTfLiteInt16) {
    TF_LITE_KERNEL_LOG(context, "Unsupported data type %s.",
                       TfLiteTypeGetName(type));
    return kTfLiteError;
  }
  output->type = type;

  data->requires_broadcast = !HaveSameShapes(input1, input2);

  // quantize prepare
  if (input1->type == kTfLiteInt8 || input1->type == kTfLiteInt16) {
    context->AddTensors(context, kNumTempTensorsForQuantization,
                        &data->scratch_tensor_index);
    TfLiteIntArrayFree(node->temporaries);
    node->temporaries = TfLiteIntArrayCreate(kNumTempTensorsForQuantization);
    node->temporaries->data[0] = data->scratch_tensor_index;
    TfLiteIntArray* base_dequantize_shape =
        TfLiteIntArrayCreate(input1->dims->size);
    for (int i = 0; i < input1->dims->size; ++i) {
      base_dequantize_shape->data[i] = input1->dims->data[i];
    }
    node->temporaries->data[0] = data->scratch_tensor_index;
    TfLiteTensor* base_dequantize;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/0,
                                                &base_dequantize));
    base_dequantize->type = kTfLiteFloat32;
    base_dequantize->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, base_dequantize,
                                                     base_dequantize_shape));

    TfLiteIntArray* exponent_dequantize_shape =
        TfLiteIntArrayCreate(input2->dims->size);
    for (int i = 0; i < input2->dims->size; ++i) {
      exponent_dequantize_shape->data[i] = input2->dims->data[i];
    }
    node->temporaries->data[1] = data->scratch_tensor_index + 1;
    TfLiteTensor* exponent_dequantize;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/1,
                                                &exponent_dequantize));
    exponent_dequantize->type = kTfLiteFloat32;
    exponent_dequantize->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, exponent_dequantize,
                                            exponent_dequantize_shape));
    TfLiteIntArray* output_dequantize_shape =
        TfLiteIntArrayCreate(input1->dims->size);
    for (int i = 0; i < input1->dims->size; ++i) {
      output_dequantize_shape->data[i] = input1->dims->data[i];
    }
    node->temporaries->data[2] = data->scratch_tensor_index + 2;
    TfLiteTensor* output_dequantize;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/2,
                                                &output_dequantize));
    output_dequantize->type = kTfLiteFloat32;
    output_dequantize->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, output_dequantize,
                                                     output_dequantize_shape));
  }
  TfLiteIntArray* output_size = nullptr;
  if (data->requires_broadcast) {
    TF_LITE_ENSURE_OK(context, CalculateShapeForBroadcast(
                                   context, input1, input2, &output_size));
  } else {
    output_size = TfLiteIntArrayCopy(input1->dims);
  }

  return context->ResizeTensor(context, output, output_size);
}

template <typename T>
void PowImpl(const TfLiteTensor* input1, const TfLiteTensor* input2,
             TfLiteTensor* output, bool requires_broadcast) {
  if (requires_broadcast) {
    optimized_ops::BroadcastPow4D(
        GetTensorShape(input1), GetTensorData<T>(input1),
        GetTensorShape(input2), GetTensorData<T>(input2),
        GetTensorShape(output), GetTensorData<T>(output));
  } else {
    reference_ops::Pow(GetTensorShape(input1), GetTensorData<T>(input1),
                       GetTensorShape(input2), GetTensorData<T>(input2),
                       GetTensorShape(output), GetTensorData<T>(output));
  }
}

TfLiteStatus CheckValue(TfLiteContext* context, const TfLiteTensor* input) {
  const int64_t num_elements = NumElements(input);
  const int32_t* data = GetTensorData<int32_t>(input);
  for (int i = 0; i < num_elements; ++i) {
    if (data[i] < 0) {
      TF_LITE_KERNEL_LOG(context,
                         "POW does not support negative value for int32.");
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

template <typename DataType>
TfLiteStatus EvalQuantize(TfLiteContext* context, TfLiteNode* node,
                          const TfLiteTensor* input1,
                          const TfLiteTensor* input2, TfLiteTensor* output,
                          bool requires_broadcast) {
  TfLiteTensor* base_dequantize = GetTemporary(context, node, 0);
  TfLiteTensor* exponent_dequantize = GetTemporary(context, node, 1);
  TfLiteTensor* output_dequantize = GetTemporary(context, node, 2);

  dequantize::DequantizeImpl<dequantize::KernelType::kGenericOptimized>(
      context, node, input1, base_dequantize);
  dequantize::DequantizeImpl<dequantize::KernelType::kGenericOptimized>(
      context, node, input2, exponent_dequantize);
  PowImpl<float>(base_dequantize, exponent_dequantize, output_dequantize,
                 requires_broadcast);

  RuntimeShape output_shape(GetTensorShape(output));
  RuntimeShape output_dequantize_shape(GetTensorShape(output_dequantize));
  tflite::QuantizationParams op_params;
  op_params.zero_point = output->params.zero_point;
  op_params.scale = output->params.scale;
  optimized_ops::AffineQuantize<DataType>(
      op_params, output_dequantize_shape,
      GetTensorData<float>(output_dequantize), output_shape,
      GetTensorData<DataType>(output));

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  switch (output->type) {
    case kTfLiteInt32: {
      // TensorFlow does not support negative for int32.
      TF_LITE_ENSURE_OK(context, CheckValue(context, input2));
      PowImpl<int32_t>(input1, input2, output, data->requires_broadcast);
      break;
    }
    case kTfLiteInt16: {
      EvalQuantize<int16_t>(context, node, input1, input2, output,
                            data->requires_broadcast);
      break;
    }
    case kTfLiteInt8: {
      EvalQuantize<int8_t>(context, node, input1, input2, output,
                           data->requires_broadcast);
      break;
    }
    case kTfLiteFloat32: {
      PowImpl<float>(input1, input2, output, data->requires_broadcast);
      break;
    }
    case kTfLiteFloat16: {
      PowImpl<Eigen::half>(input1, input2, output, data->requires_broadcast);
      break;
    }
    case kTfLiteBFloat16: {
      PowImpl<Eigen::bfloat16>(input1, input2, output,
                               data->requires_broadcast);
      break;
    }
    default: {
      TF_LITE_KERNEL_LOG(context, "Unsupported data type: %d", output->type);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

}  // namespace
}  // namespace pow

TfLiteRegistration* Register_POW() {
  static TfLiteRegistration r = {pow::Init, pow::Free, pow::Prepare, pow::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
