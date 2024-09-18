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

#include "Eigen/Core"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/dequantize.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_round_nearest_afz {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

template <typename T>
TfLiteStatus EvalImpl(const TfLiteTensor* input, TfLiteTensor* output) {
  const int num_elements = NumElements(output);
  const T* input_data = GetTensorData<T>(input);
  T* output_data = GetTensorData<T>(output);
  for (int i = 0; i < num_elements; ++i) {
    output_data[i] = T(std::round(input_data[i]));
  }
  return kTfLiteOk;
}

template <typename T>
TfLiteStatus EvalRoundNearestAFZQuantized(TfLiteContext* context,
                                          TfLiteNode* node,
                                          const TfLiteTensor* input,
                                          TfLiteTensor* output) {
  const double scale = input->params.scale;
  const int32_t zero_point = input->params.zero_point;
  const int num_elements = NumElements(output);
  const T* input_buffer = GetTensorData<T>(input);
  T* output_buffer = GetTensorData<T>(output);
  for (int i = 0; i < num_elements; ++i) {
    float dequantized_value = (input_buffer[i] - zero_point) * scale;
    float rounded_value = std::round(dequantized_value);
    int32_t quantized_value =
        static_cast<int32_t>(std::round(rounded_value / scale)) + zero_point;
    quantized_value =
        std::min(std::max(quantized_value,
                          static_cast<int32_t>(std::numeric_limits<T>::min())),
                 static_cast<int32_t>(std::numeric_limits<T>::max()));
    output_buffer[i] = static_cast<T>(quantized_value);
  }
  return TfLiteStatus::kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_EQ(context, HaveSameShapes(input, output), 1);
  TF_LITE_ENSURE(context, input->type == kTfLiteFloat32 ||
                              input->type == kTfLiteBFloat16 ||
                              input->type == kTfLiteFloat16 ||
                              input->type == kTfLiteInt16 ||
                              input->type == kTfLiteInt8);

  bool is_quantized = input->quantization.type != kTfLiteNoQuantization;

  if (input->type == kTfLiteInt8 ||
      (input->type == kTfLiteInt16 && is_quantized)) {
    const auto* input_params =
        reinterpret_cast<TfLiteAffineQuantization*>(input->quantization.params);
    const auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
        output->quantization.params);

    TF_LITE_ENSURE(context, input_params != nullptr);
    TF_LITE_ENSURE(context, input_params->scale != nullptr);
    TF_LITE_ENSURE(context, input_params->scale->size > 0);
    TF_LITE_ENSURE(context, input_params->zero_point->size > 0);

    TF_LITE_ENSURE(context, output_params != nullptr);
    TF_LITE_ENSURE(context, output_params->scale != nullptr);
    TF_LITE_ENSURE(context, output_params->scale->size > 0);
    TF_LITE_ENSURE(context, output_params->zero_point->size > 0);

    if (input->type == kTfLiteInt16) {
      TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
    }
  }

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TfLiteType data_type = input->type;
  switch (data_type) {
    case kTfLiteFloat32:
      return EvalImpl<float>(input, output);
    case kTfLiteFloat16:
      return EvalImpl<Eigen::half>(input, output);
    case kTfLiteBFloat16:
      return EvalImpl<Eigen::bfloat16>(input, output);
    case kTfLiteInt8:
      return EvalRoundNearestAFZQuantized<int8_t>(context, node, input, output);
    case kTfLiteInt16: {
      if (output->quantization.type == kTfLiteNoQuantization)
        return EvalImpl<int16_t>(input, output);
      else
        return EvalRoundNearestAFZQuantized<int16_t>(context, node, input,
                                                     output);
    }
    default: {
      TF_LITE_KERNEL_LOG(
          context, "Type %d is currently not supported by this op.", data_type);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

}  // namespace stablehlo_round_nearest_afz

TfLiteRegistration* Register_STABLEHLO_ROUND_NEAREST_AFZ() {
  static TfLiteRegistration r = {nullptr, nullptr,
                                 stablehlo_round_nearest_afz::Prepare,
                                 stablehlo_round_nearest_afz::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
