/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <cmath>
#include <limits>

#include "Eigen/Core"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_reduce_precision {
namespace{
constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

template <typename DataType>
struct FloatingPointTraits {
  static constexpr int mantissa_bits =
      std::numeric_limits<DataType>::digits - 1;
  static constexpr int max_exponent =
      std::numeric_limits<DataType>::max_exponent;
  static constexpr int min_exponent =
      std::numeric_limits<DataType>::min_exponent;
  static constexpr int exponent_bits =
      (__builtin_clz(static_cast<unsigned int>(max_exponent - min_exponent + 1)) ^
       31) +
      1;
};
template <typename DataType>
DataType ReducePrecision(DataType value, int32_t exponent_bits,
                         int32_t mantissa_bits);

template <typename DataType>
TfLiteStatus ReducePrecisionOp(const TfLiteTensor* operand,
                               int32_t exponent_bits, int32_t mantissa_bits,
                               TfLiteTensor* result) {
  for (int i = 0; i < NumElements(result); ++i) {
    DataType value = GetTensorData<DataType>(operand)[i];
    DataType reduced_value = ReducePrecision(value, exponent_bits, mantissa_bits);
    GetTensorData<DataType>(result)[i] = reduced_value;
  }
  return TfLiteStatus::kTfLiteOk;
}

template <typename DataType>
DataType ReducePrecision(DataType value, int32_t exponent_bits,
                         int32_t mantissa_bits) {
  uint32_t int_val = *reinterpret_cast<uint32_t*>(&value);

  const int32_t src_mantissa_bits = FloatingPointTraits<DataType>::mantissa_bits;
  const int32_t src_exponent_bits = FloatingPointTraits<DataType>::exponent_bits;

  if (mantissa_bits < src_mantissa_bits) {
    uint32_t last_mantissa_bit_mask = 1ull << (src_mantissa_bits - mantissa_bits);
    uint32_t base_rounding_bias = (last_mantissa_bit_mask >> 1) - 1;
    uint32_t x_last_mantissa_bit =
        (int_val & last_mantissa_bit_mask) >> (src_mantissa_bits - mantissa_bits);
    uint32_t xRoundingBias = x_last_mantissa_bit + base_rounding_bias;

    uint32_t truncation_mask = ~(last_mantissa_bit_mask - 1);
    int_val = int_val + xRoundingBias;
    int_val = int_val & truncation_mask;
  }

  if (exponent_bits < src_exponent_bits) {
    uint32_t sign_bit_mask = 1ull << 31;
    uint32_t exp_bits_mask = ((1ull << src_exponent_bits) - 1) << src_mantissa_bits;
    uint32_t exponent_bias = (1ull << (src_exponent_bits - 1)) - 1;
    uint32_t reduced_exponent_bias = (1ull << (exponent_bits - 1)) - 1;
    uint32_t reduced_max_exponent = exponent_bias + reduced_exponent_bias;
    uint32_t reduced_min_exponent = exponent_bias - reduced_exponent_bias;

    uint32_t x_exponent = int_val & exp_bits_mask;
    bool x_overflows = x_exponent > (reduced_max_exponent << src_mantissa_bits);
    bool x_underflows = x_exponent <= (reduced_min_exponent << src_mantissa_bits);

    uint32_t x_signed_zero = int_val & sign_bit_mask;
    uint32_t x_signed_inf = x_signed_zero | exp_bits_mask;

    int_val = x_overflows ? x_signed_inf : int_val;
    int_val = x_underflows ? x_signed_zero : int_val;
  }
  DataType reduced_result;
  std::memcpy(&reduced_result, &int_val, sizeof(DataType));
  if (std::isnan(value)) {
    reduced_result =
        mantissa_bits > 0 ? value : std::numeric_limits<DataType>::infinity();
  }
  return reduced_result;
}
} //namespace
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  int input_rank = input->dims->size;
  RuntimeShape input_shape = GetTensorShape(input);
  const TfLiteStablehloReducePrecisionParams* data =
      reinterpret_cast<TfLiteStablehloReducePrecisionParams*>(
          node->builtin_data);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  int result_rank = output->dims->size;
  RuntimeShape result_runtime_shape(result_rank, output->dims->data);
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
  const TfLiteStablehloReducePrecisionParams* data =
      reinterpret_cast<TfLiteStablehloReducePrecisionParams*>(
          node->builtin_data);
  int32_t exponent_bits = data->exponent_bits;
  int32_t mantissa_bits = data->mantissa_bits;
  if (data_type == kTfLiteFloat32) {
    return ReducePrecisionOp<float>(input, exponent_bits, mantissa_bits, output);
  } else if (data_type == kTfLiteFloat16) {
    return ReducePrecisionOp<Eigen::half>(input, exponent_bits, mantissa_bits,
                                          output);
  } else if (data_type == kTfLiteBFloat16) {
    return ReducePrecisionOp<tensorflow::bfloat16>(input, exponent_bits,
                                                   mantissa_bits, output);
  } else {
    TF_LITE_KERNEL_LOG(context, "(Index Type: %s) currently not supported.\n",
                       TfLiteTypeGetName(data_type));
    return TfLiteStatus::kTfLiteError;
  }
}

}  // namespace stablehlo_reduce_precision

TfLiteRegistration* Register_STABLEHLO_REDUCE_PRECISION() {
  static TfLiteRegistration r = {nullptr, nullptr,
                                 stablehlo_reduce_precision::Prepare,
                                 stablehlo_reduce_precision::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
