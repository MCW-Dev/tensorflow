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
#include <cmath>
#include <limits>
#include <type_traits>

#include "Eigen/Core"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/core/c/common.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_reduce_precision {
namespace {
constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

template <typename DataType>
struct FloatingPointTraits {
  static constexpr int kMantissaBits =
      std::numeric_limits<DataType>::digits - 1;
  static constexpr int kMaxExponent =
      std::numeric_limits<DataType>::max_exponent;
  static constexpr int kMinExponent =
      std::numeric_limits<DataType>::min_exponent;
  static constexpr int kExponentBits = (__builtin_clz(static_cast<unsigned int>(
                                            kMaxExponent - kMinExponent + 1)) ^
                                        31) +
                                       1;
  static_assert((std::is_same_v<DataType, float> && sizeof(DataType) == 4) ||
                    sizeof(DataType) == 2,
                "DataType must be float with sizeof 4 or uint16_t");
  using UIntType =
      std::conditional_t<std::is_same_v<DataType, float>, uint32_t, uint16_t>;
};

template <typename UIntType>
UIntType ReducePrecision(UIntType int_val, int32_t exponent_bits,
                         int32_t mantissa_bits, int32_t src_mantissa_bits,
                         int32_t src_exponent_bits,
                         UIntType last_mantissa_bit_mask,
                         UIntType base_rounding_bias, UIntType truncation_mask,
                         UIntType sign_bit_mask, UIntType exp_bits_mask,
                         UIntType reduced_max_exponent,
                         UIntType reduced_min_exponent);

template <typename DataType>
TfLiteStatus ReducePrecisionOp(const TfLiteTensor* operand,
                               int32_t exponent_bits, int32_t mantissa_bits,
                               TfLiteTensor* result) {
  const int num_elements = NumElements(result);
  const DataType* input = GetTensorData<DataType>(operand);
  DataType* output = GetTensorData<DataType>(result);
  using UIntType = typename FloatingPointTraits<DataType>::UIntType;
  const int32_t src_mantissa_bits =
      FloatingPointTraits<DataType>::kMantissaBits;
  const int32_t src_exponent_bits =
      FloatingPointTraits<DataType>::kExponentBits;
  UIntType last_mantissa_bit_mask = static_cast<UIntType>(1) << std::max(
                                        0, (src_mantissa_bits - mantissa_bits));
  UIntType base_rounding_bias = (last_mantissa_bit_mask >> 1) - 1;
  UIntType truncation_mask = ~(last_mantissa_bit_mask - 1);
  UIntType sign_bit_mask = static_cast<UIntType>(1)
                           << (sizeof(UIntType) * 8 - 1);
  UIntType exp_bits_mask = ((static_cast<UIntType>(1) << src_exponent_bits) - 1)
                           << src_mantissa_bits;
  UIntType exponent_bias =
      (static_cast<UIntType>(1) << (src_exponent_bits - 1)) - 1;
  UIntType reduced_exponent_bias =
      (static_cast<UIntType>(1) << (exponent_bits - 1)) - 1;
  UIntType reduced_max_exponent = exponent_bias + reduced_exponent_bias;
  UIntType reduced_min_exponent = exponent_bias - reduced_exponent_bias;
  for (int i = 0; i < num_elements; ++i) {
    if (std::isnan(input[i])) {
      output[i] = mantissa_bits > 0 ? input[i]
                                    : std::numeric_limits<DataType>::infinity();
      continue;
    }
    UIntType int_val;
    std::memcpy(&int_val, &input[i], sizeof(UIntType));
    int_val = ReducePrecision(int_val, exponent_bits, mantissa_bits,
                              src_mantissa_bits, src_exponent_bits,
                              last_mantissa_bit_mask, base_rounding_bias,
                              truncation_mask, sign_bit_mask, exp_bits_mask,
                              reduced_max_exponent, reduced_min_exponent);
    std::memcpy(&output[i], &int_val, sizeof(DataType));
  }
  return TfLiteStatus::kTfLiteOk;
}

template <typename UIntType>
UIntType ReducePrecision(UIntType int_val, int32_t exponent_bits,
                         int32_t mantissa_bits, int32_t src_mantissa_bits,
                         int32_t src_exponent_bits,
                         UIntType last_mantissa_bit_mask,
                         UIntType base_rounding_bias, UIntType truncation_mask,
                         UIntType sign_bit_mask, UIntType exp_bits_mask,
                         UIntType reduced_max_exponent,
                         UIntType reduced_min_exponent) {
  if (mantissa_bits < src_mantissa_bits) {
    UIntType last_mantissa_bit = (int_val & last_mantissa_bit_mask) >>
                                 (src_mantissa_bits - mantissa_bits);
    UIntType rounding_bias = last_mantissa_bit + base_rounding_bias;
    int_val = int_val + rounding_bias;
    int_val = int_val & truncation_mask;
  }
  if (exponent_bits < src_exponent_bits) {
    UIntType x_exponent = int_val & exp_bits_mask;
    bool x_overflows = x_exponent > (reduced_max_exponent << src_mantissa_bits);
    bool x_underflows =
        x_exponent <= (reduced_min_exponent << src_mantissa_bits);
    UIntType x_signed_zero = int_val & sign_bit_mask;
    UIntType x_signed_inf = x_signed_zero | exp_bits_mask;
    int_val = x_overflows ? x_signed_inf : int_val;
    int_val = x_underflows ? x_signed_zero : int_val;
  }
  return int_val;
}

}  // namespace

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_EQ(context,HaveSameShapes(input,output),1);
  const TfLiteStablehloReducePrecisionParams* data =
      reinterpret_cast<TfLiteStablehloReducePrecisionParams*>(
          node->builtin_data);
  TF_LITE_ENSURE_MSG(context, data->exponent_bits >= 1,
                     "stablehlo.reduce_precision: 'Exponent_bits' must be "
                     "greater than or equal to 1.");
  TF_LITE_ENSURE_MSG(context, data->mantissa_bits >= 0,
                     "stablehlo.reduce_precision: 'Mantissa_bits' must be "
                     "greater than or equal to 0.");
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
    return ReducePrecisionOp<float>(input, exponent_bits, mantissa_bits,
                                    output);
  } else if (data_type == kTfLiteFloat16) {
    return ReducePrecisionOp<Eigen::half>(input, exponent_bits, mantissa_bits,
                                          output);
  } else if (data_type == kTfLiteBFloat16) {
    return ReducePrecisionOp<Eigen::bfloat16>(input, exponent_bits,
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
