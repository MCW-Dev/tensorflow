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

#ifndef TENSORFLOW_LITE_KERNELS_STABLEHLO_BATCH_NORM_TRAINING_H_
#define TENSORFLOW_LITE_KERNELS_STABLEHLO_BATCH_NORM_TRAINING_H_

#include <cstdint>
#include <limits>
#include <vector>

#include "tensorflow/lite/kernels/internal/reference/reduce.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace stablehlo_batch_norm_training {
namespace reference {

constexpr int kMaxReduceRank = 8;

int compute_quantized_add(int input_value1, int input_value2, int left_shift,
                          double scale, int zero_point) {
  const double twice_max_input_scale = 2 * scale;
  const double real_input_multiplier = scale / twice_max_input_scale;
  const double real_output_multiplier =
      twice_max_input_scale / ((1 << left_shift) * scale);
  int32_t output_multiplier;
  int output_shift;
  int32_t input_multiplier;
  int input_shift;

  tflite::QuantizeMultiplierSmallerThanOneExp(real_input_multiplier,
                                              &input_multiplier, &input_shift);
  if (real_output_multiplier > 1) {
    tflite::QuantizeMultiplierGreaterThanOne(real_output_multiplier,
                                             &output_multiplier, &output_shift);
  } else {
    tflite::QuantizeMultiplierSmallerThanOneExp(
        real_output_multiplier, &output_multiplier, &output_shift);
  }
  input_value1 = input_value1 - zero_point;
  input_value2 = input_value2 - zero_point;
  const int shifted_input_value1 = input_value1 * (1 << left_shift);
  const int shifted_input_value2 = input_value2 * (1 << left_shift);
  const int scaled_input_value1 =
      MultiplyByQuantizedMultiplierSmallerThanOneExp(
          shifted_input_value1, input_multiplier, input_shift);
  const int scaled_input_value2 =
      MultiplyByQuantizedMultiplierSmallerThanOneExp(
          shifted_input_value2, input_multiplier, input_shift);
  const int raw_addition_value = scaled_input_value1 + scaled_input_value2;
  return MultiplyByQuantizedMultiplierSmallerThanOneExp(
             raw_addition_value, output_multiplier, output_shift) +
         zero_point;
}

int compute_quantized_sub(int input_value1, int input_value2, int left_shift,
                          double scale, int zero_point) {
  const double twice_max_input_scale = 2 * scale;
  const double real_input_multiplier = scale / twice_max_input_scale;
  const double real_output_multiplier =
      twice_max_input_scale / ((1 << left_shift) * scale);
  int32_t output_multiplier;
  int output_shift;
  int32_t input_multiplier;
  int input_shift;

  tflite::QuantizeMultiplierSmallerThanOneExp(real_input_multiplier,
                                              &input_multiplier, &input_shift);
  if (real_output_multiplier > 1) {
    tflite::QuantizeMultiplierGreaterThanOne(real_output_multiplier,
                                             &output_multiplier, &output_shift);
  } else {
    tflite::QuantizeMultiplierSmallerThanOneExp(
        real_output_multiplier, &output_multiplier, &output_shift);
  }
  input_value1 = input_value1 - zero_point;
  input_value2 = input_value2 - zero_point;
  const int shifted_input_value1 = input_value1 * (1 << left_shift);
  const int shifted_input_value2 = input_value2 * (1 << left_shift);
  const int scaled_input_value1 =
      MultiplyByQuantizedMultiplierSmallerThanOneExp(
          shifted_input_value1, input_multiplier, input_shift);
  const int scaled_input_value2 =
      MultiplyByQuantizedMultiplierSmallerThanOneExp(
          shifted_input_value2, input_multiplier, input_shift);
  const int raw_subration_value = scaled_input_value1 - scaled_input_value2;
  return MultiplyByQuantizedMultiplierSmallerThanOneExp(
             raw_subration_value, output_multiplier, output_shift) +
         zero_point;
}

template <typename DataType>
int compute_quantized_div(int input_value1, int input_value2, double scale,
                          int zero_point) {
  int32_t div_multiplier;
  int div_shift;
  const double real_div_multiplier = scale / (scale * scale);
  QuantizeMultiplier(real_div_multiplier, &div_multiplier, &div_shift);
  if (input_value2 < 0) {
    // Invert signs to avoid a negative input_value2 as input2_inv needs to be
    // positive to be used as multiplier of MultiplyByQuantizedMultiplier.
    input_value1 = -input_value1;
    input_value2 = -input_value2;
  }
  int recip_shift;

  const int32_t input2_inv = GetReciprocal(input_value2, 31, &recip_shift);
  const int headroom = CountLeadingSignBits(input_value1);
  const int32_t unscaled_quotient = MultiplyByQuantizedMultiplierGreaterThanOne(
      input_value1, input2_inv, headroom);
  const int total_shift = div_shift - recip_shift - headroom;
  int32_t unclamped_result;
  if (std::abs(total_shift) > 31) {
    unclamped_result =
        zero_point + MultiplyByQuantizedMultiplierGreaterThanOne(
                         unscaled_quotient, div_multiplier, total_shift);
  } else {
    unclamped_result =
        zero_point + MultiplyByQuantizedMultiplierSmallerThanOneExp(
                         unscaled_quotient, div_multiplier, total_shift);
  }
  return std::min(
      static_cast<int>(std::numeric_limits<DataType>::max()),
      std::max(static_cast<int>(std::numeric_limits<DataType>::min()),
               unclamped_result));
}

template <typename DataType>
int compute_quantized_mul(int input_value1, int input_value2,
                          const double scale, int zero_point) {
  int32_t mul_multiplier;
  int mul_shift;
  QuantizeMultiplier(scale, &mul_multiplier, &mul_shift);
  const int32_t input1_val = zero_point + input_value1;
  const int32_t input2_val = zero_point + input_value2;
  const int32_t unclamped_result =
      zero_point + MultiplyByQuantizedMultiplier(input1_val * input2_val,
                                                 mul_multiplier, mul_shift);
  return std::min(
      static_cast<int>(std::numeric_limits<DataType>::max()),
      std::max(static_cast<int>(std::numeric_limits<DataType>::min()),
               unclamped_result));
}

inline void SetRsqrtOutputMultiplier(const float input_scale,
                                     const float output_scale,
                                     int32_t* multiplier, int32_t* shift) {
  const double scale = 1. / (std::sqrt(input_scale) * output_scale);
  QuantizeMultiplier(scale, multiplier, shift);
}

template <typename DataType>
int compute_quantized_rsqrt(int input_value, const int32_t kShift,
                            const double scale, int zero_point) {
  int multiplier, shift;
  SetRsqrtOutputMultiplier(scale, scale, &multiplier, &shift);
  const int kMin = std::numeric_limits<DataType>::min();
  const int kMax = std::numeric_limits<DataType>::max();
  const int32_t value = (input_value - zero_point);
  if (value == 0) {
    // Assume that any value close to 0 represents the max output value.
    return static_cast<DataType>(kMax);
  }
  int32_t inv_sqrt_multiplier;
  int inv_sqrt_shift;
  GetInvSqrtQuantizedMultiplierExp(value, kReverseShift, &inv_sqrt_multiplier,
                                   &inv_sqrt_shift);
  const int32_t data = MultiplyByQuantizedMultiplier(1, inv_sqrt_multiplier,
                                                     inv_sqrt_shift + kShift);
  const int32_t output =
      MultiplyByQuantizedMultiplier(data, multiplier, shift - kShift) +
      zero_point;
  return static_cast<DataType>(std::min(std::max(output, kMin), kMax));
}

template <typename DataType>
TfLiteStatus ComputeQuantizedMean(TfLiteContext* context, TfLiteNode* node,
                                  const TfLiteTensor* operand,
                                  int64_t feature_index,
                                  TfLiteTensor* batch_mean) {
  const int operand_rank = operand->dims->size;
  std::vector<int> dimarray;
  for (int i = 0; i < operand_rank; ++i) {
    if (i != feature_index) {
      dimarray.push_back(i);
    }
  }
  int resolved_axis[kMaxReduceRank] = {0};
  int temp_index[kMaxReduceRank] = {0};
  std::vector<int> temp_sum(dimarray.size(), 0);
  int32_t multiplier;
  int shift;
  QuantizeMultiplier(1.0, &multiplier, &shift);
  TF_LITE_ENSURE(
      context, reference_ops::QuantizedMeanOrSum(
                   GetTensorData<DataType>(operand), operand->params.zero_point,
                   operand->dims->data, operand->dims->size,
                   GetTensorData<DataType>(batch_mean), multiplier, shift,
                   operand->params.zero_point, batch_mean->dims->data,
                   batch_mean->dims->size, dimarray.data(), dimarray.size(),
                   false, temp_index, resolved_axis, temp_sum.data(), false));
  return kTfLiteOk;
}

template <typename DataType>
TfLiteStatus ComputeMean(TfLiteContext* context, TfLiteNode* node,
                         const TfLiteTensor* operand, int64_t feature_index,
                         TfLiteTensor* batch_mean) {
  const int operand_rank = operand->dims->size;
  std::vector<int> dimarray;
  for (int i = 0; i < operand_rank; ++i) {
    if (i != feature_index) {
      dimarray.push_back(i);
    }
  }
  int resolved_axis[kMaxReduceRank];
  int temp_index[kMaxReduceRank];
  int temp_sum[NumElements(batch_mean)];
  TF_LITE_ENSURE(context,
                 reference_ops::ReduceGeneric<DataType>(
                     GetTensorData<DataType>(operand), operand->dims->data,
                     operand->dims->size, GetTensorData<DataType>(batch_mean),
                     batch_mean->dims->data, batch_mean->dims->size,
                     dimarray.data(), dimarray.size(), false, temp_index,
                     resolved_axis, static_cast<DataType>(0),
                     [](const DataType current, const DataType in) -> DataType {
                       return in + current;
                     }));
  int64_t operand_size = 1;
  for (int i = 0; i < operand->dims->size; ++i) {
    operand_size *= operand->dims->data[i];
  }
  int64_t feature_dim = operand->dims->data[feature_index];
  int64_t divisor = operand_size / feature_dim;

  DataType* mean_data = GetTensorData<DataType>(batch_mean);
  for (int64_t i = 0; i < NumElements(batch_mean); ++i) {
    mean_data[i] = mean_data[i] / divisor;
  }
  // }
  return kTfLiteOk;
}

template <typename DataType>
TfLiteStatus ComputeQuantizedVariance(TfLiteContext* context, TfLiteNode* node,
                                      const TfLiteTensor* operand,
                                      int64_t feature_index,
                                      TfLiteTensor* batch_mean,
                                      TfLiteTensor* batch_var,
                                      TfLiteTensor* centered_operand) {
  TF_LITE_ENSURE_STATUS(ComputeQuantizedMean<DataType>(
      context, node, operand, feature_index, batch_mean));

  DataType* mean_data = GetTensorData<DataType>(batch_mean);
  const int operand_rank = operand->dims->size;
  std::vector<int> broadcast_shape(operand_rank, 1);
  broadcast_shape[feature_index] = operand->dims->data[feature_index];

  const DataType* operand_data = GetTensorData<DataType>(operand);
  DataType* centered_operand_data = GetTensorData<DataType>(centered_operand);
  const int left_shift = (operand->type == kTfLiteInt16) ? 15 : 20;
  for (int64_t i = 0; i < NumElements(operand); ++i) {
    const int raw_centered_output = compute_quantized_sub(
        operand_data[i], mean_data[i % NumElements(batch_mean)], left_shift,
        operand->params.scale, operand->params.zero_point);
    centered_operand_data[i] = compute_quantized_mul<DataType>(
        raw_centered_output, raw_centered_output, operand->params.scale,
        operand->params.zero_point);
  }
  return ComputeQuantizedMean<DataType>(context, node, centered_operand,
                                        feature_index, batch_var);
}

template <typename DataType>
TfLiteStatus ComputeVariance(TfLiteContext* context, TfLiteNode* node,
                             const TfLiteTensor* operand, int64_t feature_index,
                             TfLiteTensor* batch_mean, TfLiteTensor* batch_var,
                             TfLiteTensor* centered_operand) {
  TF_LITE_ENSURE_STATUS(
      ComputeMean<DataType>(context, node, operand, feature_index, batch_mean));

  DataType* mean_data = GetTensorData<DataType>(batch_mean);
  const int operand_rank = operand->dims->size;
  std::vector<int> broadcast_shape(operand_rank, 1);
  broadcast_shape[feature_index] = operand->dims->data[feature_index];

  const DataType* operand_data = GetTensorData<DataType>(operand);
  DataType* centered_operand_data = GetTensorData<DataType>(centered_operand);
  for (int64_t i = 0; i < NumElements(operand); ++i) {
    centered_operand_data[i] =
        operand_data[i] - mean_data[i % broadcast_shape[feature_index]];
    centered_operand_data[i] *= centered_operand_data[i];
  }
  return ComputeMean<DataType>(context, node, centered_operand, feature_index,
                               batch_var);
}

}  // namespace reference
}  // namespace stablehlo_batch_norm_training
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_STABLEHLO_BATCH_NORM_TRAINING_H_
