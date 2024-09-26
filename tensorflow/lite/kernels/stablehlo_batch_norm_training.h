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

#include "tensorflow/lite/kernels/internal/optimized/reduce.h"
#include "tensorflow/lite/kernels/internal/reference/reduce.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace stablehlo_batch_norm_training {
namespace reference {

constexpr int kMaxReduceRank = 8;

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

  return kTfLiteOk;
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

template <typename DataType>
TfLiteStatus ComputeSum(TfLiteContext* context, TfLiteNode* node,
                        const TfLiteTensor* operand,
                        const int64_t feature_index, TfLiteTensor* batch_sum) {
  const int operand_rank = operand->dims->size;
  std::vector<int> dimarray;
  for (int i = 0; i < operand_rank; ++i) {
    if (i != feature_index) {
      dimarray.push_back(i);
    }
  }
  int resolved_axis[kMaxReduceRank];
  int temp_index[kMaxReduceRank];
  TF_LITE_ENSURE(context,
                 reference_ops::ReduceGeneric<DataType>(
                     GetTensorData<DataType>(operand), operand->dims->data,
                     operand->dims->size, GetTensorData<DataType>(batch_sum),
                     batch_sum->dims->data, batch_sum->dims->size,
                     dimarray.data(), dimarray.size(), false, temp_index,
                     resolved_axis, static_cast<DataType>(0),
                     [](const DataType current, const DataType in) -> DataType {
                       return in + current;
                     }));
  DataType* batch_sum_buffer = GetTensorData<DataType>(batch_sum);

  return kTfLiteOk;
}

}  // namespace reference
}  // namespace stablehlo_batch_norm_training
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_STABLEHLO_BATCH_NORM_TRAINING_H_
