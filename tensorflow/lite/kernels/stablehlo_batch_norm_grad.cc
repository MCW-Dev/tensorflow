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
#include <cstdint>

#include "Eigen/Core"
#include "tensorflow/lite/kernels/dequantize.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reduce.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/stablehlo_batch_norm_training.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_batch_norm_grad {
namespace {

constexpr int kMaxTempTensorsNonQuantized = 10;
constexpr int32_t kMaxReduceRank = 8;
struct OpData {
 public:
  enum {
    kOperandTensor,
    kScaleTensor,
    kMeanTensor,
    kVarianceTensor,
    kGradOutputTensor
  };
  enum {
    kOutputGradOperandTensor,
    kOutputGradScaleTensor,
    kOutputGradOffsetTensor
  };
  int scratch_tensor_index;
};

void* Init(TfLiteContext* context, const char* options, size_t options_len) {
  OpData* data = new OpData;
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete static_cast<OpData*>(buffer);
}

TfLiteStatus PrepareTemporaries(TfLiteContext* context, TfLiteNode* node,
                                const TfLiteBatchNormGradParams* params,
                                const TfLiteTensor* operand,
                                const TfLiteTensor* grad_output,
                                const TfLiteTensor* scale) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  context->AddTensors(context, kMaxTempTensorsNonQuantized,
                      &data->scratch_tensor_index);
  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(kMaxTempTensorsNonQuantized);

  node->temporaries->data[0] = data->scratch_tensor_index;
  TfLiteTensor* epsilon_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, 0, &epsilon_tensor));
  TfLiteIntArray* epsilon_tensor_shape = TfLiteIntArrayCreate(1);
  epsilon_tensor_shape->data[0] = 1;
  epsilon_tensor->type = operand->type;
  epsilon_tensor->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, epsilon_tensor,
                                                   epsilon_tensor_shape));

  node->temporaries->data[1] = data->scratch_tensor_index + 1;
  TfLiteTensor* centered_operand;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, 1, &centered_operand));
  TfLiteIntArray* centered_operand_bcast_shape =
      TfLiteIntArrayCreate(operand->dims->size);
  for (int i = 0; i < operand->dims->size; ++i) {
    centered_operand_bcast_shape->data[i] = operand->dims->data[i];
  }
  centered_operand->type = operand->type;
  centered_operand->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, centered_operand,
                                          centered_operand_bcast_shape));

  node->temporaries->data[2] = data->scratch_tensor_index + 2;
  TfLiteTensor* stddev;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, 2, &stddev));
  TfLiteIntArray* stddev_bcast_shape =
      TfLiteIntArrayCreate(operand->dims->size);
  for (int i = 0; i < operand->dims->size; ++i) {
    stddev_bcast_shape->data[i] = operand->dims->data[i];
  }
  stddev->type = operand->type;
  stddev->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, stddev, stddev_bcast_shape));

  node->temporaries->data[3] = data->scratch_tensor_index + 3;
  TfLiteTensor* normalized_operand;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, 3, &normalized_operand));
  TfLiteIntArray* normalized_operand_bcast_shape =
      TfLiteIntArrayCreate(operand->dims->size);
  for (int i = 0; i < operand->dims->size; ++i) {
    normalized_operand_bcast_shape->data[i] = operand->dims->data[i];
  }
  normalized_operand->type = operand->type;
  normalized_operand->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, normalized_operand,
                                          normalized_operand_bcast_shape));

  node->temporaries->data[4] = data->scratch_tensor_index + 4;
  TfLiteTensor* elements_per_feature_tensor;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, 4,
                                              &elements_per_feature_tensor));
  TfLiteIntArray* elements_per_feature_tensor_shape = TfLiteIntArrayCreate(0);

  elements_per_feature_tensor->type = operand->type;
  elements_per_feature_tensor->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, elements_per_feature_tensor,
                                          elements_per_feature_tensor_shape));

  node->temporaries->data[5] = data->scratch_tensor_index + 5;
  TfLiteTensor* i6;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, 5, &i6));
  TfLiteIntArray* i6_bcast_shape = TfLiteIntArrayCreate(operand->dims->size);
  for (int i = 0; i < operand->dims->size; ++i) {
    i6_bcast_shape->data[i] = operand->dims->data[i];
  }
  i6->type = operand->type;
  i6->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, i6, i6_bcast_shape));

  node->temporaries->data[6] = data->scratch_tensor_index + 6;
  TfLiteTensor* grad_output_centered_operand_mul;
  TF_LITE_ENSURE_OK(
      context,
      GetTemporarySafe(context, node, 6, &grad_output_centered_operand_mul));
  TfLiteIntArray* grad_output_centered_operand_mul_bcast_shape =
      TfLiteIntArrayCreate(operand->dims->size);
  for (int i = 0; i < operand->dims->size; ++i) {
    grad_output_centered_operand_mul_bcast_shape->data[i] =
        operand->dims->data[i];
  }
  grad_output_centered_operand_mul->type = operand->type;
  grad_output_centered_operand_mul->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(
                                 context, grad_output_centered_operand_mul,
                                 grad_output_centered_operand_mul_bcast_shape));

  node->temporaries->data[7] = data->scratch_tensor_index + 7;
  TfLiteTensor* grad_output_reduced;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, 7, &grad_output_reduced));
  TfLiteIntArray* grad_output_reduced_shape = TfLiteIntArrayCreate(1);
  grad_output_reduced_shape->data[0] =
      grad_output->dims->data[params->feature_index];

  grad_output_reduced->type = operand->type;
  grad_output_reduced->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, grad_output_reduced,
                                                   grad_output_reduced_shape));

  node->temporaries->data[8] = data->scratch_tensor_index + 8;
  TfLiteTensor* grad_scale_intermediate;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, 8, &grad_scale_intermediate));
  TfLiteIntArray* grad_scale_intermediate_shape =
      TfLiteIntArrayCreate(operand->dims->size);
  for (int i = 0; i < operand->dims->size; ++i) {
    grad_scale_intermediate_shape->data[i] = operand->dims->data[i];
  }

  grad_scale_intermediate->type = operand->type;
  grad_scale_intermediate->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, grad_scale_intermediate,
                                          grad_scale_intermediate_shape));

   TfLiteIntArray* i3_shape = TfLiteIntArrayCreate(scale->dims->size);
    for (int i = 0; i < scale->dims->size; ++i) {
      i3_shape->data[i] = scale->dims->data[i];
    }
    node->temporaries->data[9] = data->scratch_tensor_index + 9;
    TfLiteTensor* i3;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, 9, &i3));
    i3->type = kTfLiteInt8;
    i3->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, i3, i3_shape));

  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 5);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 3);

  const TfLiteTensor* operand;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, OpData::kOperandTensor, &operand));

  const TfLiteTensor* scale;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, OpData::kScaleTensor, &scale));

  const TfLiteTensor* mean;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, OpData::kMeanTensor, &mean));

  const TfLiteTensor* variance;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, OpData::kVarianceTensor, &variance));

  const TfLiteTensor* grad_output;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, OpData::kGradOutputTensor, &grad_output));

  TfLiteTensor* grad_operand;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, OpData::kOutputGradOperandTensor,
                             &grad_operand));

  TfLiteTensor* grad_scale;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, OpData::kOutputGradScaleTensor,
                                  &grad_scale));

  TfLiteTensor* grad_offset;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, OpData::kOutputGradOffsetTensor,
                             &grad_offset));

  const TfLiteBatchNormGradParams* params =
      reinterpret_cast<TfLiteBatchNormGradParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  int operand_rank = NumDimensions(operand);
  TF_LITE_ENSURE(context, params->feature_index >= 0 &&
                              params->feature_index < operand_rank);

  TF_LITE_ENSURE_TYPES_EQ(context, operand->type, scale->type);
  TF_LITE_ENSURE_TYPES_EQ(context, operand->type, mean->type);
  TF_LITE_ENSURE_TYPES_EQ(context, operand->type, variance->type);
  TF_LITE_ENSURE_TYPES_EQ(context, operand->type, grad_output->type);
  TF_LITE_ENSURE_TYPES_EQ(context, operand->type, grad_operand->type);
  TF_LITE_ENSURE_TYPES_EQ(context, scale->type, grad_scale->type);
  TF_LITE_ENSURE_TYPES_EQ(context, scale->type, grad_offset->type);

  TF_LITE_ENSURE_EQ(
      context, TfLiteIntArrayEqual(operand->dims, grad_output->dims), true);

  TF_LITE_ENSURE_EQ(context, scale->dims->data[0],
                    operand->dims->data[params->feature_index]);

  TfLiteIntArray* grad_operand_size = TfLiteIntArrayCopy(operand->dims);
  TfLiteIntArray* grad_scale_size = TfLiteIntArrayCreate(1);
  grad_scale_size->data[0] = operand->dims->data[params->feature_index];
  TfLiteIntArray* grad_offset_size = TfLiteIntArrayCreate(1);
  grad_offset_size->data[0] = operand->dims->data[params->feature_index];

  TF_LITE_ENSURE_OK(context, PrepareTemporaries(context, node, params, operand,
                                                grad_output, scale));
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(context, grad_operand, grad_operand_size));
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(context, grad_scale, grad_scale_size));
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(context, grad_offset, grad_offset_size));
  TF_LITE_ENSURE_EQ(
      context, TfLiteIntArrayEqual(operand->dims, grad_operand->dims), true);

  TF_LITE_ENSURE_EQ(context, TfLiteIntArrayEqual(scale->dims, mean->dims),
                    true);
  TF_LITE_ENSURE_EQ(context, TfLiteIntArrayEqual(scale->dims, variance->dims),
                    true);
  TF_LITE_ENSURE_EQ(context, TfLiteIntArrayEqual(scale->dims, grad_scale->dims),
                    true);
  TF_LITE_ENSURE_EQ(context,
                    TfLiteIntArrayEqual(scale->dims, grad_offset->dims), true);

  return kTfLiteOk;
}

template <typename DataType>
TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node,
                      const TfLiteTensor* operand, const TfLiteTensor* scale,
                      const TfLiteTensor* mean, const TfLiteTensor* variance,
                      const TfLiteTensor* grad_output,
                      TfLiteTensor* grad_operand, TfLiteTensor* grad_scale,
                      TfLiteTensor* grad_offset) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* epsilon_tensor = GetTemporary(context, node, 0);
  TfLiteTensor* centered_operand = GetTemporary(context, node, 1);
  TfLiteTensor* stddev = GetTemporary(context, node, 2);
  TfLiteTensor* normalized_operand = GetTemporary(context, node, 3);
  TfLiteTensor* elements_per_feature_tensor = GetTemporary(context, node, 4);
  TfLiteTensor* i6 = GetTemporary(context, node, 5);
  TfLiteTensor* grad_output_centered_operand_mul =
      GetTemporary(context, node, 6);
  TfLiteTensor* grad_output_reduced = GetTemporary(context, node, 7);
  TfLiteTensor* grad_scale_intermediate = GetTemporary(context, node, 8);
  TfLiteTensor* i3 = GetTemporary(context, node, 9);

  const TfLiteBatchNormGradParams* params =
      reinterpret_cast<TfLiteBatchNormGradParams*>(node->builtin_data);

  const int64_t feature_index = params->feature_index;
  const float epsilon = params->epsilon;

  TfLiteIntArray* feature_dims = TfLiteIntArrayCreate(1);
  feature_dims->data[0] = feature_index;

  const DataType* scale_data = GetTensorData<DataType>(scale);
  int scale_size = NumElements(scale);

  epsilon_tensor->data.f[0] = epsilon;

  ArithmeticParams op_params;
  op_params.broadcast_category = BroadcastableOpCategory::kGenericBroadcast;

  DataType* centered_operand_buffer = GetTensorData<DataType>(centered_operand);
  const float* operand_buffer = GetTensorData<float>(operand);
  const DataType* mean_data = GetTensorData<DataType>(mean);
  for (int i = 0; i < NumElements(centered_operand); ++i) {
    centered_operand_buffer[i] = static_cast<DataType>(
        operand_buffer[i] - mean_data[i % NumElements(mean)]);
  }

  int num_elements = NumElements(stddev);
  const DataType* variance_data = GetTensorData<DataType>(variance);
  int variance_size = NumElements(variance);
  DataType* stddev_buffer = GetTensorData<DataType>(stddev);
  for (int i = 0; i < NumElements(stddev); ++i) {
    stddev_buffer[i] = static_cast<DataType>(
        std::sqrt(variance_data[i % (NumElements(variance))] +
                  static_cast<DataType>(epsilon)));
  }

  DataType* normalized_buffer = GetTensorData<DataType>(normalized_operand);
  for (int i = 0; i < NumElements(normalized_operand); ++i) {
    normalized_buffer[i] = centered_operand_buffer[i] / stddev_buffer[i];
  }

  int operand_size = NumElements(operand);
  int feature_size = GetTensorShape(operand).Dims(feature_index);
  float elements_per_feature = static_cast<float>(operand_size) / feature_size;
  elements_per_feature_tensor->data.f[0] = elements_per_feature;

  TfLiteIntArray* a = TfLiteIntArrayCreate(0);

  DataType* element_per_feature_tensor_buffer =
      GetTensorData<DataType>(elements_per_feature_tensor);
  const DataType* grad_output_buffer = GetTensorData<DataType>(grad_output);
  tflite::stablehlo_batch_norm_training::reference::ComputeSum<DataType>(
      context, node, grad_output, feature_index, grad_output_reduced);

  DataType* grad_output_centered_operand_mul_buffer =
      GetTensorData<DataType>(grad_output_centered_operand_mul);
  for (int i = 0; i < NumElements(grad_output_centered_operand_mul); ++i) {
    grad_output_centered_operand_mul_buffer[i] =
        grad_output_buffer[i] * centered_operand_buffer[i];
  }

  tflite::stablehlo_batch_norm_training::reference::ComputeSum<DataType>(
      context, node, grad_output_centered_operand_mul, feature_index, i3);
  DataType* i3_buffer = GetTensorData<DataType>(i3);
  DataType* i6_buffer = GetTensorData<DataType>(i6);
  DataType* grad_output_reduced_buffer =
      GetTensorData<DataType>(grad_output_reduced);
  for (int i = 0; i < NumElements(i6); ++i) {
    i6_buffer[i] =
        ((grad_output_buffer[i] *
              element_per_feature_tensor_buffer
                  [i % (NumElements(elements_per_feature_tensor))] -
          grad_output_reduced_buffer[i % NumElements(grad_output_reduced)]) -
         (i3_buffer[i % (NumElements(i3))] * centered_operand_buffer[i]) /
             (variance_data[i % (NumElements(variance))] +
              static_cast<DataType>(epsilon)));
  }

  DataType* grad_operand_buffer = GetTensorData<DataType>(grad_operand);

  for (int i = 0; i < NumElements(grad_operand); ++i) {
    grad_operand_buffer[i] =
        ((scale_data[i % scale_size] / stddev_buffer[i]) /
         element_per_feature_tensor_buffer
             [i % (NumElements(elements_per_feature_tensor))]) *
        i6_buffer[i];
  }

  DataType* grad_scale_intermediate_buffer =
      GetTensorData<DataType>(grad_scale_intermediate);
  for (int i = 0; i < NumElements(grad_scale_intermediate); ++i) {
    grad_scale_intermediate_buffer[i] =
        static_cast<DataType>(grad_output_buffer[i] * normalized_buffer[i]);
  }

  tflite::stablehlo_batch_norm_training::reference::ComputeSum<DataType>(
      context, node, grad_scale_intermediate, feature_index, grad_scale);

  tflite::stablehlo_batch_norm_training::reference::ComputeSum<DataType>(
      context, node, grad_output, feature_index, grad_offset);
      
  TfLiteIntArrayFree(feature_dims);
  TfLiteIntArrayFree(a);
  return kTfLiteOk;
}
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* operand;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, OpData::kOperandTensor, &operand));
  const TfLiteTensor* scale;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, OpData::kScaleTensor, &scale));
  const TfLiteTensor* mean;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, OpData::kMeanTensor, &mean));
  const TfLiteTensor* variance;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, OpData::kVarianceTensor, &variance));
  const TfLiteTensor* grad_output;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, OpData::kGradOutputTensor, &grad_output));
  TfLiteTensor* grad_operand;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, OpData::kOutputGradOperandTensor,
                             &grad_operand));
  TfLiteTensor* grad_scale;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, OpData::kOutputGradScaleTensor,
                                  &grad_scale));
  TfLiteTensor* grad_offset;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, OpData::kOutputGradOffsetTensor,
                             &grad_offset));
  switch (operand->type) {
    case kTfLiteFloat32: {
      return EvalImpl<float>(context, node, operand, scale, mean, variance,
                             grad_output, grad_operand, grad_scale,
                             grad_offset);
    }
    case kTfLiteFloat16: {
      return EvalImpl<Eigen::half>(context, node, operand, scale, mean,
                                   variance, grad_output, grad_operand,
                                   grad_scale, grad_offset);
    }
    case kTfLiteBFloat16: {
      return EvalImpl<Eigen::bfloat16>(context, node, operand, scale, mean,
                                       variance, grad_output, grad_operand,
                                       grad_scale, grad_offset);
    }
    default: {
      TF_LITE_KERNEL_LOG(
          context, "Type '%s' is not supported by stablehlo.batch_norm_grad.",
          TfLiteTypeGetName(operand->type));
      return kTfLiteError;
    }
  }
}

}  // namespace
}  // namespace stablehlo_batch_norm_grad

TfLiteRegistration* Register_STABLEHLO_BATCH_NORM_GRAD() {
  static TfLiteRegistration r = {
      stablehlo_batch_norm_grad::Init, stablehlo_batch_norm_grad::Free,
      stablehlo_batch_norm_grad::Prepare, stablehlo_batch_norm_grad::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
