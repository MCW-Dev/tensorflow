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

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/portable_tensor.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/tensor_slice_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_complex {
namespace {

constexpr int kInputRTensor = 0;
constexpr int kInputITensor = 1;
constexpr int kOutputTensor = 0;

template <typename DataType>
TfLiteStatus EvalImpl(const TfLiteTensor* operand_a,
                      const TfLiteTensor* operand_b, TfLiteTensor* result) {
  const DataType* data_a =
      reinterpret_cast<const DataType*>(operand_a->data.raw);
  const DataType* data_b =
      reinterpret_cast<const DataType*>(operand_b->data.raw);
  std::complex<DataType>* result_data =
      reinterpret_cast<std::complex<DataType>*>(result->data.raw);

  const int num_elements = NumElements(operand_a);

  for (int i = 0; i < num_elements; ++i) {
    result_data[i] = std::complex<DataType>(data_a[i], data_b[i]);
  }

  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input_a;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputRTensor, &input_a));
  const TfLiteTensor* input_b;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputITensor, &input_b));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, input_a->type, input_b->type);
  TF_LITE_ENSURE_EQ(context, TfLiteIntArrayEqual(input_a->dims, output->dims),
                    true);
  if (input_a->type == kTfLiteFloat32) {
    TF_LITE_ENSURE_EQ(context, output->type, kTfLiteComplex64);
  } else if (input_a->type == kTfLiteFloat64) {
    TF_LITE_ENSURE_EQ(context, output->type, kTfLiteComplex128);
  }
  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input_a->dims));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input_a;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputRTensor, &input_a));
  const TfLiteTensor* input_b;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputITensor, &input_b));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  TfLiteType data_type = input_a->type;

  if (data_type == kTfLiteFloat32) {
    return EvalImpl<float>(input_a, input_b, output);
  } else if (data_type == kTfLiteFloat64) {
    return EvalImpl<_Float64>(input_a, input_b, output);
  } else {
    TF_LITE_KERNEL_LOG(context, "(Index Type: %s) currently not supported.\n",
                       TfLiteTypeGetName(data_type));
    return kTfLiteError;
  }
}

}  // namespace
}  // namespace stablehlo_complex

TfLiteRegistration* Register_STABLEHLO_COMPLEX() {
  static TfLiteRegistration r = {/*.init=*/nullptr,
                                 /*.free=*/nullptr,
                                 /*.prepare=*/stablehlo_complex::Prepare,
                                 /*.invoke=*/stablehlo_complex::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
