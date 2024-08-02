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
#include <vector>

#include "Eigen/Core"  // from @eigen_archive
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/tensor_slice_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_convolution {
namespace {

static constexpr int kMaxDims = 6;
static constexpr int kMaxTemporaryTensors = 2;

struct ConvolutionData {
  enum { kInputLhs, kInputRhs };
  enum { kOutput };
  int scratch_tensor_index;
  int64_t rank;
  int64_t output_shape[kMaxDims];
};

int64_t DivNegRoundAwayOrZero(int64_t num, int64_t denum) {
  return num < 0 ? (num - denum + 1) / denum : 0;
}

TfLiteIntArray* BuildOuputTensorDims(ConvolutionData& convolution_data) {
  TfLiteIntArray* dims = TfLiteIntArrayCreate(convolution_data.rank);
  for (int64_t i = 0; i < convolution_data.rank; ++i) {
    dims->data[i] = convolution_data.output_shape[i];
  }
  return dims;
}

TfLiteStatus PrepareOutput(TfLiteContext* context, TfLiteNode* node,
                           const TfLiteTensor* lhs, const TfLiteTensor* rhs,
                           TfLiteTensor* output) {
  const TfLiteStablehloConvolutionParams& convolution_params =
      *reinterpret_cast<TfLiteStablehloConvolutionParams*>(node->builtin_data);
  ConvolutionData& convolution_data =
      *reinterpret_cast<ConvolutionData*>(node->user_data);
  convolution_data.rank = NumDimensions(lhs);

  convolution_data.output_shape[convolution_params.output_batch_dimension] =
      lhs->dims->data[convolution_params.input_batch_dimension] /
      convolution_params.batch_group_count;
  convolution_data.output_shape[convolution_params.output_feature_dimension] =
      rhs->dims->data[convolution_params.kernel_output_feature_dimension];
  for (int i = 0; i < convolution_data.rank - 2; ++i) {
    int64_t lhs_dim =
        lhs->dims->data[convolution_params.input_spatial_dimensions[i]];
    int64_t rhs_dim =
        rhs->dims->data[convolution_params.kernel_spatial_dimensions[i]];
    int64_t lhs_dilation = convolution_params.lhs_dilation[i];
    int64_t rhs_dilation = convolution_params.rhs_dilation[i];
    int64_t window_stride = convolution_params.window_strides[i];

    int64_t dilated_lhs_shape =
        (lhs_dim == 0) ? 0 : (lhs_dim - 1) * lhs_dilation + 1;
    int64_t padded_lhs_shape = dilated_lhs_shape +
                               convolution_params.padding[2 * i] +
                               convolution_params.padding[2 * i + 1];
    int64_t dilated_rhs_shape =
        (rhs_dim == 0) ? 0 : (rhs_dim - 1) * rhs_dilation + 1;

    bool is_empty_window =
        (padded_lhs_shape == 0 || dilated_rhs_shape > padded_lhs_shape);
    int64_t expected_output_shape =
        is_empty_window ? 0
                        : std::floor((padded_lhs_shape - dilated_rhs_shape) /
                                     window_stride) +
                              1;

    convolution_data
        .output_shape[convolution_params.output_spatial_dimensions[i]] =
        expected_output_shape;
  }
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(context, output,
                                     BuildOuputTensorDims(convolution_data)));
  return kTfLiteOk;
}

TfLiteStatus PrepareTemporaries(TfLiteContext* context, TfLiteNode* node,
                                const TfLiteTensor* lhs,
                                const TfLiteTensor* rhs, TfLiteTensor* output) {
  const TfLiteStablehloConvolutionParams& convolution_params =
      *reinterpret_cast<TfLiteStablehloConvolutionParams*>(node->builtin_data);
  ConvolutionData& convolution_data =
      *reinterpret_cast<ConvolutionData*>(node->user_data);
  context->AddTensors(context, kMaxTemporaryTensors,
                      &convolution_data.scratch_tensor_index);

  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(2);

  // matrix multiplication prepare
  size_t rhs_tensor_size = 1;
  for (size_t i = 0; i < convolution_data.rank - 1; ++i) {
    rhs_tensor_size *= rhs->dims->data[i];
  }

  TfLiteIntArray* lhs_new_shape = TfLiteIntArrayCreate(2);
  size_t output_spatial_size = 1;
  lhs_new_shape->data[1] = rhs_tensor_size;
  for (size_t i = 0; i < convolution_data.rank - 2; ++i) {
    output_spatial_size *=
        output->dims->data[convolution_params.output_spatial_dimensions[i]];
  }
  lhs_new_shape->data[0] = output_spatial_size;

  node->temporaries->data[0] = convolution_data.scratch_tensor_index;
  TfLiteTensor* lhs_matrix;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, /*index=*/0, &lhs_matrix));
  lhs_matrix->type = rhs->type;
  lhs_matrix->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, lhs_matrix, lhs_new_shape));

  TfLiteIntArray* output_matrix_shape = TfLiteIntArrayCreate(2);
  output_matrix_shape->data[0] = lhs_new_shape->data[0];
  output_matrix_shape->data[1] = rhs->dims->data[convolution_data.rank - 1];

  node->temporaries->data[1] = convolution_data.scratch_tensor_index + 1;
  TfLiteTensor* output_matrix;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/1, &output_matrix));
  output_matrix->type = rhs->type;
  output_matrix->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, output_matrix,
                                                   output_matrix_shape));

  return kTfLiteOk;
}

void* Init(TfLiteContext* context, const char* options, size_t options_len) {
  ConvolutionData* convolution_data = new ConvolutionData();
  return convolution_data;
}

void Free(TfLiteContext* context, void* node_data) {
  delete reinterpret_cast<ConvolutionData*>(node_data);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* lhs_tensor =
      GetInput(context, node, ConvolutionData::kInputLhs);
  const TfLiteTensor* rhs_tensor =
      GetInput(context, node, ConvolutionData::kInputRhs);
  TfLiteTensor* output_tensor =
      GetOutput(context, node, ConvolutionData::kOutput);

  TF_LITE_ENSURE_OK(context, PrepareOutput(context, node, lhs_tensor,
                                           rhs_tensor, output_tensor));

  TF_LITE_ENSURE_OK(context, PrepareTemporaries(context, node, lhs_tensor,
                                                rhs_tensor, output_tensor));

  return kTfLiteOk;
}

template <typename DataType>
TfLiteStatus ConvolutionImpl(TfLiteContext* context, TfLiteNode* node,
                             const TfLiteTensor* lhs,
                             const TfLiteTensor* rhs_matrix,
                             TfLiteTensor* output, int& output_channel) {
  const TfLiteStablehloConvolutionParams& convolution_params =
      *reinterpret_cast<TfLiteStablehloConvolutionParams*>(node->builtin_data);
  ConvolutionData& convolution_data =
      *reinterpret_cast<ConvolutionData*>(node->user_data);

  const DataType* lhs_buffer = GetTensorData<DataType>(lhs);
  const DataType* rhs_buffer = GetTensorData<DataType>(rhs_matrix);
  DataType* output_buffer = GetTensorData<DataType>(output);

  size_t rhs_tensor_size = 1;
  size_t rhs_spatial_size = 1;
  size_t output_spacial_size = 1;
  for (size_t i = 1; i < convolution_data.rank; ++i) {
    rhs_tensor_size *= rhs_matrix->dims->data[i];
    if (i > 1) {
      output_spacial_size *= output->dims->data[i];
      rhs_spatial_size *= rhs_matrix->dims->data[i - 1];
    }
  }

  TfLiteTensor* lhs_matrix = GetTemporary(context, node, 0);
  TfLiteTensor* output_matrix = GetTemporary(context, node, 1);

  DataType* lhs_matrix_buffer = GetTensorData<DataType>(lhs_matrix);
  DataType* output_matrix_buffer = GetTensorData<DataType>(output_matrix);

  for (size_t i = 0; i < lhs->dims->data[0]; ++i) {
    for (size_t j = 0; j < output_spacial_size; ++j) {
      int64_t output_dims[convolution_data.rank];
      size_t output_depth = 1;
      for (size_t m = convolution_data.rank - 1; m > 1; --m) {
        output_dims[m] = (j / output_depth) % output->dims->data[m];
        output_depth *= output->dims->data[m];
      }
      for (size_t k = 0; k < lhs->dims->data[1]; ++k) {
        for (size_t l = 0; l < rhs_spatial_size; ++l) {
          int64_t filter_spacials[convolution_data.rank - 2];
          size_t depth = 1;
          for (size_t m = convolution_data.rank - 1; m > 1; --m) {
            filter_spacials[m - 2] =
                (l / depth) % rhs_matrix->dims->data[m - 1];
            depth *= rhs_matrix->dims->data[m - 1];
          }

          int64_t lhs_dims[convolution_data.rank];
          lhs_dims[0] = i;
          lhs_dims[1] = k;
          depth = 1;
          size_t lhs_index = 0;
          for (int64_t m = convolution_data.rank - 1; m >= 0; --m) {
            if (m > 1)
              lhs_dims[m] =
                  output_dims[m] * convolution_params.window_strides[m - 2] +
                  filter_spacials[m - 2] *
                      convolution_params.rhs_dilation[m - 2];
            lhs_index += lhs_dims[m] * depth;
            depth *= lhs->dims->data[m];
          }
          size_t batch_skip = (k * rhs_spatial_size) +
                              (j * lhs->dims->data[1] * rhs_spatial_size);
          lhs_matrix_buffer[l + batch_skip] = lhs_buffer[lhs_index];
        }
      }
    }

    // Calculate the output.
    int batch_size = 1;
    int n = lhs_matrix->dims->data[0];
    int m = rhs_matrix->dims->data[convolution_data.rank - 1];
    int p = lhs_matrix->dims->data[1];

    using EigenMatrixMapRowMajorConst =
        Eigen::Map<const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>>;
    using EigenMatrixMapColMajorConst =
        Eigen::Map<const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::ColMajor>>;
    using EigenMatrixMapRowMajorMutable =
        Eigen::Map<Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>>;

    for (int batch = 0; batch < batch_size; ++batch) {
      EigenMatrixMapRowMajorConst eigen_lhs(
          GetTensorData<DataType>(lhs_matrix) + batch * n * p, n, p);
      EigenMatrixMapColMajorConst eigen_rhs(
          GetTensorData<DataType>(rhs_matrix) + batch * m * p, p, m);
      EigenMatrixMapRowMajorMutable eigen_dst(
          GetTensorData<DataType>(output_matrix) + batch * n * m, n, m);

      if (m == 1) {
        eigen_dst.col(0).noalias() = eigen_lhs * eigen_rhs.col(0);
      } else if (n == 1) {
        eigen_dst.row(0).noalias() = eigen_lhs.row(0) * eigen_rhs;
      } else {
        eigen_dst.noalias() = eigen_lhs * eigen_rhs;
      }
    }

    for (size_t k = 0;
         k < output_matrix->dims->data[0] * output_matrix->dims->data[1]; ++k) {
      size_t output_index =
          (k / output_matrix->dims->data[1]) % output_matrix->dims->data[0] +
          i * (output_matrix->dims->data[0] * output_matrix->dims->data[1]) +
          output_matrix->dims->data[0] *
              (output_channel + (k % output_matrix->dims->data[1]));
      output_buffer[output_index] = output_matrix_buffer[k];
    }
  }

  output_channel += rhs_matrix->dims->data[convolution_data.rank - 1];

  return kTfLiteOk;
}

template <typename DataType>
void EvalWithType(TfLiteContext* context, TfLiteNode* node,
                  const TfLiteTensor* lhs, const TfLiteTensor* rhs,
                  TfLiteTensor* output) {
  int output_channel = 0;
  TF_LITE_ENSURE_OK(context, ConvolutionImpl<DataType>(context, node, lhs, rhs,
                                                       output, output_channel));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* lhs_tensor =
      GetInput(context, node, ConvolutionData::kInputLhs);
  const TfLiteTensor* rhs_tensor =
      GetInput(context, node, ConvolutionData::kInputRhs);
  TfLiteTensor* output_tensor =
      GetOutput(context, node, ConvolutionData::kOutput);

  TfLiteType data_type = lhs_tensor->type;

  if (data_type == kTfLiteInt8) {
    EvalWithType<int8_t>(context, node, lhs_tensor, rhs_tensor, output_tensor);
  } else if (data_type == kTfLiteInt16) {
    EvalWithType<int16_t>(context, node, lhs_tensor, rhs_tensor, output_tensor);
  } else if (data_type == kTfLiteInt32) {
    EvalWithType<int32_t>(context, node, lhs_tensor, rhs_tensor, output_tensor);
  } else if (data_type == kTfLiteInt64) {
    EvalWithType<int64_t>(context, node, lhs_tensor, rhs_tensor, output_tensor);
  } else if (data_type == kTfLiteFloat32) {
    EvalWithType<float>(context, node, lhs_tensor, rhs_tensor, output_tensor);
  } else if (data_type == kTfLiteFloat16) {
    EvalWithType<Eigen::half>(context, node, lhs_tensor, rhs_tensor,
                              output_tensor);
  } else if (data_type == kTfLiteBFloat16) {
    EvalWithType<Eigen::bfloat16>(context, node, lhs_tensor, rhs_tensor,
                                  output_tensor);
  } else {
    TF_LITE_KERNEL_LOG(context, "(Index Type: %s) currently not supported.\n",
                       TfLiteTypeGetName(data_type));
    return TfLiteStatus::kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace
}  // namespace stablehlo_convolution

TfLiteRegistration* Register_STABLEHLO_CONVOLUTION() {
  static TfLiteRegistration r = {/*.init=*/stablehlo_convolution::Init,
                                 /*.free=*/stablehlo_convolution::Free,
                                 /*.prepare=*/stablehlo_convolution::Prepare,
                                 /*.invoke=*/stablehlo_convolution::Eval};
  return &r;
}
}  // namespace builtin
}  // namespace ops
}  // namespace tflite
