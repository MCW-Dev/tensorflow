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
#include <unordered_set>
#include <vector>

#include "Eigen/Core"  // from @eigen_archive
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/dequantize.h"
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

struct ConvolutionData {
  enum { kInputlhs, kInputrhs };
  enum { kOutput };
  int scratch_tensor_index;
  int64_t lhs_transpose_permutations[kMaxDims];
  int64_t rhs_transpose_permutations[kMaxDims];
  int64_t output_transpose_permutations[kMaxDims];
  int64_t rank;
  int64_t output_shape[kMaxDims];
  int64_t pad_input_shape[kMaxDims];
  int64_t pad_input_strides[kMaxDims];
  int64_t pad_output_strides[kMaxDims];
  int64_t pad_input_offset;
  int64_t pad_output_offset;
};

inline bool IsQuantized(const TfLiteTensor* input) {
  if (input->quantization.type == kTfLiteAffineQuantization &&
      input->quantization.params) {
    auto* quant_params =
        reinterpret_cast<TfLiteAffineQuantization*>(input->quantization.params);
    return (quant_params->scale && quant_params->scale->size > 0);
  }
  return false;
}

int64_t DivNegRoundAwayOrZero(int64_t num, int64_t denum) {
  return num < 0 ? (num - denum + 1) / denum : 0;
}

TfLiteIntArray* BuildOuputTensorDims(TfLiteNode* node) {
  ConvolutionData& convolution_data =
      *reinterpret_cast<ConvolutionData*>(node->user_data);
  TfLiteIntArray* dims = TfLiteIntArrayCreate(convolution_data.rank);
  for (int64_t i = 0; i < convolution_data.rank; ++i) {
    dims->data[i] = convolution_data.output_shape[i];
  }
  return dims;
}

bool IsUnique(const int64_t& batch_dimension, const int64_t& feature_dimension,
              const int64_t* operand, size_t N) {
  std::unordered_set<int64_t> seen_elements;
  if (!seen_elements.insert(batch_dimension).second) {
    return false;
  }
  if (!seen_elements.insert(feature_dimension).second) {
    return false;
  }
  for (int i = 0; i < N; ++i) {
    if (!seen_elements.insert(operand[i]).second) {
      return false;
    }
  }
  return true;
}

bool IsGreaterThanZero(const int64_t* operand, size_t N) {
  return std::all_of(operand, operand + N, [](int64_t x) { return x > 0; });
}

bool IsInRange(const int64_t& batch_dimension, const int64_t& feature_dimension,
               const int64_t* operand, size_t N) {
  auto is_in_range = [N](int64_t v) { return v >= 0 && v < N; };
  if (!is_in_range(batch_dimension) || !is_in_range(feature_dimension)) {
    return false;
  }
  return std::all_of(operand, operand + N - 2, is_in_range);
}

TfLiteStatus CheckParameters(TfLiteContext* context, TfLiteNode* node,
                             const TfLiteTensor* lhs, const TfLiteTensor* rhs,
                             TfLiteTensor* output) {
  const TfLiteStablehloConvolutionParams& convolution_params =
      *reinterpret_cast<TfLiteStablehloConvolutionParams*>(node->builtin_data);

  TF_LITE_ENSURE_MSG(
      context, NumDimensions(lhs) == NumDimensions(rhs),
      "'stablehlo.convolution' rank of lhs and rhs must be same");
  TF_LITE_ENSURE_TYPES_EQ(context, lhs->type, rhs->type);
  TF_LITE_ENSURE_MSG(
      context, convolution_params.num_precision_config == 2,
      "'stablehlo.convolution' size of precision_config must be two.");
  TF_LITE_ENSURE_MSG(
      context, convolution_params.num_window_strides == NumDimensions(lhs) - 2,
      "'stablehlo.convolution' size of window_stride must be rank - 2");
  TF_LITE_ENSURE_MSG(context,
                     convolution_params.num_window_strides ==
                             convolution_params.num_padding / 2 ||
                         convolution_params.num_window_strides ==
                             convolution_params.num_lhs_dilation ||
                         convolution_params.num_window_strides ==
                             convolution_params.num_rhs_dilation ||
                         convolution_params.num_window_strides ==
                             convolution_params.num_input_spatial_dimensions ||
                         convolution_params.num_window_strides ==
                             convolution_params.num_kernel_spatial_dimensions ||
                         convolution_params.num_window_strides ==
                             convolution_params.num_output_spatial_dimensions,
                     "'stablehlo.convolution' operation parameter array "
                     "sizes are not consistent.");
  TF_LITE_ENSURE_MSG(context,
                     IsGreaterThanZero(convolution_params.window_strides,
                                       convolution_params.num_window_strides),
                     "'stablehlo.convolution' the window_stride must be > 0");
  TF_LITE_ENSURE_MSG(context,
                     IsGreaterThanZero(convolution_params.lhs_dilation,
                                       convolution_params.num_lhs_dilation),
                     "'stablehlo.convolution' the lhs_dilation must be > 0");
  TF_LITE_ENSURE_MSG(context,
                     IsGreaterThanZero(convolution_params.rhs_dilation,
                                       convolution_params.num_rhs_dilation),
                     "'stablehlo.convolution' the rhs_dilation must be > 0");
  TF_LITE_ENSURE_MSG(context,
                     lhs->dims->data[convolution_params.input_batch_dimension] %
                             convolution_params.batch_group_count ==
                         0,
                     "'stablehlo.convolution' dim(lhs,input_batch_dimension) % "
                     "batch_group_count = 0");
  TF_LITE_ENSURE_MSG(
      context,
      lhs->dims->data[convolution_params.input_feature_dimension] %
              convolution_params.feature_group_count ==
          0,
      "'stablehlo.convolution' dim(lhs,input_feature_dimension) % "
      "(feature_group_count) = 0");
  TF_LITE_ENSURE_MSG(
      context,
      IsInRange(convolution_params.input_batch_dimension,
                convolution_params.input_feature_dimension,
                convolution_params.input_spatial_dimensions,
                NumDimensions(lhs)),
      "'stablehlo.convolution' the input_dimensions must be >= 0 and < rank");
  TF_LITE_ENSURE_MSG(
      context,
      IsUnique(convolution_params.input_batch_dimension,
               convolution_params.input_feature_dimension,
               convolution_params.input_spatial_dimensions,
               convolution_params.num_input_spatial_dimensions),
      "'stablehlo.convolution' the input_dimensions must be unique");
  TF_LITE_ENSURE_MSG(
      context,
      rhs->dims->data[convolution_params.kernel_input_feature_dimension] ==
          lhs->dims->data[convolution_params.input_feature_dimension] /
              convolution_params.feature_group_count,
      "'stablehlo.convolution' dim(rhs,kernel_input_feature_dimension) = "
      "Dim(lhs,input_feature_dimension) / feature_group_count");
  TF_LITE_ENSURE_MSG(
      context,
      rhs->dims->data[convolution_params.kernel_output_feature_dimension] %
              convolution_params.batch_group_count ==
          0,
      "'stablehlo.convolution' dim(rhs,kernel_output_feature_dimension) % "
      "batch_group_count = 0");
  TF_LITE_ENSURE_MSG(
      context,
      rhs->dims->data[convolution_params.kernel_output_feature_dimension] %
              convolution_params.feature_group_count ==
          0,
      "'stablehlo.convolution' dim(rhs,kernel_output_feature_dimension) % "
      "(feature_group_count) = 0");
  TF_LITE_ENSURE_MSG(
      context,
      IsInRange(convolution_params.kernel_output_feature_dimension,
                convolution_params.kernel_input_feature_dimension,
                convolution_params.kernel_spatial_dimensions,
                NumDimensions(lhs)),
      "'stablehlo.convolution' the kernel_dimensions must be >= 0 and < rank");
  TF_LITE_ENSURE_MSG(
      context,
      IsUnique(convolution_params.kernel_output_feature_dimension,
               convolution_params.kernel_input_feature_dimension,
               convolution_params.kernel_spatial_dimensions,
               convolution_params.num_kernel_spatial_dimensions),
      "'stablehlo.convolution' the kernel_dimensions must be unique");
  TF_LITE_ENSURE_MSG(
      context,
      IsInRange(convolution_params.output_batch_dimension,
                convolution_params.output_feature_dimension,
                convolution_params.output_spatial_dimensions,
                NumDimensions(lhs)),
      "'stablehlo.convolution' the output_dimensions must be >= 0 and < rank");
  TF_LITE_ENSURE_MSG(
      context,
      IsUnique(convolution_params.output_batch_dimension,
               convolution_params.output_feature_dimension,
               convolution_params.output_spatial_dimensions,
               convolution_params.num_output_spatial_dimensions),
      "'stablehlo.convolution' the output_dimensions must be unique");
  TF_LITE_ENSURE_MSG(
      context, convolution_params.feature_group_count > 0,
      "'stablehlo.convolution' the feature_group_count must be > 0");
  TF_LITE_ENSURE_MSG(
      context, convolution_params.batch_group_count > 0,
      "'stablehlo.convolution' the batch_group_count must be > 0");
  TF_LITE_ENSURE_MSG(context,
                     convolution_params.batch_group_count == 1 ||
                         convolution_params.feature_group_count == 1,
                     "'stablehlo.convolution' the batch_group_count == 1 or "
                     "feature_group_count == 1");
  if (IsQuantized(lhs) || IsQuantized(rhs) || IsQuantized(output)) {
    TF_LITE_ENSURE_MSG(
        context, IsQuantized(lhs) && IsQuantized(rhs) && IsQuantized(output),
        "'stablehlo.convolution' lhs.IsQuantized() && "
        "rhs.IsQuantized() && output.IsQuantized()");
    if (!dequantize::IsQuantizedPerChannel(rhs)) {
      TF_LITE_ENSURE_MSG(
          context, !dequantize::IsQuantizedPerChannel(output),
          "'stablehlo.convolution' If is_per_tensor_quantized(rhs), then "
          "is_per_tensor_quantized(output)");
    }
  }
  if (dequantize::IsQuantizedPerChannel(rhs)) {
    auto* rhs_quant_params =
        reinterpret_cast<TfLiteAffineQuantization*>(rhs->quantization.params);
    TF_LITE_ENSURE_MSG(
        context,
        rhs_quant_params->quantized_dimension ==
            convolution_params.kernel_output_feature_dimension,
        "'stablehlo.convolution' If is_per_axis_quantized(rhs), then "
        "quantization_dimension(rhs) = "
        "convolution_params.kernel_output_feature_dimension");
  }
  if (dequantize::IsQuantizedPerChannel(output)) {
    auto* output_quant_params = reinterpret_cast<TfLiteAffineQuantization*>(
        output->quantization.params);
    TF_LITE_ENSURE_MSG(context,
                       output_quant_params->quantized_dimension ==
                           convolution_params.output_feature_dimension,
                       "'stablehlo.convolution' If "
                       "is_per_axis_quantized(output), then "
                       "quantization_dimension(output) = "
                       "convolution_params.output_feature_dimension");
  }
  return kTfLiteOk;
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
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, output,
                                                   BuildOuputTensorDims(node)));
  return kTfLiteOk;
}

TfLiteStatus PrepareTemporaries(TfLiteContext* context, TfLiteNode* node,
                                const TfLiteTensor* lhs,
                                const TfLiteTensor* rhs, TfLiteTensor* output) {
  const TfLiteStablehloConvolutionParams& convolution_params =
      *reinterpret_cast<TfLiteStablehloConvolutionParams*>(node->builtin_data);
  ConvolutionData& convolution_data =
      *reinterpret_cast<ConvolutionData*>(node->user_data);

  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(11);

  // lhs transpose preapre
  convolution_data.lhs_transpose_permutations[0] =
      convolution_params.input_batch_dimension;
  convolution_data.lhs_transpose_permutations[1] =
      convolution_params.input_feature_dimension;
  for (int i = 0; i < convolution_data.rank - 2; ++i) {
    convolution_data.lhs_transpose_permutations[i + 2] =
        convolution_params.input_spatial_dimensions[i];
  }

  node->temporaries->data[0] = convolution_data.scratch_tensor_index;
  TfLiteTensor* lhs_transpose;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/0, &lhs_transpose));
  TfLiteIntArray* lhs_transpose_shape =
      TfLiteIntArrayCreate(convolution_data.rank);
  for (int i = 0; i < convolution_data.rank; ++i) {
    lhs_transpose_shape->data[i] =
        lhs->dims->data[convolution_data.lhs_transpose_permutations[i]];
  }

  lhs_transpose->type = lhs->type;
  lhs_transpose->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, lhs_transpose,
                                                   lhs_transpose_shape));

  // rhs transpose prepare
  convolution_data.rhs_transpose_permutations[0] =
      convolution_params.kernel_input_feature_dimension;
  for (int i = 0; i < convolution_data.rank - 2; ++i) {
    convolution_data.rhs_transpose_permutations[i + 1] =
        convolution_params.kernel_spatial_dimensions[i];
  }
  convolution_data.rhs_transpose_permutations[convolution_data.rank - 1] =
      convolution_params.kernel_output_feature_dimension;

  node->temporaries->data[1] = convolution_data.scratch_tensor_index + 1;
  TfLiteTensor* rhs_transpose;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/1, &rhs_transpose));
  TfLiteIntArray* rhs_transposed_shape =
      TfLiteIntArrayCreate(convolution_data.rank);
  for (int i = 0; i < convolution_data.rank; ++i) {
    rhs_transposed_shape->data[i] =
        rhs->dims->data[convolution_data.rhs_transpose_permutations[i]];
  }

  rhs_transpose->type = rhs->type;
  rhs_transpose->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, rhs_transpose,
                                                   rhs_transposed_shape));

  // output transpose preapre
  convolution_data.output_transpose_permutations[convolution_params
                                                     .output_batch_dimension] =
      0;
  convolution_data.output_transpose_permutations
      [convolution_params.output_feature_dimension] = 1;
  for (int i = 0; i < convolution_data.rank - 2; ++i) {
    convolution_data.output_transpose_permutations
        [convolution_params.output_spatial_dimensions[i]] = i + 2;
  }

  node->temporaries->data[2] = convolution_data.scratch_tensor_index + 2;
  TfLiteTensor* output_transpose;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/2, &output_transpose));
  TfLiteIntArray* output_transposed_shape =
      TfLiteIntArrayCreate(convolution_data.rank);
  output_transposed_shape->data[0] =
      output->dims->data[convolution_params.output_batch_dimension];
  output_transposed_shape->data[1] =
      output->dims->data[convolution_params.output_feature_dimension];
  for (int i = 0; i < convolution_data.rank - 2; ++i) {
    output_transposed_shape->data[i + 2] =
        output->dims->data[convolution_params.output_spatial_dimensions[i]];
  }

  output_transpose->type = rhs->type;
  output_transpose->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, output_transpose,
                                                   output_transposed_shape));

  // pad prepare
  convolution_data.pad_input_offset = 0;
  convolution_data.pad_output_offset = 0;
  int64_t lhs_padded_spatials[convolution_data.rank - 2];
  for (int i = convolution_data.rank - 1; i > 1; --i) {
    lhs_padded_spatials[i - 2] = lhs_transpose->dims->data[i] +
                                 (convolution_params.lhs_dilation[i - 2] - 1) *
                                     (lhs_transpose->dims->data[i] - 1) +
                                 convolution_params.padding[2 * (i - 2)] +
                                 convolution_params.padding[(2 * (i - 2)) + 1];
  }

  node->temporaries->data[3] = convolution_data.scratch_tensor_index + 3;
  TfLiteTensor* lhs_padded;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, /*index=*/3, &lhs_padded));
  TfLiteIntArray* lhs_padded_shape =
      TfLiteIntArrayCreate(convolution_data.rank);
  int64_t pad_output_shape[kMaxDims];
  pad_output_shape[0] = lhs_padded_shape->data[0] =
      lhs_transpose->dims->data[0];
  pad_output_shape[1] = lhs_padded_shape->data[1] =
      lhs_transpose->dims->data[1];
  for (int i = 0; i < convolution_data.rank - 2; ++i) {
    pad_output_shape[i + 2] = lhs_padded_shape->data[i + 2] =
        lhs_padded_spatials[i];
  }

  lhs_padded->type = rhs->type;
  lhs_padded->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(context, lhs_padded, lhs_padded_shape));

  int64_t edge_pad_high[kMaxDims];
  int64_t edge_pad_low[kMaxDims];
  int64_t interior_pad[kMaxDims];
  edge_pad_high[0] = edge_pad_low[0] = interior_pad[0] = 0;
  edge_pad_high[1] = edge_pad_low[1] = interior_pad[1] = 0;
  for (int64_t i = 2; i < convolution_data.rank; ++i) {
    edge_pad_low[i] = convolution_params.padding[2 * (i - 2)];
    edge_pad_high[i] = convolution_params.padding[2 * (i - 2) + 1];
    interior_pad[i] = convolution_params.lhs_dilation[i - 2] - 1;
  }
  int64_t pad_rank = convolution_data.rank;
  int64_t pad_output_dimension_sizes[kMaxDims];
  pad_output_dimension_sizes[pad_rank - 1] = 1;
  convolution_data.pad_output_strides[pad_rank - 1] =
      interior_pad[pad_rank - 1] + 1;
  for (int64_t i = pad_rank - 2; i >= 0; --i) {
    pad_output_dimension_sizes[i] =
        pad_output_shape[i + 1] * pad_output_dimension_sizes[i + 1];
    convolution_data.pad_output_strides[i] =
        pad_output_dimension_sizes[i] * (interior_pad[i] + 1);
  }
  for (int64_t i = 0; i < pad_rank; ++i) {
    convolution_data.pad_output_offset +=
        std::max<int64_t>(edge_pad_low[i], 0) * pad_output_dimension_sizes[i];
  }
  convolution_data.pad_input_strides[pad_rank - 1] = 1;
  for (int64_t i = pad_rank - 1; i >= 1; --i) {
    convolution_data.pad_input_strides[i - 1] =
        lhs_transpose->dims->data[i] * convolution_data.pad_input_strides[i];
  }
  for (int64_t i = 0; i < pad_rank; ++i) {
    convolution_data.pad_input_shape[i] =
        lhs_transpose->dims->data[i] +
        DivNegRoundAwayOrZero(edge_pad_low[i], interior_pad[i] + 1) +
        DivNegRoundAwayOrZero(edge_pad_high[i], interior_pad[i] + 1);
  }
  for (int64_t i = 0; i < pad_rank; ++i) {
    convolution_data.pad_input_offset -=
        DivNegRoundAwayOrZero(edge_pad_low[i], interior_pad[i] + 1) *
        convolution_data.pad_input_strides[i];
    if (edge_pad_low[i] < 0) {
      int64_t tmp_offset =
          ((interior_pad[i] + 1 + edge_pad_low[i]) % (interior_pad[i] + 1));
      if (tmp_offset < 0) {
        tmp_offset += interior_pad[i] + 1;
      }
      convolution_data.pad_output_offset +=
          tmp_offset * pad_output_dimension_sizes[i];
    }
  }
  // padding prepare end

  // slice prepare
  int64_t num_slices = convolution_params.batch_group_count *
                       convolution_params.feature_group_count;
  int64_t slice_dimension = convolution_data.rank - 1;

  node->temporaries->data[4] = convolution_data.scratch_tensor_index + 4;
  TfLiteTensor* rhs_slice;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, /*index=*/4, &rhs_slice));
  TfLiteIntArray* rhs_slice_shape = TfLiteIntArrayCreate(convolution_data.rank);
  for (size_t i = 0; i < convolution_data.rank; ++i) {
    if (i == slice_dimension) {
      rhs_slice_shape->data[i] = (rhs_transpose->dims->data[i] / num_slices);
    } else {
      rhs_slice_shape->data[i] = rhs_transpose->dims->data[i];
    }
  }
  rhs_slice->type = rhs->type;
  rhs_slice->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, rhs_slice, rhs_slice_shape));

  slice_dimension = 0;
  if (convolution_params.feature_group_count > 1) {
    slice_dimension = 1;
  }
  node->temporaries->data[5] = convolution_data.scratch_tensor_index + 5;
  TfLiteTensor* lhs_slice;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, /*index=*/5, &lhs_slice));
  TfLiteIntArray* lhs_slice_shape = TfLiteIntArrayCreate(convolution_data.rank);
  for (size_t i = 0; i < convolution_data.rank; ++i) {
    if (i == slice_dimension) {
      lhs_slice_shape->data[i] = (lhs_padded->dims->data[i] / num_slices);
    } else {
      lhs_slice_shape->data[i] = lhs_padded->dims->data[i];
    }
  }
  lhs_slice->type = rhs->type;
  lhs_slice->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, lhs_slice, lhs_slice_shape));
  // split prepare end

  // matrix multiplication prepare
  size_t rhs_slice_tensor_size = 1;
  for (size_t i = 0; i < convolution_data.rank - 1; ++i) {
    rhs_slice_tensor_size *= rhs_slice->dims->data[i];
  }

  TfLiteIntArray* lhs_new_shape = TfLiteIntArrayCreate(2);
  size_t output_spatial_size = 1;
  lhs_new_shape->data[1] = rhs_slice_tensor_size;
  for (size_t i = 0; i < convolution_data.rank - 2; ++i) {
    output_spatial_size *=
        output->dims->data[convolution_params.output_spatial_dimensions[i]];
  }
  lhs_new_shape->data[0] = output_spatial_size;

  node->temporaries->data[6] = convolution_data.scratch_tensor_index + 6;
  TfLiteTensor* lhs_matrix;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, /*index=*/6, &lhs_matrix));
  lhs_matrix->type = rhs->type;
  lhs_matrix->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, lhs_matrix, lhs_new_shape));

  TfLiteIntArray* output_matrix_shape = TfLiteIntArrayCreate(2);
  output_matrix_shape->data[0] = lhs_new_shape->data[0];
  output_matrix_shape->data[1] =
      rhs_slice->dims->data[convolution_data.rank - 1];

  node->temporaries->data[7] = convolution_data.scratch_tensor_index + 7;
  TfLiteTensor* output_matrix;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/7, &output_matrix));
  output_matrix->type = rhs->type;
  output_matrix->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, output_matrix,
                                                   output_matrix_shape));

  // Quantize prepare
  TfLiteIntArray* lhs_dequantize_shape =
      TfLiteIntArrayCreate(convolution_data.rank);
  for (int i = 0; i < convolution_data.rank; ++i) {
    lhs_dequantize_shape->data[i] = lhs->dims->data[i];
  }

  node->temporaries->data[8] = convolution_data.scratch_tensor_index + 8;
  TfLiteTensor* lhs_dequantize;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/8, &lhs_dequantize));
  lhs_dequantize->type = kTfLiteFloat32;
  lhs_dequantize->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, lhs_dequantize,
                                                   lhs_dequantize_shape));

  TfLiteIntArray* rhs_dequantize_shape =
      TfLiteIntArrayCreate(convolution_data.rank);
  for (int i = 0; i < convolution_data.rank; ++i) {
    rhs_dequantize_shape->data[i] = rhs->dims->data[i];
  }

  node->temporaries->data[9] = convolution_data.scratch_tensor_index + 9;
  TfLiteTensor* rhs_dequantize;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/9, &rhs_dequantize));
  rhs_dequantize->type = kTfLiteFloat32;
  rhs_dequantize->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, rhs_dequantize,
                                                   rhs_dequantize_shape));

  TfLiteIntArray* output_dequantize_shape =
      TfLiteIntArrayCreate(convolution_data.rank);
  for (int i = 0; i < convolution_data.rank; ++i) {
    output_dequantize_shape->data[i] = output->dims->data[i];
  }

  node->temporaries->data[10] = convolution_data.scratch_tensor_index + 10;
  TfLiteTensor* output_dequantize;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/10,
                                              &output_dequantize));
  output_dequantize->type = kTfLiteFloat32;
  output_dequantize->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, output_dequantize,
                                                   output_dequantize_shape));

  return kTfLiteOk;
}

void* Init(TfLiteContext* context, const char* options, size_t options_len) {
  ConvolutionData* convolution_data = new ConvolutionData();
  context->AddTensors(context, 11, &convolution_data->scratch_tensor_index);
  return convolution_data;
}

void Free(TfLiteContext* context, void* node_data) {
  delete reinterpret_cast<ConvolutionData*>(node_data);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* lhs_tensor =
      GetInput(context, node, ConvolutionData::kInputlhs);
  const TfLiteTensor* rhs_tensor =
      GetInput(context, node, ConvolutionData::kInputrhs);
  TfLiteTensor* output_tensor =
      GetOutput(context, node, ConvolutionData::kOutput);

  TF_LITE_ENSURE_OK(context, CheckParameters(context, node, lhs_tensor,
                                             rhs_tensor, output_tensor));

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

  TfLiteTensor* lhs_matrix = GetTemporary(context, node, 6);
  TfLiteTensor* output_matrix = GetTemporary(context, node, 7);

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
  ConvolutionData& convolution_data =
      *reinterpret_cast<ConvolutionData*>(node->user_data);
  const auto& convolution_params =
      *reinterpret_cast<TfLiteStablehloConvolutionParams*>(node->builtin_data);

  // Transpose
  TfLiteTensor* lhs_transpose = GetTemporary(context, node, 0);
  TfLiteTensor* rhs_transpose = GetTemporary(context, node, 1);
  TfLiteTensor* output_transpose = GetTemporary(context, node, 2);

  RuntimeShape lhs_transpose_shape(GetTensorShape(lhs_transpose));
  RuntimeShape lhs_shape(GetTensorShape(lhs));
  TransposeParams lhs_params;
  lhs_params.perm_count = convolution_data.rank;
  for (int i = 0; i < NumDimensions(lhs); ++i) {
    lhs_params.perm[i] = convolution_data.lhs_transpose_permutations[i];
  }
  optimized_ops::Transpose(lhs_params, lhs_shape, GetTensorData<DataType>(lhs),
                           lhs_transpose_shape,
                           GetTensorData<DataType>(lhs_transpose));

  RuntimeShape rhs_transposed_shape(GetTensorShape(rhs_transpose));
  RuntimeShape rhs_shape(GetTensorShape(rhs));
  TransposeParams rhs_params;
  rhs_params.perm_count = convolution_data.rank;
  for (int i = 0; i < NumDimensions(rhs); ++i) {
    rhs_params.perm[i] = convolution_data.rhs_transpose_permutations[i];
  }
  optimized_ops::Transpose(rhs_params, rhs_shape, GetTensorData<DataType>(rhs),
                           rhs_transposed_shape,
                           GetTensorData<DataType>(rhs_transpose));

  RuntimeShape output_transposed_shape(GetTensorShape(output_transpose));
  RuntimeShape output_shape(GetTensorShape(output));
  TransposeParams output_params;
  output_params.perm_count = convolution_data.rank;
  for (int i = 0; i < convolution_data.rank; ++i) {
    output_params.perm[i] = convolution_data.output_transpose_permutations[i];
  }

  // pad
  TfLiteTensor* lhs_padded = GetTemporary(context, node, 3);

  int lhs_pad_size = 1;
  for (int i = 0; i < convolution_data.rank; ++i)
    lhs_pad_size *= lhs_padded->dims->data[i];
  memset(GetTensorData<DataType>(lhs_padded), 0,
         lhs_pad_size * sizeof(DataType));

  optimized_ops::StridedCopy<DataType>(
      convolution_data.rank,
      GetTensorData<DataType>(lhs) + convolution_data.pad_input_offset,
      convolution_data.pad_input_shape, convolution_data.pad_input_strides,
      GetTensorData<DataType>(lhs_padded) + convolution_data.pad_output_offset,
      convolution_data.pad_output_strides, sizeof(DataType),
      /*depth=*/0);

  // slice
  TfLiteTensor* lhs_slice = GetTemporary(context, node, 5);
  TfLiteTensor* rhs_slice = GetTemporary(context, node, 4);
  RuntimeShape lhs_padded_shape(GetTensorShape(lhs_padded));
  RuntimeShape lhs_slice_shape(GetTensorShape(lhs_slice));
  RuntimeShape rhs_slice_shape(GetTensorShape(rhs_slice));
  SliceParams slice_params;
  slice_params.begin_count = convolution_data.rank;
  slice_params.size_count = convolution_data.rank;
  memset(slice_params.begin, 0, sizeof(slice_params.begin));
  int32_t lhs_size[convolution_data.rank];
  int32_t rhs_size[convolution_data.rank];
  for (int i = 0; i < convolution_data.rank; ++i) {
    lhs_size[i] = lhs_slice->dims->data[i];
    rhs_size[i] = rhs_slice->dims->data[i];
  }

  int output_channel = 0;

  for (int i = 0; i < convolution_params.batch_group_count *
                          convolution_params.feature_group_count;
       ++i) {
    // slicing lhs
    if (convolution_params.batch_group_count != 1) {
      slice_params.begin[0] = i * lhs_slice->dims->data[0];
    } else if (convolution_params.feature_group_count != 1) {
      slice_params.begin[1] = i * lhs_slice->dims->data[1];
    }
    slice_params.begin[convolution_data.rank - 1] = 0;
    std::copy(lhs_size, lhs_size + convolution_data.rank, slice_params.size);
    optimized_ops::Slice<DataType>(slice_params, lhs_padded_shape, lhs_padded,
                                   lhs_slice_shape, lhs_slice);

    // slicing rhs
    if (convolution_params.batch_group_count != 1) {
      slice_params.begin[0] = 0;
    } else if (convolution_params.feature_group_count != 1) {
      slice_params.begin[1] = 0;
    }
    slice_params.begin[convolution_data.rank - 1] =
        i * rhs_slice->dims->data[convolution_data.rank - 1];
    std::copy(rhs_size, rhs_size + convolution_data.rank, slice_params.size);
    optimized_ops::Slice<DataType>(slice_params, GetTensorShape(rhs_transpose),
                                   rhs_transpose, rhs_slice_shape, rhs_slice);

    TF_LITE_ENSURE_OK(
        context, ConvolutionImpl<DataType>(context, node, lhs_slice, rhs_slice,
                                           output_transpose, output_channel));
  }
  optimized_ops::Transpose(output_params, output_transposed_shape,
                           GetTensorData<DataType>(output_transpose),
                           output_shape, GetTensorData<DataType>(output));
}

template <typename DataType>
TfLiteStatus EvalQuantize(TfLiteContext* context, TfLiteNode* node,
                          const TfLiteTensor* lhs, const TfLiteTensor* rhs,
                          TfLiteTensor* output) {
  TfLiteTensor* lhs_dequantize = GetTemporary(context, node, 8);
  TfLiteTensor* rhs_dequantize = GetTemporary(context, node, 9);
  TfLiteTensor* output_dequantize = GetTemporary(context, node, 10);
  dequantize::DequantizeImpl<dequantize::KernelType::kGenericOptimized>(
      context, node, lhs, lhs_dequantize);
  dequantize::DequantizeImpl<dequantize::KernelType::kGenericOptimized>(
      context, node, rhs, rhs_dequantize);
  EvalWithType<float>(context, node, lhs_dequantize, rhs_dequantize,
                      output_dequantize);
  RuntimeShape output_shape(GetTensorShape(output));
  RuntimeShape output_dequantize_shape(GetTensorShape(output_dequantize));
  if (dequantize::IsQuantizedPerChannel(output)) {
    const auto* quantization_params =
        reinterpret_cast<const TfLiteAffineQuantization*>(
            output->quantization.params);
    PerChannelQuantizationParams per_channel_op_params;
    per_channel_op_params.quantized_dimension =
        quantization_params->quantized_dimension;
    per_channel_op_params.scale = quantization_params->scale->data;
    per_channel_op_params.zero_point = quantization_params->zero_point->data;
    reference_ops::PerChannelQuantize(
        per_channel_op_params, output_dequantize_shape,
        GetTensorData<float>(output_dequantize), output_shape,
        GetTensorData<DataType>(output));
  } else {
    tflite::QuantizationParams op_params;
    op_params.zero_point = output->params.zero_point;
    op_params.scale = output->params.scale;
    optimized_ops::AffineQuantize<DataType>(
        op_params, output_dequantize_shape,
        GetTensorData<float>(output_dequantize), output_shape,
        GetTensorData<DataType>(output));
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* lhs_tensor =
      GetInput(context, node, ConvolutionData::kInputlhs);
  const TfLiteTensor* rhs_tensor =
      GetInput(context, node, ConvolutionData::kInputrhs);
  TfLiteTensor* output_tensor =
      GetOutput(context, node, ConvolutionData::kOutput);

  TfLiteType data_type = lhs_tensor->type;

  if (data_type == kTfLiteInt8) {
    EvalQuantize<int8_t>(context, node, lhs_tensor, rhs_tensor, output_tensor);
  } else if (data_type == kTfLiteInt16) {
    EvalQuantize<int16_t>(context, node, lhs_tensor, rhs_tensor, output_tensor);
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
