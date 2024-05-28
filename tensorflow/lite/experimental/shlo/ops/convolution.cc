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

#include "tensorflow/lite/experimental/shlo/ops/convolution.h"

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/convolution_helper_functions.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

// Constraints Check
absl::Status CheckParameters(ConvolutionOp& op, const Tensor& lhs,
                             const Tensor& rhs, const Tensor& output) {
  if (op.attributes.precision_configs.size() != 2) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: Size of precision_config must be two.");
  }
  Axis rank = lhs.Rank();
  if (lhs.Rank() != rhs.Rank()) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: Rank of lhs and rhs must be same");
  }
  if (output.Rank() != lhs.Rank()) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: Rank of lhs and output must be same");
  }
  if (!lhs.IsQuantized()) {
    SHLO_REF_RETURN_ON_ERROR(
        CheckSameBaselineType(CheckCtx("Convolution"), lhs, rhs));
    SHLO_REF_RETURN_ON_ERROR(
        CheckSameBaselineType(CheckCtx("Convolution"), lhs, output));
  }
  if (op.attributes.window_strides.size() != rank - 2) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: Size of window_stride must be rank - 2");
  }
  if (!IsGreaterThanZero(op.attributes.window_strides)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: The window_stride must be > 0");
  }
  if (op.attributes.padding.shape().Dim(0) != rank - 2 ||
      op.attributes.padding.shape().Dim(1) != 2) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: Shape of padding must be [rank - 2, 2]");
  }
  if (op.attributes.lhs_dilation.size() != rank - 2) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: Shape of lhs_dilation must be rank - 2");
  }
  if (!IsGreaterThanZero(op.attributes.lhs_dilation)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: The lhs_dilation must be > 0");
  }
  if (op.attributes.rhs_dilation.size() != rank - 2) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: Shape of rhs_dilation must be rank - 2");
  }
  if (!IsGreaterThanZero(op.attributes.rhs_dilation)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: The rhs_dilation must be > 0");
  }
  if (lhs.shape().Dim(static_cast<Axis>(op.attributes.input_batch_dimension)) %
          op.attributes.batch_group_count !=
      0) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: "
        "Dim(lhs,input_batch_dimension) % batch_group_count = 0");
  }
  if (lhs.shape().Dim(
          static_cast<Axis>(op.attributes.input_feature_dimension)) %
          op.attributes.feature_group_count !=
      0) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: "
        "Dim(lhs,input_feature_dimension) % (feature_group_count) = 0");
  }
  if (op.attributes.input_spatial_dimensions.size() != rank - 2) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: "
        "Size of input_spatial_dimensions must be rank - 2");
  }
  if (!IsUnique(op.attributes.input_batch_dimension,
                op.attributes.input_feature_dimension,
                op.attributes.input_spatial_dimensions)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: The input_dimensions must be unique");
  }
  if (!IsInRange(op.attributes.input_batch_dimension,
                 op.attributes.input_feature_dimension,
                 op.attributes.input_spatial_dimensions, rank)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: The input_dimensions must be >= 0 and < rank");
  }
  if (rhs.shape().Dim(
          static_cast<Axis>(op.attributes.kernel_input_feature_dimension)) !=
      lhs.shape().Dim(
          static_cast<Axis>(op.attributes.input_feature_dimension)) /
          op.attributes.feature_group_count) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: "
        "Dim(rhs,kernel_input_feature_dimension) = "
        "Dim(lhs,input_feature_dimension) / feature_group_count");
  }
  if (rhs.shape().Dim(
          static_cast<Axis>(op.attributes.kernel_output_feature_dimension)) %
          op.attributes.batch_group_count !=
      0) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: "
        "Dim(rhs,kernel_output_feature_dimension) % batch_group_count = 0");
  }
  if (rhs.shape().Dim(
          static_cast<Axis>(op.attributes.kernel_output_feature_dimension)) %
          op.attributes.feature_group_count !=
      0) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: "
        "Dim(rhs,kernel_output_feature_dimension) % (feature_group_count) = 0");
  }
  if (op.attributes.kernel_spatial_dimensions.size() != rank - 2) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: "
        "Size of kernel_spatial_dimensions must be rank - 2");
  }
  if (!IsUnique(op.attributes.kernel_output_feature_dimension,
                op.attributes.kernel_input_feature_dimension,
                op.attributes.kernel_spatial_dimensions)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: "
        "The kernel_dimensions must be unique");
  }
  if (!IsInRange(op.attributes.kernel_output_feature_dimension,
                 op.attributes.kernel_input_feature_dimension,
                 op.attributes.kernel_spatial_dimensions, rank)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: The kernel_dimensions must be >= 0 and < rank");
  }
  if (op.attributes.output_spatial_dimensions.size() != rank - 2) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: "
        "Size of output_spatial_dimensions must be rank - 2");
  }
  if (!IsUnique(op.attributes.output_batch_dimension,
                op.attributes.output_feature_dimension,
                op.attributes.output_spatial_dimensions)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: "
        "The output_dimensions must be unique");
  }
  if (!IsInRange(op.attributes.output_batch_dimension,
                 op.attributes.output_feature_dimension,
                 op.attributes.output_spatial_dimensions, rank)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: The output_dimensions must be >= 0 and < rank");
  }
  if (op.attributes.feature_group_count <= 0) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: The feature_group_count must be > 0");
  }
  if (op.attributes.batch_group_count <= 0) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: The batch_group_count must be > 0");
  }
  if (op.attributes.batch_group_count != 1 &&
      op.attributes.feature_group_count != 1) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: The batch_group_count == 1 or "
        "feature_group_count == 1");
  }
  if (output.shape().Dim(
          static_cast<Axis>(op.attributes.output_batch_dimension)) !=
      lhs.shape().Dim(static_cast<Axis>(op.attributes.input_batch_dimension)) /
          op.attributes.batch_group_count) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: "
        "output.shape().Dim(output_batch_dimension) == "
        "lhs.shape().Dim(input_batch_dimension) / batch_group_count");
  }
  if (output.shape().Dim(
          static_cast<Axis>(op.attributes.output_feature_dimension)) !=
      rhs.shape().Dim(
          static_cast<Axis>(op.attributes.kernel_output_feature_dimension))) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: "
        "output.shape().Dim(output_feature_dimension) == "
        "rhs.shape().Dim(kernel_output_feature_dimension)");
  }
  if (!CheckOutputSpatial(op, lhs, rhs, output)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: "
        "output.shape().Dim(spatial_dim) is not properly set");
  }
  if (lhs.IsQuantized() || rhs.IsQuantized() || output.IsQuantized()) {
    if (!(lhs.IsQuantized() && rhs.IsQuantized() && output.IsQuantized())) {
      return absl::FailedPreconditionError(
          "stablehlo.convolution: lhs.IsQuantized() && "
          "rhs.IsQuantized() && output.IsQuantized()");
    }
    if (rhs.IsPerTensorQuantized()) {
      if (!(output.IsPerTensorQuantized())) {
        return absl::FailedPreconditionError(
            "stablehlo.convolution: If is_per_tensor_quantized(rhs), then "
            "is_per_tensor_quantized(output)");
      }
      // This will be replaced by Tensor::ExpressedType()
      if (lhs.quantized_per_tensor_element_type().ExpressedType() !=
          rhs.quantized_per_tensor_element_type().ExpressedType()) {
        return absl::FailedPreconditionError(
            "stablehlo.convolution: The expressed_type of lhs must be the same "
            "as the expressed_type of rhs.");
      }
      if (output.quantized_per_tensor_element_type().ExpressedType() !=
          rhs.quantized_per_tensor_element_type().ExpressedType()) {
        return absl::FailedPreconditionError(
            "stablehlo.convolution: The expressed_type of output must be the "
            "same as the expressed_type of rhs.");
      }
    }
    if (rhs.IsPerAxisQuantized()) {
      if (rhs.quantized_per_axis_element_type().QuantizedDimension() !=
          op.attributes.kernel_output_feature_dimension) {
        return absl::FailedPreconditionError(
            "stablehlo.convolution:  If is_per_axis_quantized(rhs), then "
            "quantization_dimension(rhs) = "
            "op.attributes.kernel_output_feature_dimension");
      }
      // This will be replaced by Tensor::ExpressedType()
      if (lhs.quantized_per_tensor_element_type().ExpressedType() !=
          rhs.quantized_per_axis_element_type().ExpressedType()) {
        return absl::FailedPreconditionError(
            "stablehlo.convolution: The expressed_type of lhs must be the same "
            "as the expressed_type of rhs.");
      }
      if (output.IsPerTensorQuantized()) {
        if (output.quantized_per_tensor_element_type().ExpressedType() !=
            rhs.quantized_per_axis_element_type().ExpressedType()) {
          return absl::FailedPreconditionError(
              "stablehlo.convolution: The expressed_type of output must be the "
              "same as the expressed_type of rhs.");
        }
      }
    }
    if (output.IsPerAxisQuantized()) {
      if (output.quantized_per_axis_element_type().QuantizedDimension() !=
          op.attributes.output_feature_dimension) {
        return absl::FailedPreconditionError(
            "stablehlo.convolution:  If "
            "is_per_axis_quantized(output), then "
            "quantization_dimension(output) = "
            "op.attributes.output_feature_dimension");
      }
      // This will be replaced by Tensor::ExpressedType()
      if (output.quantized_per_axis_element_type().ExpressedType() !=
          rhs.quantized_per_axis_element_type().ExpressedType()) {
        return absl::FailedPreconditionError(
            "stablehlo.convolution: The expressed_type of output must be the "
            "same as the expressed_type of rhs.");
      }
    }
    if (lhs.StorageType() != rhs.StorageType()) {
      return absl::FailedPreconditionError(
          "stablehlo.convolution: "
          "The storage_type of lhs must be the same as storage_type of rhs");
    }
  }
  return absl::OkStatus();
}

// Will be updated with data preparation for padding and other functions
template <DataType storage_type>
absl::Status PrepareImpl(ConvolutionOp& op, const Tensor& lhs,
                         const Tensor& rhs, Tensor& output) {
  using StorageT = StorageType<storage_type>;

  SHLO_REF_RETURN_ON_ERROR(CheckParameters(op, lhs, rhs, output));

  return absl::OkStatus();
}

// Convolution
template <DataType storage_type>
absl::Status ConvolutionImpl(ConvolutionOp& op, size_t& output_channel,
                             const Tensor& lhs, const Tensor& rhs,
                             Tensor& output) {
  using StorageT = StorageType<storage_type>;
  const StorageT* lhs_buffer = lhs.GetDataAs<storage_type>();
  const StorageT* rhs_buffer = rhs.GetDataAs<storage_type>();
  StorageT* output_buffer = output.GetDataAs<storage_type>();

  size_t rhs_tensor_size = 1;
  size_t rhs_spatial_size = 1;
  size_t output_spatial_size = 1;
  for (size_t i = 1; i < rhs.Rank(); ++i) {
    rhs_tensor_size *= rhs.shape().Dim(i);
    if (i > 1) {
      output_spatial_size *= output.shape().Dim(i);
      rhs_spatial_size *= rhs.shape().Dim(i);
    }
  }

  for (size_t i = 0; i < lhs.shape().Dim(0); ++i) {
    for (size_t j = 0; j < output_spatial_size; ++j) {
      // This will be replaced by tensor GetNdIndex function
      int64_t output_dimensions[output.Rank()];
      size_t output_depth = 1;
      for (size_t m = output.Rank() - 1; m > 1; --m) {
        output_dimensions[m] = (j / output_depth) % output.shape().Dim(m);
        output_depth *= output.shape().Dim(m);
      }
      for (size_t k = 0; k < rhs_spatial_size; ++k) {
        for (size_t l = 0; l < lhs.shape().Dim(1); ++l) {
          // This will be replaced by tensor GetNdIndex function
          int64_t filter_spatials[rhs.Rank() - 2];
          size_t depth = 1;
          for (size_t m = rhs.Rank() - 1; m > 1; --m) {
            filter_spatials[m - 2] = (k / depth) % rhs.shape().Dim(m);
            depth *= rhs.shape().Dim(m);
          }

          // This will be replaced by tensor FlattenIndex function
          int64_t lhs_dimensions[lhs.Rank()];
          lhs_dimensions[0] = i;
          lhs_dimensions[1] = l;
          depth = 1;
          size_t lhs_index = 0;
          for (int64_t m = lhs.Rank() - 1; m >= 0; --m) {
            if (m > 1) {
              lhs_dimensions[m] =
                  output_dimensions[m] * op.attributes.window_strides[m - 2] +
                  filter_spatials[m - 2] * op.attributes.rhs_dilation[m - 2];
            }
            lhs_index += lhs_dimensions[m] * depth;
            depth *= lhs.shape().Dim(m);
          }

          size_t channel_skip = l * rhs_spatial_size;

          for (size_t m = 0; m < rhs.shape().Dim(0); ++m) {
            size_t batch_skip = m * rhs_tensor_size;
            output_dimensions[0] = i;
            output_dimensions[1] = output_channel + m;
            output_depth = 1;
            size_t output_index = 0;
            // This will be replaced by tensor FlattenIndex function
            for (int64_t n = output.Rank() - 1; n >= 0; --n) {
              output_index += output_dimensions[n] * output_depth;
              output_depth *= output.shape().Dim(n);
            }
            output_buffer[output_index] +=
                rhs_buffer[batch_skip + channel_skip + k] *
                lhs_buffer[lhs_index];
          }
        }
      }
    }
  }
  output_channel += rhs.shape().Dim(0);

  return absl::OkStatus();
}

template <DataType storage_type>
absl::Status EvaluateImpl(ConvolutionOp& op, const Tensor& lhs,
                          const Tensor& rhs, Tensor& output) {
  size_t output_channel = 0;
  SHLO_REF_RETURN_ON_ERROR(
      ConvolutionImpl<storage_type>(op, output_channel, lhs, rhs, output));
  return absl::OkStatus();
}

ConvolutionOp Create(const ConvolutionOp::Attributes& attributes) {
  return {.attributes = attributes};
}

absl::Status Prepare(ConvolutionOp& op, const Tensor& lhs, const Tensor& rhs,
                     Tensor& output) {
  DISPATCH_INT_FLOAT(PrepareImpl, lhs.StorageType(), op, lhs, rhs, output);
}

absl::Status Evaluate(ConvolutionOp& op, const Tensor& lhs, const Tensor& rhs,
                      Tensor& output) {
  DISPATCH_INT_FLOAT(EvaluateImpl, output.tensor_element_type(), op, lhs, rhs,
                     output);
}
}  // namespace shlo_ref