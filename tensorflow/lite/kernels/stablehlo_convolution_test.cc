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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <vector>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_convolution {
namespace {

using testing::ElementsAre;
using testing::ElementsAreArray;
using testing::FloatNear;
using testing::Pointwise;

class StablehloConvolutionOpModel : public SingleOpModel {
 public:
  StablehloConvolutionOpModel(const TensorData& lhs, const TensorData& rhs,
                              const TfLiteStablehloConvolutionParams& params) {
    lhs_ = AddInput(lhs);
    rhs_ = AddInput(rhs);
    output_ = AddOutput(TensorData(lhs.type));
    SetBuiltinOp(
        BuiltinOperator_STABLEHLO_CONVOLUTION,
        BuiltinOptions2_StablehloConvolutionOptions,
        CreateStablehloConvolutionOptions(
            builder_,
            builder_.CreateVector(
                std::vector(params.window_strides,
                            params.window_strides + params.num_window_strides)),
            builder_.CreateVector(std::vector(
                params.padding, params.padding + params.num_padding)),
            builder_.CreateVector(
                std::vector(params.lhs_dilation,
                            params.lhs_dilation + params.num_lhs_dilation)),
            builder_.CreateVector(
                std::vector(params.rhs_dilation,
                            params.rhs_dilation + params.num_rhs_dilation)),
            builder_.CreateVector(std::vector({false})),
            params.input_batch_dimension, params.input_feature_dimension,
            builder_.CreateVector(
                std::vector(params.input_spatial_dimensions,
                            params.input_spatial_dimensions +
                                params.num_input_spatial_dimensions)),
            params.kernel_input_feature_dimension,
            params.kernel_output_feature_dimension,
            builder_.CreateVector(
                std::vector(params.kernel_spatial_dimensions,
                            params.kernel_spatial_dimensions +
                                params.num_kernel_spatial_dimensions)),
            params.output_batch_dimension, params.output_feature_dimension,
            builder_.CreateVector(
                std::vector(params.output_spatial_dimensions,
                            params.output_spatial_dimensions +
                                params.num_output_spatial_dimensions)),
            params.feature_group_count, params.batch_group_count,
            builder_.CreateVector(std::vector(params.precision_config,
                                              params.precision_config + 2)))
            .Union());
    BuildInterpreter({GetShape(lhs_), GetShape(rhs_)});
  }

  template <typename T>
  void SetLhs(std::initializer_list<T> data) {
    PopulateTensor<T>(lhs_, data);
  }

  template <typename T>
  void SetRhs(std::initializer_list<T> data) {
    PopulateTensor<T>(rhs_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

 protected:
  int lhs_;
  int rhs_;
  int output_;
};

TEST(StablehloConvolutionOpTest, ConvolutionFloat32) {
  TfLiteStablehloConvolutionParams params = {
      {1, 1},        // window_strides
      2,             // num_window_strides
      {0, 0, 0, 0},  // padding
      4,             // num_padding
      {1, 1},        // lhs_dilation
      2,             // num_lhs_dilation
      {1, 1},        // rhs_dilation
      2,             // num_rhs_dilation
      0,             // input_batch_dimension
      1,             // input_feature_dimension
      {2, 3},        // input_spatial_dimensions
      2,             // num_input_spatial_dimensions
      1,             // kernel_input_feature_dimension
      0,             // kernel_output_feature_dimension
      {2, 3},        // kernel_spatial_dimenstions
      2,             // nun_kernel_spatial_dimensions
      0,             // output_batch_dimension
      1,             // output_feature_dimension
      {2, 3},        // output_spatial_dimensions
      2,             // num_output_spatial_dimensions
      1,             // feature_group_count
      1,             // batch_group_count
      {tflite::StablehloPrecisionConfig::StablehloPrecisionConfig_DEFAULT,
       tflite::StablehloPrecisionConfig::
           StablehloPrecisionConfig_DEFAULT},  // precision config
  };
  StablehloConvolutionOpModel model({TensorType_FLOAT32, {1, 1, 2, 2}},
                                    {TensorType_FLOAT32, {1, 1, 1, 1}}, params);
  model.SetLhs<float>({1.16, 2.43, 3.81, 4.77});
  model.SetRhs<float>({2.21});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<float> expected_values = {2.5636, 5.3703, 8.4201, 10.5417};
  EXPECT_THAT(model.GetOutput<float>(),
              Pointwise(FloatNear(1e-5), expected_values));
}

TEST(StablehloConvolutionOpTest, ConvolutionBFloat16) {
  TfLiteStablehloConvolutionParams params = {
      {1, 1},        // window_strides
      2,             // num_window_strides
      {0, 0, 0, 0},  // padding
      4,             // num_padding
      {1, 1},        // lhs_dilation
      2,             // num_lhs_dilation
      {1, 1},        // rhs_dilation
      2,             // num_rhs_dilation
      0,             // input_batch_dimension
      1,             // input_feature_dimension
      {2, 3},        // input_spatial_dimensions
      2,             // num_input_spatial_dimensions
      1,             // kernel_input_feature_dimension
      0,             // kernel_output_feature_dimension
      {2, 3},        // kernel_spatial_dimenstions
      2,             // nun_kernel_spatial_dimensions
      0,             // output_batch_dimension
      1,             // output_feature_dimension
      {2, 3},        // output_spatial_dimensions
      2,             // num_output_spatial_dimensions
      1,             // feature_group_count
      1,             // batch_group_count
      {tflite::StablehloPrecisionConfig::StablehloPrecisionConfig_DEFAULT,
       tflite::StablehloPrecisionConfig::
           StablehloPrecisionConfig_DEFAULT},  // precision config
  };
  StablehloConvolutionOpModel model({TensorType_BFLOAT16, {1, 1, 2, 2}},
                                    {TensorType_BFLOAT16, {1, 1, 1, 1}},
                                    params);
  std::initializer_list<Eigen::bfloat16> lhs_data{
      Eigen::bfloat16(1.16), Eigen::bfloat16(2.43), Eigen::bfloat16(3.81),
      Eigen::bfloat16(4.77)};
  std::initializer_list<Eigen::bfloat16> rhs_data{Eigen::bfloat16(2.21)};
  model.SetLhs<Eigen::bfloat16>(lhs_data);
  model.SetRhs<Eigen::bfloat16>(rhs_data);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::initializer_list<Eigen::bfloat16> expected_values = {
      Eigen::bfloat16(2.54688), Eigen::bfloat16(5.375), Eigen::bfloat16(8.375),
      Eigen::bfloat16(10.5625)};
  EXPECT_THAT(model.GetOutput<Eigen::bfloat16>(),
              Pointwise(FloatNear(1e-5), expected_values));
}

TEST(StablehloConvolutionOpTest, ConvolutionFloat16) {
  TfLiteStablehloConvolutionParams params = {
      {1, 1},        // window_strides
      2,             // num_window_strides
      {0, 0, 0, 0},  // padding
      4,             // num_padding
      {1, 1},        // lhs_dilation
      2,             // num_lhs_dilation
      {1, 1},        // rhs_dilation
      2,             // num_rhs_dilation
      0,             // input_batch_dimension
      1,             // input_feature_dimension
      {2, 3},        // input_spatial_dimensions
      2,             // num_input_spatial_dimensions
      1,             // kernel_input_feature_dimension
      0,             // kernel_output_feature_dimension
      {2, 3},        // kernel_spatial_dimenstions
      2,             // nun_kernel_spatial_dimensions
      0,             // output_batch_dimension
      1,             // output_feature_dimension
      {2, 3},        // output_spatial_dimensions
      2,             // num_output_spatial_dimensions
      1,             // feature_group_count
      1,             // batch_group_count
      {tflite::StablehloPrecisionConfig::StablehloPrecisionConfig_DEFAULT,
       tflite::StablehloPrecisionConfig::
           StablehloPrecisionConfig_DEFAULT},  // precision config
  };
  StablehloConvolutionOpModel model({TensorType_FLOAT16, {1, 1, 2, 2}},
                                    {TensorType_FLOAT16, {1, 1, 1, 1}}, params);
  std::initializer_list<Eigen::half> lhs_data{
      Eigen::half(1.16), Eigen::half(2.43), Eigen::half(3.81),
      Eigen::half(4.77)};
  std::initializer_list<Eigen::half> rhs_data{Eigen::half(2.21)};
  model.SetLhs<Eigen::half>(lhs_data);
  model.SetRhs<Eigen::half>(rhs_data);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::initializer_list<Eigen::half> expected_values = {
      Eigen::half(2.56445), Eigen::half(5.37109), Eigen::half(8.42188),
      Eigen::half(10.5469)};
  EXPECT_THAT(model.GetOutput<Eigen::half>(),
              Pointwise(FloatNear(1e-5), expected_values));
}

TEST(StablehloConvolutionOpTest, ConvolutionInt64) {
  TfLiteStablehloConvolutionParams params = {
      {1},     // window_strides
      1,       // num_window_strides
      {0, 0},  // padding
      2,       // num_padding
      {1},     // lhs_dilation
      1,       // num_lhs_dilation
      {1},     // rhs_dilation
      1,       // num_rhs_dilation
      0,       // input_batch_dimension
      1,       // input_feature_dimension
      {2},     // input_spatial_dimensions
      1,       // num_input_spatial_dimensions
      1,       // kernel_input_feature_dimension
      0,       // kernel_output_feature_dimension
      {2},     // kernel_spatial_dimenstions
      1,       // nun_kernel_spatial_dimensions
      0,       // output_batch_dimension
      1,       // output_feature_dimension
      {2},     // output_spatial_dimensions
      1,       // num_output_spatial_dimensions
      1,       // feature_group_count
      1,       // batch_group_count
      {tflite::StablehloPrecisionConfig::StablehloPrecisionConfig_DEFAULT,
       tflite::StablehloPrecisionConfig::
           StablehloPrecisionConfig_DEFAULT},  // precision config
  };
  StablehloConvolutionOpModel model({TensorType_INT64, {1, 1, 10}},
                                    {TensorType_INT64, {1, 1, 1}}, params);
  model.SetLhs<int64_t>({1, 2, 3, 4, 5, 6, 7, 9, 4, 2});
  model.SetRhs<int64_t>({5});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<int64_t> expected_values = {5,  10, 15, 20, 25,
                                          30, 35, 45, 20, 10};
  EXPECT_THAT(model.GetOutput<int64_t>(), ElementsAreArray(expected_values));
}

}  // namespace
}  // namespace stablehlo_convolution
}  // namespace builtin
}  // namespace ops
}  // namespace tflite
