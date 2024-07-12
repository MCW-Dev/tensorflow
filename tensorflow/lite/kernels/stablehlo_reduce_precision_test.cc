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
#include <gtest/gtest.h>

#include <initializer_list>
#include <vector>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using testing::ElementsAreArray;

class ReducePrecisionOpModel : public SingleOpModel {
 public:
  ReducePrecisionOpModel(const TensorData& input1,
                         const TfLiteStablehloReducePrecisionParams& params) {
    input1_ = AddInput(input1);
    output_ = AddOutput(TensorData(input1.type, GetShape(input1_)));
    SetBuiltinOp(BuiltinOperator_STABLEHLO_REDUCE_PRECISION,
                 BuiltinOptions2_StablehloReducePrecisionOptions,
                 CreateStablehloReducePrecisionOptions(
                     builder_, params.exponent_bits, params.mantissa_bits)
                     .Union());
    BuildInterpreter({GetShape(input1_)});
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor<T>(input1_, data);
  }
  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

 protected:
  int input1_;
  int output_;
};

TEST(StablehloReducePrecisionOpTest, ReducePrecisionFloat32) {
  TfLiteStablehloReducePrecisionParams params = {5, 10};

  ReducePrecisionOpModel model({TensorType_FLOAT32, {1}}, params);
  model.SetInput<float>({2.76674843});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<float> expected_values = {2.76757813};
  EXPECT_THAT(model.GetOutput<float>(), ElementsAreArray(expected_values));
}

TEST(StablehloReducePrecisionOpTest, ReducePrecisionFloat32VariedShape) {
  TfLiteStablehloReducePrecisionParams params = {5, 10};

  ReducePrecisionOpModel model({TensorType_FLOAT32, {5, 7}}, params);
  model.SetInput<float>(
      {-0.993861913, -0.716096282, -1.94070697,  0.474192441,  4.20493698,
       -3.38537836,  -3.6731472,   3.16929746,   -1.36569667,  4.73274279,
       -0.438586295, -0.517679751, 7.06460381,   -2.35960102,  -0.523919463,
       1.576170e+00, -1.32587886,  -2.21371269,  -0.403554231, 0.365458697,
       0.996900379,  -2.89147782,  -4.60086203,  1.01571786,   0.791271567,
       5.432890e-01, -2.61252475,  2.32681966,   -3.13387656,  -0.554039478,
       -3.36884379,  -0.862945318, -0.893986999, 1.75586152,   6.04183912});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<float> expected_values = {
      -0.993652343, -0.716308593, -1.94042969,  0.474121094,  4.203125,
      -3.38476563,  -3.67382813,  3.16992188,   -1.36523438,  4.734375,
      -0.438476563, -0.517578125, 7.06640625,   -2.359375,    -0.523925781,
      1.57617188,   -1.32617188,  -2.21289063,  -0.403564453, 0.365478516,
      0.997070312,  -2.890625,    -4.6015625,   1.015625,     0.791503906,
      0.543457031,  -2.61328125,  2.32617188,   -3.13476563,  -0.554199219,
      -3.36914063,  -0.862792968, -0.894042968, 1.75585938,   6.04296875};
  EXPECT_THAT(model.GetOutput<float>(), ElementsAreArray(expected_values));
}

TEST(StablehloReducePrecisionOpTest, ReducePrecisionFloat16) {
  TfLiteStablehloReducePrecisionParams params = {8, 7};
  ReducePrecisionOpModel model({TensorType_FLOAT16, {1}}, params);
  std::initializer_list<Eigen::half> half{Eigen::half{-1.836910e+00f}};
  model.SetInput<Eigen::half>(half);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<float> expected_values = {Eigen::half(-1.835940e+00)};
  EXPECT_THAT(model.GetOutput<Eigen::half>(),
              ElementsAreArray(expected_values));
}

TEST(StablehloReducePrecisionOpTest, ReducePrecisionBfloat16) {
  TfLiteStablehloReducePrecisionParams params = {8, 23};
  ReducePrecisionOpModel model({TensorType_BFLOAT16, {1}}, params);
  std::initializer_list<Eigen::bfloat16> input{Eigen::bfloat16(6.875000e-01)};
  model.SetInput<Eigen::bfloat16>(input);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<float> expected_values = {Eigen::bfloat16(6.875000e-01)};
  EXPECT_THAT(model.GetOutput<Eigen::bfloat16>(),
              ElementsAreArray(expected_values));
}

}  // namespace
}  // namespace tflite