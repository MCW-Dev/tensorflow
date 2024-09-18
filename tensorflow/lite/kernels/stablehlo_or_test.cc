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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;

class OrOpModel : public SingleOpModel {
 public:
  OrOpModel(const TensorData& input1, const TensorData& input2,
            const TensorData& output) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_STABLEHLO_OR, BuiltinOptions_NONE, 0);
    SetBypassDefaultDelegates();
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

 protected:
  int input1_;
  int input2_;
  int output_;
};

TEST(StablehloElementwise, OrInt32) {
  OrOpModel model({TensorType_INT32, {1, 2, 2, 1}},
                  {TensorType_INT32, {1, 2, 2, 1}}, {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(
      model.input1(), {2147483647, -2147483648, -2147483648, 4294967295});
  model.PopulateTensor<int32_t>(model.input2(),
                                {0, 2147483647, -2147483648, 4294967295});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput<int32_t>(),
              ElementsAre(2147483647, -1, -2147483648, 4294967295));
}

TEST(StablehloElementwise, OrInt8) {
  OrOpModel model({TensorType_INT8, {1, 3, 1}}, {TensorType_INT8, {1, 3, 1}},
                  {TensorType_INT8, {}});
  model.PopulateTensor<int8_t>(model.input1(), {127, -128, -128});
  model.PopulateTensor<int8_t>(model.input2(), {0, 127, -128});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput<int8_t>(), ElementsAre(127, -1, -128));
}

TEST(StablehloElementwise, OrInt16) {
  OrOpModel model({TensorType_INT16, {1, 1, 3}}, {TensorType_INT16, {1, 1, 3}},
                  {TensorType_INT16, {}});
  model.PopulateTensor<int16_t>(model.input1(), {32767, -32768, -32768});
  model.PopulateTensor<int16_t>(model.input2(), {0, 32767, -32768});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput<int16_t>(), ElementsAre(32767, -1, -32768));
}

TEST(StablehloElementwise, OrBool) {
  OrOpModel model({TensorType_BOOL, {2, 1, 2, 1}},
                  {TensorType_BOOL, {2, 1, 2, 1}}, {TensorType_BOOL, {}});
  model.PopulateTensor<bool>(model.input1(), {0, 0, 1, 1});
  model.PopulateTensor<bool>(model.input2(), {0, 1, 0, 1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput<bool>(), ElementsAre(0, 1, 1, 1));
}

}  // namespace
}  // namespace tflite
