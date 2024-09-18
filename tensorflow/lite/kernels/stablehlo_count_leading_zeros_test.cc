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
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using testing::ElementsAreArray;

class CountLeadingZerosOpModel : public SingleOpModel {
 public:
  CountLeadingZerosOpModel(const TensorData& input) {
    input_ = AddInput(input);
    output_ = AddOutput(TensorData(input.type, GetShape(input_)));
    SetBuiltinOp(BuiltinOperator_STABLEHLO_COUNT_LEADING_ZEROS,
                 BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(input_)});
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor<T>(input_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

 protected:
  int input_;
  int output_;
};

TEST(CountLeadingZerosOpTest, Int64Test) {
  CountLeadingZerosOpModel model({TensorType_INT64, {2, 2}});
  model.SetInput<int64_t>({0, 1, 128, -1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput<int64_t>(), ElementsAreArray({64, 63, 56, 0}));
}

TEST(CountLeadingZerosOpTest, Int32Test) {
  CountLeadingZerosOpModel model({TensorType_INT32, {2, 2}});
  model.SetInput<int32_t>({8, 31, 24, 1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput<int32_t>(), ElementsAreArray({28, 27, 27, 31}));
}

TEST(CountLeadingZerosOpTest, Int16Test) {
  CountLeadingZerosOpModel model({TensorType_INT16, {2, 2}});
  model.SetInput<int16_t>({11, 5, 9, 13});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput<int16_t>(), ElementsAreArray({12, 13, 12, 12}));
}

TEST(CountLeadingZerosOpTest, Int8Test) {
  CountLeadingZerosOpModel model({TensorType_INT8, {2, 2}});
  model.SetInput<int8_t>({8, 2, 5, 7});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput<int8_t>(), ElementsAreArray({4, 6, 5, 5}));
}

}  // namespace
}  // namespace tflite
