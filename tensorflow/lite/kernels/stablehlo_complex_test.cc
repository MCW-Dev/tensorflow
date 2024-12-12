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
#include <initializer_list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using testing::ElementsAreArray;

class ComplexOpModel : public SingleOpModel {
 public:
  ComplexOpModel(const TensorData& input1, const TensorData& input2,
                 const TensorData& output) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);

    SetBuiltinOp(BuiltinOperator_STABLEHLO_COMPLEX, BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  template <typename T>
  void SetInputs(std::initializer_list<T> data_input1,
                 std::initializer_list<T> data_input2) {
    PopulateTensor<T>(input1_, data_input1);
    PopulateTensor<T>(input2_, data_input2);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

 protected:
  int input1_;
  int input2_;
  int output_;
};

TEST(ComplexOpTest, Float64Test) {
  ComplexOpModel model({TensorType_FLOAT64, {2}}, {TensorType_FLOAT64, {2}},
                       {TensorType_COMPLEX128, {2}});
  model.SetInputs<_Float64>({1.0, 3.0}, {2.0, 4.0});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<TfLiteComplex64> expected_values = {{1.0, 2.0}, {3.0, 4.0}};
  EXPECT_THAT(model.GetOutput<TfLiteComplex64>(),
              ElementsAreArray(expected_values));
}

}  // namespace
}  // namespace tflite
