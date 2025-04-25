/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <math.h>
#include <stdint.h>

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "Eigen/Core"
#include "tensorflow/lite/kernels/internal/test_util.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

template <typename T>
class PowOpModel : public SingleOpModel {
 public:
  PowOpModel(const TensorData& input1, const TensorData& input2,
             const TensorData& output) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_POW, BuiltinOptions_PowOptions,
                 CreatePowOptions(builder_).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input1_;
  int input2_;
  int output_;
};

TEST(PowOpModel, Simple) {
  PowOpModel<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                            {TensorType_INT32, {1, 2, 2, 1}},
                            {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {12, 2, 7, 8});
  model.PopulateTensor<int32_t>(model.input2(), {1, 2, 3, 1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(12, 4, 343, 8));
}

template <typename T>
inline float GetTolerance(float min, float max) {
  float kQuantizedTolerance = (max - min) / (std::numeric_limits<T>::max() -
                                             std::numeric_limits<T>::min());
  if (std::is_same<T, int8_t>::value) {
    kQuantizedTolerance += (max - min) / 256.0f;
  } else if (std::is_same<T, int16_t>::value) {
    kQuantizedTolerance += (max - min) / 512.0f;
  }

  return kQuantizedTolerance;
}

TEST(PowOpModel, NegativeAndZeroValue) {
  PowOpModel<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                            {TensorType_INT32, {1, 2, 2, 1}},
                            {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {0, 2, -7, 8});
  model.PopulateTensor<int32_t>(model.input2(), {1, 2, 3, 0});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 4, -343, 1));
}

TEST(PowOpModel, Float) {
  PowOpModel<float> model({TensorType_FLOAT32, {1, 2, 2, 1}},
                          {TensorType_FLOAT32, {1, 2, 2, 1}},
                          {TensorType_FLOAT32, {}});
  model.PopulateTensor<float>(model.input1(), {0.3, 0.4, 0.7, 5.8});
  model.PopulateTensor<float>(model.input2(), {0.5, 2.7, 3.1, 3.2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {0.5477226, 0.08424846, 0.33098164, 277.313}, 1e-3)));
}

TEST(PowOpModel, NegativeFloatTest) {
  PowOpModel<float> model({TensorType_FLOAT32, {1, 2, 2, 1}},
                          {TensorType_FLOAT32, {1, 2, 2, 1}},
                          {TensorType_FLOAT32, {}});
  model.PopulateTensor<float>(model.input1(), {0.3, 0.4, 0.7, 5.8});
  model.PopulateTensor<float>(model.input2(), {0.5, -2.7, 3.1, -3.2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {0.5477226, 11.869653, 0.33098164, 0.003606}, 1e-3)));
}

TEST(PowOpModel, Float16) {
  PowOpModel<Eigen::half> model({TensorType_FLOAT16, {1, 2, 2, 1}},
                                {TensorType_FLOAT16, {1, 2, 2, 1}},
                                {TensorType_FLOAT16, {}});
  model.PopulateTensor<Eigen::half>(
      model.input1(),
      {Eigen::half(0.3), Eigen::half(0.4), Eigen::half(0.7), Eigen::half(5.8)});
  model.PopulateTensor<Eigen::half>(
      model.input2(),
      {Eigen::half(0.5), Eigen::half(2.7), Eigen::half(3.1), Eigen::half(3.2)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  ASSERT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {Eigen::half(0.5477226), Eigen::half(0.08424846),
                   Eigen::half(0.33098164), Eigen::half(277.313)},
                  1e0)));
}

TEST(PowOpModel, NegativeFloat16Test) {
  PowOpModel<Eigen::half> model({TensorType_FLOAT16, {1, 2, 2, 1}},
                                {TensorType_FLOAT16, {1, 2, 2, 1}},
                                {TensorType_FLOAT16, {}});
  model.PopulateTensor<Eigen::half>(
      model.input1(),
      {Eigen::half(0.3), Eigen::half(0.4), Eigen::half(0.7), Eigen::half(5.8)});
  model.PopulateTensor<Eigen::half>(
      model.input2(), {Eigen::half(0.5), Eigen::half(-2.7), Eigen::half(3.1),
                       Eigen::half(-3.2)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  ASSERT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {Eigen::half(0.5477226), Eigen::half(11.869653),
                   Eigen::half(0.33098164), Eigen::half(0.003606)},
                  1e-3)));
}

TEST(PowOpModel, BFloat16) {
  PowOpModel<Eigen::bfloat16> model({TensorType_BFLOAT16, {1, 2, 2, 1}},
                                    {TensorType_BFLOAT16, {1, 2, 2, 1}},
                                    {TensorType_BFLOAT16, {}});
  model.PopulateTensor<Eigen::bfloat16>(
      model.input1(), {Eigen::bfloat16(0.3), Eigen::bfloat16(0.4),
                       Eigen::bfloat16(0.7), Eigen::bfloat16(5.8)});
  model.PopulateTensor<Eigen::bfloat16>(
      model.input2(), {Eigen::bfloat16(0.5), Eigen::bfloat16(2.7),
                       Eigen::bfloat16(3.1), Eigen::bfloat16(3.1)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  ASSERT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {Eigen::bfloat16(0.5477226), Eigen::bfloat16(0.08424846),
                   Eigen::bfloat16(0.33098164), Eigen::bfloat16(232.609)},
                  1e0)));
}

TEST(PowOpModel, NegativeBFloat16Test) {
  PowOpModel<Eigen::bfloat16> model({TensorType_BFLOAT16, {1, 2, 2, 1}},
                                    {TensorType_BFLOAT16, {1, 2, 2, 1}},
                                    {TensorType_BFLOAT16, {}});
  model.PopulateTensor<Eigen::bfloat16>(
      model.input1(), {Eigen::bfloat16(0.3), Eigen::bfloat16(0.4),
                       Eigen::bfloat16(0.7), Eigen::bfloat16(5.8)});
  model.PopulateTensor<Eigen::bfloat16>(
      model.input2(), {Eigen::bfloat16(0.5), Eigen::bfloat16(-2.7),
                       Eigen::bfloat16(3.1), Eigen::bfloat16(-3.2)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  ASSERT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {Eigen::bfloat16(0.5477226), Eigen::bfloat16(11.869653),
                   Eigen::bfloat16(0.33098164), Eigen::bfloat16(0.003606)},
                  1e-3)));
}

TEST(PowOpModel, BroadcastTest) {
  PowOpModel<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                            {TensorType_INT32, {1}}, {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {12, 2, 7, 8});
  model.PopulateTensor<int32_t>(model.input2(), {4});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(20736, 16, 2401, 4096));
}

TEST(PowOpModel, BroadcastFloatTest) {
  PowOpModel<float> model({TensorType_FLOAT32, {1, 2, 2, 1}},
                          {TensorType_FLOAT32, {1}}, {TensorType_FLOAT32, {}});
  model.PopulateTensor<float>(model.input1(), {12, 2, 7, 8});
  model.PopulateTensor<float>(model.input2(), {4});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(20736, 16, 2401, 4096));
}

TEST(PowOpModel, BroadcastFloat16Test) {
  PowOpModel<Eigen::half> model({TensorType_FLOAT16, {1, 2, 2, 1}},
                                {TensorType_FLOAT16, {1}},
                                {TensorType_FLOAT16, {}});
  model.PopulateTensor<Eigen::half>(
      model.input1(),
      {Eigen::half(12), Eigen::half(2), Eigen::half(7), Eigen::half(8)});
  model.PopulateTensor<Eigen::half>(model.input2(), {Eigen::half(4)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  ASSERT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(),
              ElementsAre(Eigen::half(20736), Eigen::half(16),
                          Eigen::half(2401), Eigen::half(4096)));
}

TEST(PowOpModel, BroadcastBFloat16Test) {
  PowOpModel<Eigen::bfloat16> model({TensorType_BFLOAT16, {1, 2, 2, 1}},
                                    {TensorType_BFLOAT16, {1}},
                                    {TensorType_BFLOAT16, {}});
  model.PopulateTensor<Eigen::bfloat16>(
      model.input1(), {Eigen::bfloat16(12), Eigen::bfloat16(2),
                       Eigen::bfloat16(7), Eigen::bfloat16(8)});
  model.PopulateTensor<Eigen::bfloat16>(model.input2(), {Eigen::bfloat16(4)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  ASSERT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(),
              ElementsAre(Eigen::bfloat16(20736), Eigen::bfloat16(16),
                          Eigen::bfloat16(2401), Eigen::bfloat16(4096)));
}

template <typename T>
void CalculateTrueResults(const std::vector<T>& input_data, T exponent,
                          int flat_size, std::vector<T>* output_data) {
  for (int i = 0; i < flat_size; ++i) {
    output_data->at(i) = static_cast<T>(std::pow(input_data[i], exponent));
  }
}

TEST(PowOpModel, FloatSingleIntegerExponentTest) {
  PowOpModel<float> model({TensorType_FLOAT32, {1, 2, 2, 1}},
                          {TensorType_FLOAT32, {1}}, {TensorType_FLOAT32, {}});
  const int input_size = 1 * 2 * 2 * 1;
  for (int i = 1; i < 20; ++i) {
    std::vector<float> input_data(input_size);
    for (int index = 0; index < input_size; ++index) {
      // For exponent is float case, if base < 0, we will result in nan, so
      // we only populate positive base.
      input_data[index] = UniformRandomFloat(0, 1.5);
    }
    model.PopulateTensor<float>(model.input1(), input_data);
    float exponent = static_cast<float>(i);
    // Random deviate exponent, e.g., 1.99999 or 2.00001.
    exponent += UniformRandomInt(-1, 1) * 1e-5;
    model.PopulateTensor<float>(model.input2(), {exponent});
    ASSERT_EQ(model.Invoke(), kTfLiteOk);
    EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
    std::vector<float> output_data(input_size);
    CalculateTrueResults(input_data, exponent, input_size, &output_data);
    EXPECT_THAT(model.GetOutput(),
                ElementsAreArray(ArrayFloatNear(output_data, 1e-2)));
  }
}

TEST(PowOpModel, IntSingleIntegerExponentTest) {
  PowOpModel<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                            {TensorType_INT32, {1}}, {TensorType_INT32, {}});
  const int input_size = 1 * 2 * 2 * 1;
  for (int i = 1; i < 20; ++i) {
    std::vector<int32_t> input_data(input_size);
    for (int index = 0; index < input_size; ++index) {
      input_data[index] = UniformRandomInt(-2, -2);
    }
    model.PopulateTensor<int32_t>(model.input1(), input_data);
    int exponent = i;
    model.PopulateTensor<int32_t>(model.input2(), {exponent});
    ASSERT_EQ(model.Invoke(), kTfLiteOk);
    EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
    std::vector<int32_t> output_data(input_size);
    CalculateTrueResults(input_data, exponent, input_size, &output_data);
    EXPECT_THAT(model.GetOutput(), ElementsAreArray(output_data));
  }
}

template <typename T>
class QuantizedPowOpModel : public PowOpModel<T> {
 public:
  QuantizedPowOpModel(const TensorData& input1, const TensorData& input2,
                      const TensorData& output)
      : PowOpModel<T>(input1, input2, output) {}
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(this->template ExtractVector<T>(this->output_), 
                         this->GetScale(this->output_), 
                         this->GetZeroPoint(this->output_));
  }
};

TEST(QuantizedPowOpModel, SimpleQuantizedInt16) {
  QuantizedPowOpModel<int16_t> model({TensorType_INT16, {1, 2, 2, 1}, -142, 142},
                            {TensorType_INT16, {1, 2, 2, 1}, -142, 142},
                            {TensorType_INT16, {}, -142, 142});
  const float kQuantizedTolerance = GetTolerance<int16_t>(-142.f,142.f);
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> exponent = {3.0f, 2.0f, 2.0f, 2.0f};
  model.QuantizeAndPopulate<int16_t>(model.input1(), input_data);
  model.QuantizeAndPopulate<int16_t>(model.input2(), exponent);
  std::vector<float> expected_data(4);
  for (int i = 0; i < input_data.size(); ++i) {
    expected_data[i] = std::pow(input_data[i], exponent[i]);
  }
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  ASSERT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear(expected_data, kQuantizedTolerance)));
}

TEST(QuantizedPowOpModel, SimpleQuantizedInt8) {
  QuantizedPowOpModel<int8_t> model({TensorType_INT8, {1, 2, 2, 1}, -127, 127},
                            {TensorType_INT8, {1, 2, 2, 1}, -127, 127},
                            {TensorType_INT8, {}, -127, 127});
  const float kQuantizedTolerance = GetTolerance<int8_t>(-127.f,127.f);
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> exponent = {3.0f, 2.0f, 2.0f, 2.0f};
  model.QuantizeAndPopulate<int8_t>(model.input1(), input_data);
  model.QuantizeAndPopulate<int8_t>(model.input2(), exponent);
  std::vector<float> expected_data(4);
  for (int i = 0; i < input_data.size(); ++i) {
    expected_data[i] = std::pow(input_data[i], exponent[i]);
  }
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  ASSERT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear(expected_data, kQuantizedTolerance)));
}

TEST(QuantizedPowOpModel, BroadcastQuantizedInt16) {
  QuantizedPowOpModel<int16_t> model({TensorType_INT16, {1, 2, 2, 1}, -142, 142},
                            {TensorType_INT16, {1}, -142, 142},
                            {TensorType_INT16, {}, -142, 142});
  const float kQuantizedTolerance = GetTolerance<int16_t>(-142.f,142.f);
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  float exponent = {3.0f};
  model.QuantizeAndPopulate<int16_t>(model.input1(), input_data);
  model.QuantizeAndPopulate<int16_t>(model.input2(), {exponent});
  std::vector<float> expected_data(4);
  for (int i = 0; i < input_data.size(); ++i) {
    expected_data[i] = std::pow(input_data[i], exponent);
  }
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  ASSERT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear(expected_data, kQuantizedTolerance)));
}

TEST(QuantizedPowOpModel, BroadcastQuantizedInt8) {
  QuantizedPowOpModel<int8_t> model({TensorType_INT8, {1, 2, 2, 1}, -127, 127},
                            {TensorType_INT8, {1}, -127, 127},
                            {TensorType_INT8, {}, -127, 127});
  const float kQuantizedTolerance = GetTolerance<int8_t>(-127.f,127.f);
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  float exponent = {2.0f};
  model.QuantizeAndPopulate<int8_t>(model.input1(), input_data);
  model.QuantizeAndPopulate<int8_t>(model.input2(), {exponent});
  std::vector<float> expected_data(4);
  for (int i = 0; i < input_data.size(); ++i) {
    expected_data[i] = std::pow(input_data[i], exponent);
  }
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  ASSERT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear(expected_data, kQuantizedTolerance)));
}

}  // namespace
}  // namespace tflite
