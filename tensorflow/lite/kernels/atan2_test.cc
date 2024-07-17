// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <cmath>

#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

template <typename T>
tflite::TensorType GetTTEnum();
template <>
tflite::TensorType GetTTEnum<float>() {
  return tflite::TensorType_FLOAT32;
}

template <>
tflite::TensorType GetTTEnum<double>() {
  return tflite::TensorType_FLOAT64;
}

template <>
tflite::TensorType GetTTEnum<Eigen::half>() {
  return tflite::TensorType_FLOAT16;
}

template <>
tflite::TensorType GetTTEnum<Eigen::bfloat16>() {
  return tflite::TensorType_BFLOAT16;
}

class Atan2Model : public tflite::SingleOpModel {
 public:
  Atan2Model(tflite::TensorData y, tflite::TensorData x,
             tflite::TensorData output) {
    y_ = AddInput(y);
    x_ = AddInput(x);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_ATAN2, BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(y_), GetShape(x_)});
  }

  template <typename T>
  std::vector<T> GetOutput(const std::vector<T>& y, const std::vector<T>& x) {
    PopulateTensor<T>(y_, y);
    PopulateTensor<T>(x_, x);
    Invoke();
    return ExtractVector<T>(output_);
  }

 private:
  int y_;
  int x_;
  int output_;
};

template <>
std::vector<Eigen::half> Atan2Model::GetOutput(
    const std::vector<Eigen::half>& y, const std::vector<Eigen::half>& x) {
  PopulateTensor<Eigen::half>(y_, y);
  PopulateTensor<Eigen::half>(x_, x);
  Invoke();
  return ExtractVector<Eigen::half>(output_);
}

template <>
std::vector<Eigen::bfloat16> Atan2Model::GetOutput(
    const std::vector<Eigen::bfloat16>& y,
    const std::vector<Eigen::bfloat16>& x) {
  PopulateTensor<Eigen::bfloat16>(y_, y);
  PopulateTensor<Eigen::bfloat16>(x_, x);
  Invoke();
  return ExtractVector<Eigen::bfloat16>(output_);
}

template <typename Float>
class Atan2Test : public ::testing::Test {
 public:
  using FloatType = Float;
};

using TestTypes = ::testing::Types<float, double, Eigen::half, Eigen::bfloat16>;

TEST(Atan2_test, Atan2_testWorks) {
  Atan2Model model({TensorType_FLOAT16, {1, 2, 3}},
                   {TensorType_FLOAT16, {1, 2, 3}},
                   {TensorType_FLOAT16, {1, 2, 3}});

  std::vector<Eigen::half> y_data = {
      Eigen::half(2.955080e+00),  Eigen::half(2.557370e-02),
      Eigen::half(-3.945310e+00), Eigen::half(1.383790e+00),
      Eigen::half(-5.034180e-01), Eigen::half(1.483400e+00)};
  std::vector<Eigen::half> x_data = {
      Eigen::half(2.291020e+00),  Eigen::half(-9.304680e+00),
      Eigen::half(-7.431640e-01), Eigen::half(-5.268550e-01),
      Eigen::half(-1.673830e+00), Eigen::half(-2.681640e+00)};

  auto got = model.GetOutput<Eigen::half>(y_data, x_data);

  ASSERT_EQ(got.size(), 6);
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(got[i], Eigen::half(std::atan2((y_data[i]), (x_data[i]))));
  }
}

TEST(Atan2_Test, Atan2_testWorked) {
  Atan2Model model({TensorType_BFLOAT16, {1, 2, 3}},
                   {TensorType_BFLOAT16, {1, 2, 3}},
                   {TensorType_BFLOAT16, {1, 2, 3}});

  std::vector<Eigen::bfloat16> y_data = {
      Eigen::bfloat16(2.250000e+00),  Eigen::bfloat16(1.171880e+00),
      Eigen::bfloat16(-2.812500e+00), Eigen::bfloat16(2.265630e+00),
      Eigen::bfloat16(4.628910e-01),  Eigen::bfloat16(-5.590820e-02)};
  std::vector<Eigen::bfloat16> x_data = {
      Eigen::bfloat16(-2.140630e+00), Eigen::bfloat16(2.609380e+00),
      Eigen::bfloat16(-1.875000e+00), Eigen::bfloat16(3.222660e-01),
      Eigen::bfloat16(-3.312500e+00), Eigen::bfloat16(-3.281250e+00)};

  auto got = model.GetOutput<Eigen::bfloat16>(y_data, x_data);

  ASSERT_EQ(got.size(), 6);
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(got[i],
                    Eigen::bfloat16(std::atan2((y_data[i]), (x_data[i]))));
  }
}

TYPED_TEST_SUITE(Atan2Test, TestTypes);

TYPED_TEST(Atan2Test, TestScalar) {
  using Float = typename TestFixture::FloatType;
  tflite::TensorData y = {GetTTEnum<Float>(), {}};
  tflite::TensorData x = {GetTTEnum<Float>(), {}};
  tflite::TensorData output = {GetTTEnum<Float>(), {}};
  Atan2Model m(y, x, output);

  auto got = m.GetOutput<Float>({Float(0.0)}, {Float(0.0)});
  ASSERT_EQ(got.size(), 1);
  EXPECT_FLOAT_EQ(got[0], 0.0);

  ASSERT_FLOAT_EQ(m.GetOutput<Float>({Float(1.0)}, {Float(0.0)})[0],
                  Float(M_PI / 2));
  ASSERT_FLOAT_EQ(m.GetOutput<Float>({Float(0.0)}, {Float(1.0)})[0],
                  Float(0.0));
  ASSERT_FLOAT_EQ(m.GetOutput<Float>({Float(-1.0)}, {Float(0.0)})[0],
                  Float(-M_PI / 2));
}

TYPED_TEST(Atan2Test, TestBatch) {
  using Float = typename TestFixture::FloatType;
  tflite::TensorData y = {GetTTEnum<Float>(), {4, 2, 1}};
  tflite::TensorData x = {GetTTEnum<Float>(), {4, 2, 1}};
  tflite::TensorData output = {GetTTEnum<Float>(), {4, 2, 1}};
  Atan2Model m(y, x, output);

  std::vector<Float> y_data = {
      Float(0.132423),  Float(0.246563),  Float(0.345357), Float(0.4345345),
      Float(0.5345345), Float(0.6123243), Float(0.77546),  Float(0.843345)};
  std::vector<Float> x_data = {
      Float(0.8324234), Float(0.7643534), Float(0.6635434), Float(0.5876867),
      Float(0.4345345), Float(0.32432),   Float(0.234323),  Float(0.123422)};

  auto got = m.GetOutput<Float>(y_data, x_data);
  ASSERT_EQ(got.size(), 8);
  for (int i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(got[i], Float(std::atan2(y_data[i], x_data[i])));
  }
}

}  // namespace
}  // namespace tflite 
