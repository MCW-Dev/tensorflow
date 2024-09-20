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
#include <memory>
#include <vector>

#include "Eigen/Core"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

template <typename T>
tflite::TensorType GetTTEnum();

template <>
tflite::TensorType GetTTEnum<Eigen::half>() {
  return tflite::TensorType_FLOAT16;
}

template <>
tflite::TensorType GetTTEnum<Eigen::bfloat16>() {
  return tflite::TensorType_BFLOAT16;
}

template <>
tflite::TensorType GetTTEnum<float>() {
  return tflite::TensorType_FLOAT32;
}

template <>
tflite::TensorType GetTTEnum<int8_t>() {
  return tflite::TensorType_INT8;
}

template <>
tflite::TensorType GetTTEnum<int16_t>() {
  return tflite::TensorType_INT16;
}

using ::testing::ElementsAreArray;

class StablehloBatchNormTrainingOpModel : public SingleOpModel {
 public:
  StablehloBatchNormTrainingOpModel(
      const TensorData& input, const TensorData& scale,
      const TensorData& offset, const TensorData& output,
      const TensorData& batch_mean, const TensorData& batch_var,
      const TfLiteStablehloBatchNormTrainingParams& params) {
    input_ = AddInput(SymmetricInt16Scaling(input));
    scale_ = AddInput(SymmetricInt16Scaling(scale));
    offset_ = AddInput(SymmetricInt16Scaling(offset));
    output_ = AddOutput(SymmetricInt16Scaling(output));
    batch_mean_ = AddOutput(SymmetricInt16Scaling(batch_mean));
    batch_var_ = AddOutput(SymmetricInt16Scaling(batch_var));
    SetBuiltinOp(BuiltinOperator_STABLEHLO_BATCH_NORM_TRAINING,
                 BuiltinOptions2_StablehloBatchNormTrainingOptions,
                 CreateStablehloBatchNormTrainingOptions(
                     builder_, params.epsilon, params.feature_index)
                     .Union());
    BuildInterpreter({GetShape(input_), GetShape(scale_), GetShape(offset_)},
                     /*num_threads=*/-1, /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/false, /*allocate_and_delegate=*/false,
                     /*use_simple_allocator=*/false);

    AllocateAndDelegate(true);
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor<T>(input_, data);
  }

  template <typename T>
  void SetQInput(std::initializer_list<T> data) {
    QuantizeAndPopulate<T>(input_, data);
  }

  template <typename T>
  void SetScale(std::initializer_list<T> data) {
    PopulateTensor<T>(scale_, data);
  }

  template <typename T>
  void SetQScale(std::initializer_list<T> data) {
    QuantizeAndPopulate<T>(scale_, data);
  }

  template <typename T>
  void SetOffset(std::initializer_list<T> data) {
    PopulateTensor<T>(offset_, data);
  }

  template <typename T>
  void SetQOffset(std::initializer_list<T> data) {
    QuantizeAndPopulate<T>(offset_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  template <typename T>
  std::vector<T> GetBatchMean() {
    return ExtractVector<T>(batch_mean_);
  }

  template <typename T>
  std::vector<T> GetBatchVar() {
    return ExtractVector<T>(batch_var_);
  }

  template <typename QuantizedType>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<QuantizedType>(
        this->template ExtractVector<QuantizedType>(this->output_),
        GetScale(this->output_), GetZeroPoint(this->output_));
  }

  template <typename QuantizedType>
  std::vector<float> GetDequantizedBatchMean() {
    return Dequantize<QuantizedType>(
        this->template ExtractVector<QuantizedType>(this->batch_mean_),
        GetScale(this->batch_mean_), GetZeroPoint(this->batch_mean_));
  }

  template <typename QuantizedType>
  std::vector<float> GetDequantizedBatchVar() {
    return Dequantize<QuantizedType>(
        this->template ExtractVector<QuantizedType>(this->batch_var_),
        GetScale(this->batch_var_), GetZeroPoint(this->batch_var_));
  }

  int input() { return input_; }
  int scale() { return scale_; }
  int offset() { return offset_; }

  TensorData SymmetricInt16Scaling(TensorData tensor) {
    // Symmetric range and null zero-point is required for INT16 tensors. As
    // SingleOpModel::QuantizationParams calculates the scale on an asymmetric
    // base [int_type::min, int_type::max], manually calculate the scale on a
    // symmetric range [int_type::min+1, int_type::max] to ensure a null
    // zero-point.
    if (tensor.type == TensorType_INT16) {
      CHECK_EQ(std::abs(tensor.min), tensor.max);
      tensor.scale = tensor.max / std::numeric_limits<int16_t>::max();
      tensor.zero_point = 0;
      tensor.min = 0;
      tensor.max = 0;
    }
    return tensor;
  }

 protected:
  int input_;
  int scale_;
  int offset_;
  int output_;
  int batch_mean_;
  int batch_var_;
};

template <typename Float>
class StablehloBatchNormTrainingTestFloat : public ::testing::Test {
 public:
  using FloatType = Float;
};

using FloatTestTypes = ::testing::Types<float, Eigen::half, Eigen::bfloat16>;

TYPED_TEST_SUITE(StablehloBatchNormTrainingTestFloat, FloatTestTypes);

TYPED_TEST(StablehloBatchNormTrainingTestFloat,
           StablehloBatchNormTrainingFloatTest) {
  using Float = typename TestFixture::FloatType;
  TfLiteStablehloBatchNormTrainingParams params = {0.0 /*epsilon*/,
                                                   2 /*feature_index*/};
  StablehloBatchNormTrainingOpModel model(
      {GetTTEnum<Float>(), {2, 2, 2}}, {GetTTEnum<Float>(), {2}},
      {GetTTEnum<Float>(), {2}}, {GetTTEnum<Float>(), {}},
      {GetTTEnum<Float>(), {}}, {GetTTEnum<Float>(), {}}, params);
  model.SetInput<Float>({static_cast<Float>(1.0), static_cast<Float>(2.0),
                         static_cast<Float>(3.0), static_cast<Float>(4.0),
                         static_cast<Float>(3.0), static_cast<Float>(4.0),
                         static_cast<Float>(1.0), static_cast<Float>(2.0)});
  model.SetScale<Float>({static_cast<Float>(1.0), static_cast<Float>(1.0)});
  model.SetOffset<Float>({static_cast<Float>(1.0), static_cast<Float>(1.0)});

  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput<Float>(),
              ElementsAreArray(ArrayFloatNear(
                  {static_cast<Float>(0.0), static_cast<Float>(0.0),
                   static_cast<Float>(2.0), static_cast<Float>(2.0),
                   static_cast<Float>(2.0), static_cast<Float>(2.0),
                   static_cast<Float>(0.0), static_cast<Float>(0.0)},
                  0.1)));
  EXPECT_THAT(model.GetBatchMean<Float>(),
              ElementsAreArray(ArrayFloatNear(
                  {static_cast<Float>(2.0), static_cast<Float>(3.0)}, 0.1)));
  EXPECT_THAT(model.GetBatchVar<Float>(),
              ElementsAreArray(ArrayFloatNear(
                  {static_cast<Float>(1.0), static_cast<Float>(1.0)}, 0.1)));
}

TYPED_TEST(StablehloBatchNormTrainingTestFloat, Ex2) {
  using Float = typename TestFixture::FloatType;
  TfLiteStablehloBatchNormTrainingParams params = {0.0 /*epsilon*/,
                                                   1 /*feature_index*/};
  StablehloBatchNormTrainingOpModel model(
      {GetTTEnum<Float>(), {2, 3}}, {GetTTEnum<Float>(), {3}},
      {GetTTEnum<Float>(), {3}}, {GetTTEnum<Float>(), {}},
      {GetTTEnum<Float>(), {}}, {GetTTEnum<Float>(), {}}, params);
  model.SetInput<Float>({static_cast<Float>(1.0), static_cast<Float>(2.0),
                         static_cast<Float>(3.0), static_cast<Float>(4.0),
                         static_cast<Float>(5.0), static_cast<Float>(6.0)});
  model.SetScale<Float>({static_cast<Float>(1.0), static_cast<Float>(1.0),
                         static_cast<Float>(1.0)});
  model.SetOffset<Float>({static_cast<Float>(0.0), static_cast<Float>(0.0),
                          static_cast<Float>(0.0)});

  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput<Float>(),
              ElementsAreArray(ArrayFloatNear(
                  {static_cast<Float>(-1.0), static_cast<Float>(-1.0),
                   static_cast<Float>(-1.0), static_cast<Float>(1.0),
                   static_cast<Float>(1.0), static_cast<Float>(1.0)},
                  0.1)));
  EXPECT_THAT(model.GetBatchMean<Float>(),
              ElementsAreArray(ArrayFloatNear(
                  {static_cast<Float>(2.5), static_cast<Float>(3.5),
                   static_cast<Float>(4.5)},
                  0.1)));
  EXPECT_THAT(model.GetBatchVar<Float>(),
              ElementsAreArray(ArrayFloatNear(
                  {static_cast<Float>(2.25), static_cast<Float>(2.25),
                   static_cast<Float>(2.25)},
                  0.1)));
}

TYPED_TEST(StablehloBatchNormTrainingTestFloat,
           StablehloBatchNormTrainingFloatTestWithNonzeroEpsilon) {
  using Float = typename TestFixture::FloatType;
  TfLiteStablehloBatchNormTrainingParams params = {1.0 /*epsilon*/,
                                                   1 /*feature_index*/};
  StablehloBatchNormTrainingOpModel model(
      {GetTTEnum<Float>(), {2, 2, 2}}, {GetTTEnum<Float>(), {2}},
      {GetTTEnum<Float>(), {2}}, {GetTTEnum<Float>(), {}},
      {GetTTEnum<Float>(), {}}, {GetTTEnum<Float>(), {}}, params);
  model.SetInput<Float>(
      {static_cast<Float>(4.721), static_cast<Float>(-1.903),
       static_cast<Float>(1.939), static_cast<Float>(-3.508),
       static_cast<Float>(-5.371), static_cast<Float>(-0.0968),
       static_cast<Float>(10.517), static_cast<Float>(-9.530)});
  model.SetScale<Float>({static_cast<Float>(2.0), static_cast<Float>(3.0)});
  model.SetOffset<Float>({static_cast<Float>(2.0), static_cast<Float>(2.0)});

  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  const std::vector<Float> expected_values = {
      static_cast<Float>(4.821),  static_cast<Float>(1.312),
      static_cast<Float>(3.363),  static_cast<Float>(0.684),
      static_cast<Float>(-0.467), static_cast<Float>(2.019),
      static_cast<Float>(7.859),  static_cast<Float>(-1.672)};
  const std::vector<Float> expected_mean = {static_cast<Float>(-0.662),
                                            static_cast<Float>(-0.145)};
  const std::vector<Float> expected_var = {static_cast<Float>(13.560),
                                           static_cast<Float>(57.780)};
  EXPECT_THAT(model.GetOutput<Float>(),
              ElementsAreArray(ArrayFloatNear(
                  {static_cast<Float>(4.821), static_cast<Float>(1.312),
                   static_cast<Float>(3.363), static_cast<Float>(0.684),
                   static_cast<Float>(-0.467), static_cast<Float>(2.019),
                   static_cast<Float>(7.859), static_cast<Float>(-1.672)},
                  0.1)));
  EXPECT_THAT(
      model.GetBatchMean<Float>(),
      ElementsAreArray(ArrayFloatNear(
          {static_cast<Float>(-0.662), static_cast<Float>(-0.145)}, 0.1)));
  EXPECT_THAT(
      model.GetBatchVar<Float>(),
      ElementsAreArray(ArrayFloatNear(
          {static_cast<Float>(13.560), static_cast<Float>(57.780)}, 0.1)));
}

// for quantized, the error shouldn't exceed step
template <typename T>
float GetTolerance(float min, float max) {
  float kQuantizedStep =
      2.0 * (max - min) /
      (std::numeric_limits<T>::max() - std::numeric_limits<T>::min());
  return kQuantizedStep;
}

template <typename Int>
class StablehloBatchNormTrainingTestInt : public ::testing::Test {
 public:
  using IntType = Int;
};

using IntTestTypes = ::testing::Types<int8_t, int16_t>;

TYPED_TEST_SUITE(StablehloBatchNormTrainingTestInt, IntTestTypes);

TYPED_TEST(StablehloBatchNormTrainingTestInt,
           StablehloBatchNormTrainingQuantizedTest) {
  using Int = typename TestFixture::IntType;
  TfLiteStablehloBatchNormTrainingParams params = {0.0 /*epsilon*/,
                                                   2 /*feature_index*/};
  float kArithmeticQuantizedTolerance = GetTolerance<Int>(-60.0f, 60.0f);
  const float kQuantizedStep = 2.0 / 255.0;
  const float kMulQuantizedTolerance =
    2.0 * kQuantizedStep + kQuantizedStep * kQuantizedStep;
  float kQuantizedTolerance = 4*kArithmeticQuantizedTolerance + kMulQuantizedTolerance;
  StablehloBatchNormTrainingOpModel model(
      {GetTTEnum<Int>(), {2, 2, 2}, -60.0f, 60.0f},
      {GetTTEnum<Int>(), {2}, -60.0f, 60.0f},
      {GetTTEnum<Int>(), {2}, -60.0f, 60.0f},
      {GetTTEnum<Int>(), {}, -60.0f, 60.0f},
      {GetTTEnum<Int>(), {}, -60.0f, 60.0f},
      {GetTTEnum<Int>(), {}, -60.0f, 60.0f}, params);
  model.QuantizeAndPopulate<Int>(
      model.input(), {static_cast<float>(1.0), static_cast<float>(-2.0),
                      static_cast<float>(3.0), static_cast<float>(-4.0),
                      static_cast<float>(3.0), static_cast<float>(-4.0),
                      static_cast<float>(-1.0), static_cast<float>(2.0)});
  model.QuantizeAndPopulate<Int>(
      model.scale(), {static_cast<float>(1.0), static_cast<float>(1.0)});
  model.QuantizeAndPopulate<Int>(
      model.offset(), {static_cast<float>(1.0), static_cast<float>(1.0)});

  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetDequantizedOutput<Int>(),
              ElementsAreArray(ArrayFloatNear(
                  {static_cast<float>(0.69), static_cast<float>(1.0),
                   static_cast<float>(1.90), static_cast<float>(0.18),
                   static_cast<float>(1.90), static_cast<float>(0.18),
                   static_cast<float>(-0.5), static_cast<float>(2.63)},
                  kQuantizedTolerance)));
  EXPECT_THAT(model.GetDequantizedBatchMean<Int>(),
              ElementsAreArray(ArrayFloatNear(
                  {static_cast<float>(1.5), static_cast<float>(-2.0)},
                  kQuantizedTolerance)));
  EXPECT_THAT(model.GetDequantizedBatchVar<Int>(),
              ElementsAreArray(ArrayFloatNear(
                  {static_cast<float>(2.75), static_cast<float>(6.0)},
                  kQuantizedTolerance)));
}

}  // namespace
}  // namespace tflite
