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

#include "tensorflow/lite/experimental/shlo/ops/reverse.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/bf16.h"
#include "tensorflow/lite/experimental/shlo/f16.h"
#include "tensorflow/lite/experimental/shlo/ops/test_util.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/status_matcher.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

using shlo_ref::testing::StatusIs;
using testing::Eq;
using testing::FloatEq;
using testing::Pointwise;

namespace shlo_ref {
namespace {

template <class T>
struct NonQuantizedIntReverseTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedIntReverseTest, IntTestTypes, TestParamNames);
TYPED_TEST(NonQuantizedIntReverseTest, IntTestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;
  const Shape operand_shape({3, 2});
  const Shape output_shape({3, 2});
  Vector<StorageT> operand_data = Vector<StorageT>{1, 2, 3, 4, 5, 6};
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> dimensions = {0, 1};
  Vector<StorageT> output_data(output_shape.NumElements());

  Tensor operand{.type = TensorType{.shape = operand_shape,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{.type = TensorType{.shape = output_shape,
                                          .element_type = TypeParam::kStorage},
                       .data = output_data.data()};

  auto op = Create(ReverseOp::Attributes{
      .dimensions = dimensions,
  });

  Vector<StorageT> expected_data = Vector<StorageT>{6, 5, 4, 3, 2, 1};

  ASSERT_OK(Prepare(op, operand, output_tensor));
  ASSERT_OK(Evaluate(op, operand, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_data));
}

using kF32TestTypes = ::testing::Types<TestParam<DataType::kF32>>;
template <class T>
struct NonQuantizedkF32ReverseTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedkF32ReverseTest, kF32TestTypes, TestParamNames);
TYPED_TEST(NonQuantizedkF32ReverseTest, kF32TestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;
  const Shape operand_shape({3, 4, 5});
  const Shape output_shape({3, 4, 5});
  Vector<StorageT> operand_data{
      -1.39323878,  -2.07355523,  3.61496592,  -1.41638482,  4.3499279,
      4.61657476,   -0.847035408, 0.39680019,  4.4041729,    3.43903923,
      -1.1432842,   2.33014345,   -4.82425261, -1.52138329,  8.390310e+00,
      0.895365178,  -1.7085067,   -1.49679315, 0.981733798,  -2.47507167,
      -3.56485152,  -2.95351219,  1.17888641,  1.69931138,   -0.0145214852,
      1.68052375,   2.70574522,   -1.23294461, -1.20053291,  -3.10411549,
      -0.671810328, 2.47057939,   1.90365231,  -0.240815163, -5.70334673,
      5.26833439,   -2.79723477,  -2.1762886,  -1.09088278,  -0.494020909,
      2.68829536,   1.48864734,   -2.68438172, 4.32412481,   -6.42869281,
      5.74774504,   0.600558162,  -3.89856243, -2.57673311,  2.84599566,
      2.61949801,   -1.06490338,  3.48048162,  -2.98022199,  0.0734020919,
      2.57875085,   3.73390079,   -3.21010566, 2.22122025,   -3.73207211};
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> dimensions = {0, 1, 2};
  Vector<StorageT> output_data(output_shape.NumElements());

  Tensor operand{.type = TensorType{.shape = operand_shape,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{.type = TensorType{.shape = output_shape,
                                          .element_type = TypeParam::kStorage},
                       .data = output_data.data()};

  auto op = Create(ReverseOp::Attributes{
      .dimensions = dimensions,
  });

  Vector<StorageT> expected_data{
      -3.73207211,   2.22122025,   -3.21010566, 3.73390079,   2.57875085,
      0.0734020919,  -2.98022199,  3.48048162,  -1.06490338,  2.61949801,
      2.84599566,    -2.57673311,  -3.89856243, 0.600558162,  5.74774504,
      -6.42869281,   4.32412481,   -2.68438172, 1.48864734,   2.68829536,
      -0.494020909,  -1.09088278,  -2.1762886,  -2.79723477,  5.26833439,
      -5.70334673,   -0.240815163, 1.90365231,  2.47057939,   -0.671810328,
      -3.10411549,   -1.20053291,  -1.23294461, 2.70574522,   1.68052375,
      -0.0145214852, 1.69931138,   1.17888641,  -2.95351219,  -3.56485152,
      -2.47507167,   0.981733798,  -1.49679315, -1.7085067,   0.895365178,
      8.390310e+00,  -1.52138329,  -4.82425261, 2.33014345,   -1.1432842,
      3.43903923,    4.4041729,    0.39680019,  -0.847035408, 4.61657476,
      4.3499279,     -1.41638482,  3.61496592,  -2.07355523,  -1.39323878};

  ASSERT_OK(Prepare(op, operand, output_tensor));
  ASSERT_OK(Evaluate(op, operand, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_data));
}

template <class T>
struct ConstraintTest : ::testing::Test {};

TYPED_TEST_SUITE(ConstraintTest, IntTestTypes, TestParamNames);
TYPED_TEST(ConstraintTest, NonUniqueDimensionsRaiseAnError) {
  using StorageT = typename TypeParam::StorageT;
  const Shape operand_shape({3, 2});
  const Shape output_shape({3, 2});
  Vector<StorageT> operand_data =
      Vector<StorageT>{(1), (2), (3), (4), (5), (6)};
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> dimensions = {1, 1};
  Vector<StorageT> output_data(output_shape.NumElements());

  Tensor operand{.type = TensorType{.shape = operand_shape,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{.type = TensorType{.shape = output_shape,
                                          .element_type = TypeParam::kStorage},
                       .data = output_data.data()};

  auto op = Create(ReverseOp::Attributes{
      .dimensions = dimensions,
  });
  Vector<StorageT> expected_data =
      Vector<StorageT>{(6), (5), (4), (3), (2), (1)};

  const absl::Status status = Prepare(op, operand, output_tensor);
  EXPECT_THAT(status, shlo_ref::testing::StatusIs(
                          absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(
      status.message(),
      "stablehlo.reverse: Every dimension to be reversed must be unique.");
}

}  // namespace
}  // namespace shlo_ref