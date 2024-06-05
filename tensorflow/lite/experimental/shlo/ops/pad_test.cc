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

#include "tensorflow/lite/experimental/shlo/ops/pad.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/ops/test_util.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/status_matcher.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

using shlo_ref::testing::StatusIs;
using testing::Eq;
using testing::FloatEq;
using testing::FloatNear;
using testing::Pointwise;

namespace shlo_ref {

namespace {

template <class T>
struct ConstraintTest : ::testing::Test {};
TYPED_TEST_SUITE(ConstraintTest, IntTestTypes, TestParamNames);

TYPED_TEST(ConstraintTest, InvalidOutputShapeRaiseAnError) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_operand({5, 4});
  const Shape shape_padding_value{};
  const Shape shape_result({7, 4});

  Vector<StorageT> operand_data{0, 0, 0, 0, 0, 1, 2, 0, 0, 3,
                                4, 0, 0, 5, 6, 0, 0, 0, 0, 0};
  Vector<StorageT> expected_data{-1, -1, -1, -1, -1, -1, 0, -1, 0, -1,
                                 -1, 1,  -1, 2,  -1, -1, 3, -1, 4, -1,
                                 -1, 5,  -1, 6,  -1, -1, 0, -1};
  Vector<DimensionSize> pad_value{-1};
  Vector<DimensionSize> edge_padding_low{1, -1};
  Vector<DimensionSize> edge_padding_high{1, -1};
  Vector<DimensionSize> interior_padding{0, 1};
  Vector<StorageT> output_data(shape_result.NumElements());

  Tensor operand{.type = TensorType{.shape = shape_operand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor padding_value{.type = TensorType{.shape = shape_padding_value,
                                          .element_type = TypeParam::kStorage},
                       .data = pad_value.data()};
  Tensor output_tensor{.type = TensorType{.shape = shape_result,
                                          .element_type = TypeParam::kStorage},
                       .data = output_data.data()};

  auto op = Create(PadOp::Attributes{.edge_padding_low = edge_padding_low,
                                     .edge_padding_high = edge_padding_high,
                                     .interior_padding = interior_padding});

  const absl::Status status =
      Prepare(op, operand, padding_value, output_tensor);

  EXPECT_THAT(status, shlo_ref::testing::StatusIs(
                          absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(status.message(), "stablehlo.pad: Invalid output shape.");
}

template <class T>
struct NonQuantizedIntPadTest : ::testing::Test {};
TYPED_TEST_SUITE(NonQuantizedIntPadTest, IntTestTypes, TestParamNames);

TYPED_TEST(NonQuantizedIntPadTest, IntTestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_operand({5, 4});
  const Shape shape_padding_value{};
  const Shape shape_result({7, 5});

  Vector<StorageT> operand_data{0, 0, 0, 0, 0, 1, 2, 0, 0, 3,
                                4, 0, 0, 5, 6, 0, 0, 0, 0, 0};
  Vector<StorageT> expected_data{-1, -1, -1, -1, -1, -1, 0,  -1, 0,  -1, -1, 1,
                                 -1, 2,  -1, -1, 3,  -1, 4,  -1, -1, 5,  -1, 6,
                                 -1, -1, 0,  -1, 0,  -1, -1, -1, -1, -1, -1};
  Vector<DimensionSize> pad_value{-1};
  Vector<DimensionSize> edge_padding_low{1, -1};
  Vector<DimensionSize> edge_padding_high{1, -1};
  Vector<DimensionSize> interior_padding{0, 1};
  Vector<StorageT> output_data(shape_result.NumElements());

  Tensor operand{.type = TensorType{.shape = shape_operand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor padding_value{.type = TensorType{.shape = shape_padding_value,
                                          .element_type = TypeParam::kStorage},
                       .data = pad_value.data()};
  Tensor output_tensor{.type = TensorType{.shape = shape_result,
                                          .element_type = TypeParam::kStorage},
                       .data = output_data.data()};

  auto op = Create(PadOp::Attributes{.edge_padding_low = edge_padding_low,
                                     .edge_padding_high = edge_padding_high,
                                     .interior_padding = interior_padding});

  ASSERT_OK(Prepare(op, operand, padding_value, output_tensor));
  ASSERT_OK(Evaluate(op, operand, padding_value, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_data));
}

TYPED_TEST(NonQuantizedIntPadTest, IntTestTypesTensorsWork2) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_operand({2, 3});
  const Shape shape_padding_value{};
  const Shape shape_result({5, 9});

  Vector<StorageT> operand_data{1, 2, 3, 4, 5, 6};
  Vector<StorageT> expected_data{0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 4, 0, 0, 5, 0, 0, 6, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  Vector<DimensionSize> pad_value{0};
  Vector<DimensionSize> edge_padding_low{0, 1};
  Vector<DimensionSize> edge_padding_high{2, 1};
  Vector<DimensionSize> interior_padding{1, 2};
  Vector<StorageT> output_data(shape_result.NumElements());

  Tensor operand{.type = TensorType{.shape = shape_operand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor padding_value{.type = TensorType{.shape = shape_padding_value,
                                          .element_type = TypeParam::kStorage},
                       .data = pad_value.data()};
  Tensor output_tensor{.type = TensorType{.shape = shape_result,
                                          .element_type = TypeParam::kStorage},
                       .data = output_data.data()};

  auto op = Create(PadOp::Attributes{.edge_padding_low = edge_padding_low,
                                     .edge_padding_high = edge_padding_high,
                                     .interior_padding = interior_padding});

  ASSERT_OK(Prepare(op, operand, padding_value, output_tensor));
  ASSERT_OK(Evaluate(op, operand, padding_value, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_data));
}
}  // namespace
}  // namespace shlo_ref
