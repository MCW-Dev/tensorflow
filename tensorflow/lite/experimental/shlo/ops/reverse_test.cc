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