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

#include "tensorflow/lite/experimental/shlo/ops/transpose.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/status/status.h"
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
struct NonQuantizedIntTransposeTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedIntTransposeTest, IntTestTypes, TestParamNames);

TYPED_TEST(NonQuantizedIntTransposeTest, IntTestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_operand({2, 3, 2});
  const Shape shape_r({3, 2, 2});
  Vector<StorageT> operand_data =
      Vector<StorageT>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  absl::InlinedVector<Axis, kMaxNumDimensions> permutation = {1, 0, 2};
  Vector<StorageT> output_data(shape_r.NumElements());

  Tensor operand{.type = TensorType{.shape = shape_operand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(TransposeOp::Attributes{
      .permutation = permutation,
  });

  Vector<StorageT> expected_data =
      Vector<StorageT>{1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12};

  ASSERT_OK(Prepare(op, operand, output_tensor));
  ASSERT_OK(Evaluate(op, operand, output_tensor));
  EXPECT_THAT(output_data, expected_data);
}

TYPED_TEST(NonQuantizedIntTransposeTest, IntTestTypesRaiseAnError1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_operand({2, 3, 2});
  const Shape shape_r({3, 2, 2});
  Vector<StorageT> operand_data =
      Vector<StorageT>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  absl::InlinedVector<Axis, kMaxNumDimensions> permutation = {1, 3, 2};
  Vector<StorageT> output_data(shape_r.NumElements());

  Tensor operand{.type = TensorType{.shape = shape_operand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(TransposeOp::Attributes{
      .permutation = permutation,
  });

  const absl::Status status = Prepare(op, operand, output_tensor);
  EXPECT_THAT(status, shlo_ref::testing::StatusIs(
                          absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(status.message(),
              "stablehlo.transpose: The permutation should be in the range of "
              "operand rank.");
}

template <class T>
struct QuantizedIntTransposeTest : ::testing::Test {};

TYPED_TEST_SUITE(QuantizedIntTransposeTest, QuantizedTestTypes, TestParamNames);

TYPED_TEST(QuantizedIntTransposeTest, QuantizedTestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;

  const Shape shape_operand({2, 3, 2});
  const Shape shape_r({3, 2, 2});
  Vector<StorageT> operand_data =
      Vector<StorageT>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  absl::InlinedVector<Axis, kMaxNumDimensions> permutation = {1, 0, 2};
  Vector<StorageT> output_data(shape_r.NumElements());
  const ExpressedT scale = static_cast<ExpressedT>(1.5);
  const StorageT zero_point = static_cast<StorageT>(0);
  const QuantizedElementTypePerTensor tensor_type =
      QuantizedElementTypePerTensor(TypeParam::kStorage, zero_point,
                                    TypeParam::kExpressed, scale);

  Tensor operand{
      .type = QuantizedPerTensorTensorType{.shape = shape_operand,
                                           .element_type = tensor_type},
      .data = operand_data.data()};
  Tensor output_tensor{
      .type = QuantizedPerTensorTensorType{.shape = shape_r,
                                           .element_type = tensor_type},
      .data = output_data.data()};

  auto op = Create(TransposeOp::Attributes{
      .permutation = permutation,
  });

  Vector<StorageT> expected_data{1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12};

  ASSERT_OK(Prepare(op, operand, output_tensor));
  ASSERT_OK(Evaluate(op, operand, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_data));
}

TYPED_TEST(QuantizedIntTransposeTest, QuantizedTestTypesTensorsWork2) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;

  const Shape shape_operand({1, 3, 2});
  const Shape shape_r({3, 1, 2});
  Vector<StorageT> operand_data = Vector<StorageT>{1, 2, 3, 4, 5, 6};
  absl::InlinedVector<Axis, kMaxNumDimensions> permutation = {1, 0, 2};
  Vector<StorageT> output_data(shape_r.NumElements());
  std::vector<StorageT> zeroes = {0};
  std::vector<float> scales = {1.2f};

  QuantizedElementTypePerAxis tensor_type_axis(
      TypeParam::kStorage, zeroes, TypeParam::kExpressed, scales, 0);
  QuantizedElementTypePerAxis tensor_type_axis_output(
      TypeParam::kStorage, zeroes, TypeParam::kExpressed, scales, 1);

  Tensor operand{
      .type = QuantizedPerAxisTensorType{.shape = shape_operand,
                                         .element_type = tensor_type_axis},
      .data = operand_data.data()};
  Tensor output_tensor{
      .type =
          QuantizedPerAxisTensorType{.shape = shape_r,
                                     .element_type = tensor_type_axis_output},
      .data = output_data.data()};

  auto op = Create(TransposeOp::Attributes{
      .permutation = permutation,
  });

  Vector<StorageT> expected_data{1, 2, 3, 4, 5, 6};

  ASSERT_OK(Prepare(op, operand, output_tensor));
  ASSERT_OK(Evaluate(op, operand, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_data));
}

TYPED_TEST(QuantizedIntTransposeTest,
           InvalidQuantizationDimensionRaiseAnError) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;

  const Shape shape_operand({2, 3, 2});
  const Shape shape_r({3, 2, 2});
  Vector<StorageT> operand_data =
      Vector<StorageT>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  absl::InlinedVector<Axis, kMaxNumDimensions> permutation = {1, 0, 2};
  Vector<StorageT> output_data(shape_r.NumElements());
  std::initializer_list<float> zero_points = {0, 0};
  std::initializer_list<float> scales = {1.2, 1.1};
  QuantizedElementTypePerAxis tensor_type_axis(
      TypeParam::kStorage, zero_points, TypeParam::kExpressed, scales, 0);

  Tensor operand{
      .type = QuantizedPerAxisTensorType{.shape = shape_operand,
                                         .element_type = tensor_type_axis},
      .data = operand_data.data()};
  Tensor output_tensor{
      .type = QuantizedPerAxisTensorType{.shape = shape_r,
                                         .element_type = tensor_type_axis},
      .data = output_data.data()};

  auto op = Create(TransposeOp::Attributes{
      .permutation = permutation,
  });

  const absl::Status status = Prepare(op, operand, output_tensor);
  EXPECT_THAT(status, shlo_ref::testing::StatusIs(
                          absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(
      status.message(),
      "stablehlo.transpose: The quantization dimension of operand should be "
      "equal to the permutation of quantization dimension of output.");
}

TYPED_TEST(QuantizedIntTransposeTest, DifferentElementTypeRaiseAnError) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;

  const Shape shape_operand({2, 3, 2});
  const Shape shape_r({3, 2, 2});
  Vector<StorageT> operand_data =
      Vector<StorageT>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  absl::InlinedVector<Axis, kMaxNumDimensions> permutation = {1, 0, 2};
  Vector<StorageT> output_data(shape_r.NumElements());
  std::initializer_list<float> zero_points = {0, 0};
  std::initializer_list<float> scales = {1.2, 1.1};
  std::initializer_list<float> zero_points_output = {1, 0};
  std::initializer_list<float> scales_output = {2, 1.2};

  QuantizedElementTypePerAxis tensor_type_axis(
      TypeParam::kStorage, zero_points, TypeParam::kExpressed, scales, 0);
  QuantizedElementTypePerAxis tensor_type_axis_output(
      TypeParam::kStorage, zero_points_output, TypeParam::kExpressed,
      scales_output, 1);

  Tensor operand{
      .type = QuantizedPerAxisTensorType{.shape = shape_operand,
                                         .element_type = tensor_type_axis},
      .data = operand_data.data()};
  Tensor output_tensor{
      .type =
          QuantizedPerAxisTensorType{.shape = shape_r,
                                     .element_type = tensor_type_axis_output},
      .data = output_data.data()};

  auto op = Create(TransposeOp::Attributes{
      .permutation = permutation,
  });

  const absl::Status status = Prepare(op, operand, output_tensor);
  EXPECT_THAT(status, shlo_ref::testing::StatusIs(
                          absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(status.message(),
              ("stablehlo.transpose: The element type of operand must be same "
               "as the element type of output."));
}

}  // namespace

}  // namespace shlo_ref
