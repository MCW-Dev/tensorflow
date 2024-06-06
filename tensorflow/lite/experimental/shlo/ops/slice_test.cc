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

#include "tensorflow/lite/experimental/shlo/ops/slice.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/bf16.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/f16.h"
#include "tensorflow/lite/experimental/shlo/ops/test_util.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/status_matcher.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

using testing::Eq;
using testing::FloatEq;
using testing::Pointwise;

namespace shlo_ref {
namespace {
using kSI32TestTypes = ::testing::Types<TestParam<DataType::kSI32>>;
using kBF16TestTypes = ::testing::Types<TestParam<DataType::kBF16>>;
using kF32TestTypes = ::testing::Types<TestParam<DataType::kF32>>;

template <class T>
struct NonQuantizedkF32SliceTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedkF32SliceTest, kF32TestTypes, TestParamNames);
TYPED_TEST(NonQuantizedkF32SliceTest, kF32TestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape operand_shape({5});
  const Shape output_shape({2});

  Vector<float> lhs_float_data(
      {6.3510623, -0.349791348, 1.07969737, 6.3544569, 3.81555367});
  Vector<StorageT> lhs_data(lhs_float_data.begin(), lhs_float_data.end());
  Vector<StorageT> output_data(output_shape.NumElements());
  Vector<int64_t> start_indices_data({1});
  Vector<int64_t> limit_indices_data({5});
  Vector<int64_t> strides_data({2});

  Tensor operand{.type = TensorType{.shape = operand_shape,
                                    .element_type = TypeParam::kStorage},
                 .data = lhs_data.data()};
  Tensor output{.type = TensorType{.shape = output_shape,
                                   .element_type = TypeParam::kStorage},
                .data = output_data.data()};

  auto op = Create(SliceOp::Attributes{.start_indices = start_indices_data,
                                       .limit_indices = limit_indices_data,
                                       .strides = strides_data});

  Vector<float> expected_float_data = {-0.349791348, 6.3544569};
  Vector<StorageT> expected_data(expected_float_data.begin(),
                                 expected_float_data.end());

  ASSERT_OK(Prepare(op, operand, output));
  ASSERT_OK(Evaluate(op, operand, output));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

TYPED_TEST(NonQuantizedkF32SliceTest, kF32TestTypesTensorsWork2) {
  using StorageT = typename TypeParam::StorageT;

  const Shape operand_shape({5, 3});
  const Shape output_shape({1, 0});

  Vector<float> lhs_float_data(
      {-3.36093402, -0.727180302, -1.36010623, -1.01435328, 0.688827634,
       -0.296935827, -2.48586321, -1.59691119, -2.82093692, 1.89409471,
       4.85353708, 2.03780484, 2.1872561, 0.797380387, 1.21235549});
  Vector<StorageT> lhs_data(lhs_float_data.begin(), lhs_float_data.end());
  Vector<StorageT> output_data(output_shape.NumElements());
  Vector<int64_t> start_indices_data({1, 1});
  Vector<int64_t> limit_indices_data({2, 1});
  Vector<int64_t> strides_data({1, 1});

  Tensor operand{.type = TensorType{.shape = operand_shape,
                                    .element_type = TypeParam::kStorage},
                 .data = lhs_data.data()};
  Tensor output{.type = TensorType{.shape = output_shape,
                                   .element_type = TypeParam::kStorage},
                .data = output_data.data()};

  auto op = Create(SliceOp::Attributes{.start_indices = start_indices_data,
                                       .limit_indices = limit_indices_data,
                                       .strides = strides_data});

  Vector<float> expected_float_data = {};
  Vector<StorageT> expected_data(expected_float_data.begin(),
                                 expected_float_data.end());

  ASSERT_OK(Prepare(op, operand, output));
  ASSERT_OK(Evaluate(op, operand, output));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

template <class T>
struct NonQuantizedkBF16SliceTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedkBF16SliceTest, kBF16TestTypes, TestParamNames);
TYPED_TEST(NonQuantizedkBF16SliceTest, kBF16TestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape operand_shape({3});
  const Shape output_shape({1});

  Vector<float> lhs_float_data({-5.078130e-01, 1.767580e-01, -3.890630e+00});
  Vector<StorageT> lhs_data(lhs_float_data.begin(), lhs_float_data.end());
  Vector<StorageT> output_data(output_shape.NumElements());
  Vector<int64_t> start_indices_data({1});
  Vector<int64_t> limit_indices_data({2});
  Vector<int64_t> strides_data({1});

  Tensor operand{.type = TensorType{.shape = operand_shape,
                                    .element_type = TypeParam::kStorage},
                 .data = lhs_data.data()};
  Tensor output{.type = TensorType{.shape = output_shape,
                                   .element_type = TypeParam::kStorage},
                .data = output_data.data()};

  auto op = Create(SliceOp::Attributes{.start_indices = start_indices_data,
                                       .limit_indices = limit_indices_data,
                                       .strides = strides_data});

  Vector<float> expected_float_data = {1.767580e-01};
  Vector<StorageT> expected_data(expected_float_data.begin(),
                                 expected_float_data.end());

  ASSERT_OK(Prepare(op, operand, output));
  ASSERT_OK(Evaluate(op, operand, output));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

template <class T>
struct NonQuantizedkSI32SliceTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedkSI32SliceTest, kSI32TestTypes, TestParamNames);
TYPED_TEST(NonQuantizedkSI32SliceTest, kSI32TestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape operand_shape({3});
  const Shape output_shape({1});

  Vector<StorageT> lhs_data({1, -4, 1});
  Vector<StorageT> output_data(output_shape.NumElements());
  Vector<int64_t> start_indices_data({1});
  Vector<int64_t> limit_indices_data({2});
  Vector<int64_t> strides_data({1});

  Tensor operand{.type = TensorType{.shape = operand_shape,
                                    .element_type = TypeParam::kStorage},
                 .data = lhs_data.data()};
  Tensor output{.type = TensorType{.shape = output_shape,
                                   .element_type = TypeParam::kStorage},
                .data = output_data.data()};

  auto op = Create(SliceOp::Attributes{.start_indices = start_indices_data,
                                       .limit_indices = limit_indices_data,
                                       .strides = strides_data});

  Vector<StorageT> expected_data = {-4};

  ASSERT_OK(Prepare(op, operand, output));
  ASSERT_OK(Evaluate(op, operand, output));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_data));
}

}  // namespace
}  // namespace shlo_ref