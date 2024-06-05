
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

#include "tensorflow/lite/experimental/shlo/ops/sort.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "absl/algorithm/container.h"
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

namespace shlo_ref {

namespace {
template <class T>
struct NonQuantizedkSI32SortTest : ::testing::Test {};
using kSI32TestTypes = ::testing::Types<TestParam<DataType::kSI32>>;
TYPED_TEST_SUITE(NonQuantizedkSI32SortTest, kSI32TestTypes, TestParamNames);

TYPED_TEST(NonQuantizedkSI32SortTest, kSI32TestTypesTensorsWork01) {
  using StorageT = typename TypeParam::StorageT;
  const Shape shapeInput({2, 3});
  const Shape shapeR({2, 3});
  std::vector<StorageT> input0_data = {1, 2, 3, 3, 2, 1};
  std::vector<StorageT> input1_data = {3, 2, 1, 1, 2, 3};
  int64_t dimension = 0;
  bool is_stable = true;
  auto comparator = [](absl::Span<const int64_t> args) {
    return (args[0] > args[1]);
  };
  std::vector<StorageT> output0_data(shapeR.NumElements());
  std::vector<StorageT> output1_data(shapeR.NumElements());

  Tensor input0{.type = TensorType{.shape = shapeInput,
                                   .element_type = TypeParam::kStorage},
                .data = input0_data.data()};
  Tensor input1{.type = TensorType{.shape = shapeInput,
                                   .element_type = TypeParam::kStorage},
                .data = input1_data.data()};

  Tensor output0_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output0_data.data()};
  Tensor output1_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output1_data.data()};

  std::vector<StorageT> expected_output0_data = {3, 2, 3, 1, 2, 1};
  std::vector<StorageT> expected_output1_data = {1, 2, 1, 3, 2, 3};

  Tensor expected_output0_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = expected_output0_data.data()};
  Tensor expected_output1_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = expected_output1_data.data()};

  std::vector<Tensor> inputs = {input0, input1};
  std::vector<Tensor> outputs = {output0_tensor, output1_tensor};

  auto op = Create(sortOp::Attributes{
      .dimension = dimension,
      .is_stable = is_stable,
      .comparator = comparator,
  });

  ASSERT_OK(Prepare(op, inputs, outputs));
  ASSERT_OK(Evaluate(op, inputs, outputs));

  EXPECT_EQ(output0_data, expected_output0_data);
  EXPECT_EQ(output1_data, expected_output1_data);
}

TYPED_TEST(NonQuantizedkSI32SortTest, kSI32TestTypesTensorsWork02) {
  using StorageT = typename TypeParam::StorageT;
  const Shape shapeInput({2, 3});
  const Shape shapeR({2, 3});
  std::vector<StorageT> input0_data = {1, 2, 3, 3, 2, 1};
  std::vector<StorageT> input1_data = {3, 2, 1, 1, 2, 3};
  int64_t dimension = 1;
  bool is_stable = true;
  auto comparator = [](absl::Span<const int64_t> args) {
    return (args[0] > args[1]);
  };
  std::vector<StorageT> output0_data(shapeR.NumElements());
  std::vector<StorageT> output1_data(shapeR.NumElements());

  Tensor input0{.type = TensorType{.shape = shapeInput,
                                   .element_type = TypeParam::kStorage},
                .data = input0_data.data()};
  Tensor input1{.type = TensorType{.shape = shapeInput,
                                   .element_type = TypeParam::kStorage},
                .data = input1_data.data()};

  Tensor output0_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output0_data.data()};
  Tensor output1_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output1_data.data()};

  std::vector<StorageT> expected_output0_data = {3, 2, 1, 3, 2, 1};
  std::vector<StorageT> expected_output1_data = {1, 2, 3, 1, 2, 3};

  Tensor expected_output0_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = expected_output0_data.data()};
  Tensor expected_output1_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = expected_output1_data.data()};

  std::vector<Tensor> inputs = {input0, input1};
  std::vector<Tensor> outputs = {output0_tensor, output1_tensor};

  auto op = Create(sortOp::Attributes{
      .dimension = dimension,
      .is_stable = is_stable,
      .comparator = comparator,
  });
  ASSERT_OK(Prepare(op, inputs, outputs));
  ASSERT_OK(Evaluate(op, inputs, outputs));

  EXPECT_EQ(output0_data, expected_output0_data);
  EXPECT_EQ(output1_data, expected_output1_data);
}

TYPED_TEST(NonQuantizedkSI32SortTest, kSI32TestTypesTensorsWork03) {
  using StorageT = typename TypeParam::StorageT;
  const Shape shapeInput({2, 3});
  const Shape shapeR({2, 3});
  std::vector<StorageT> input0_data = {1, 2, 3, 3, 2, 1};
  std::vector<StorageT> input1_data = {3, 2, 1, 1, 2, 3};
  int64_t dimension = 1;
  bool is_stable = true;
  auto comparator = [](absl::Span<const int64_t> args) {
    return (args[0] > args[1]);
  };
  std::vector<StorageT> output0_data(shapeR.NumElements());
  std::vector<StorageT> output1_data(shapeR.NumElements());

  Tensor input0{.type = TensorType{.shape = shapeInput,
                                   .element_type = TypeParam::kStorage},
                .data = input0_data.data()};
  Tensor input1{.type = TensorType{.shape = shapeInput,
                                   .element_type = TypeParam::kStorage},
                .data = input1_data.data()};

  Tensor output0_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output0_data.data()};
  Tensor output1_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output1_data.data()};

  std::vector<StorageT> expected_output0_data = {3, 2, 1, 3, 2, 1};
  std::vector<StorageT> expected_output1_data = {1, 2, 3, 1, 2, 3};

  Tensor expected_output0_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = expected_output0_data.data()};
  Tensor expected_output1_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = expected_output1_data.data()};

  std::vector<Tensor> inputs = {input0, input1};
  std::vector<Tensor> outputs = {output0_tensor, output1_tensor};

  auto op = Create(sortOp::Attributes{
      .dimension = dimension,
      .is_stable = is_stable,
      .comparator = comparator,
  });
  ASSERT_OK(Prepare(op, inputs, outputs));
  ASSERT_OK(Evaluate(op, inputs, outputs));

  EXPECT_EQ(output0_data, expected_output0_data);
  EXPECT_EQ(output1_data, expected_output1_data);
}

TYPED_TEST(NonQuantizedkSI32SortTest, kSI32TestTypesTensorsWork04) {
  using StorageT = typename TypeParam::StorageT;
  const Shape shapeInput({2, 4});
  const Shape shapeR({2, 4});
  std::vector<StorageT> input0_data = {5, 2, 3, 1, 3, 2, 5, 4};
  std::vector<StorageT> input1_data = {3, 2, 1, 6, 1, 2, 3, 7};
  std::vector<StorageT> input2_data = {5, 2, 3, 2, 4, 5, 6, 9};
  int64_t dimension = 0;
  bool is_stable = true;
  auto comparator = [](absl::Span<const int64_t> args) {
    return (args[0] > args[1]);
  };
  std::vector<StorageT> output0_data(shapeR.NumElements());
  std::vector<StorageT> output1_data(shapeR.NumElements());
  std::vector<StorageT> output2_data(shapeR.NumElements());

  Tensor input0{.type = TensorType{.shape = shapeInput,
                                   .element_type = TypeParam::kStorage},
                .data = input0_data.data()};
  Tensor input1{.type = TensorType{.shape = shapeInput,
                                   .element_type = TypeParam::kStorage},
                .data = input1_data.data()};
  Tensor input2{.type = TensorType{.shape = shapeInput,
                                   .element_type = TypeParam::kStorage},
                .data = input2_data.data()};

  Tensor output0_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output0_data.data()};
  Tensor output1_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output1_data.data()};
  Tensor output2_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output2_data.data()};

  std::vector<StorageT> expected_output0_data = {5, 2, 5, 4, 3, 2, 3, 1};
  std::vector<StorageT> expected_output1_data = {3, 2, 3, 7, 1, 2, 1, 6};
  std::vector<StorageT> expected_output2_data = {5, 2, 6, 9, 4, 5, 3, 2};

  Tensor expected_output0_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = expected_output0_data.data()};
  Tensor expected_output1_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = expected_output1_data.data()};
  Tensor expected_output2_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = expected_output2_data.data()};
  std::vector<Tensor> inputs = {input0, input1, input2};
  std::vector<Tensor> outputs = {output0_tensor, output1_tensor,
                                 output2_tensor};

  auto op = Create(sortOp::Attributes{
      .dimension = dimension,
      .is_stable = is_stable,
      .comparator = comparator,
  });
  ASSERT_OK(Prepare(op, inputs, outputs));
  ASSERT_OK(Evaluate(op, inputs, outputs));

  EXPECT_EQ(output0_data, expected_output0_data);
  EXPECT_EQ(output1_data, expected_output1_data);
  EXPECT_EQ(output2_data, expected_output2_data);
}

TYPED_TEST(NonQuantizedkSI32SortTest, kSI32TestTypesTensorsWork05) {
  using StorageT = typename TypeParam::StorageT;
  const Shape shapeInput({2, 4});
  const Shape shapeR({2, 4});
  std::vector<StorageT> input0_data = {5, 2, 3, 1, 3, 2, 5, 4};
  std::vector<StorageT> input1_data = {3, 2, 1, 6, 1, 2, 3, 7};
  std::vector<StorageT> input2_data = {5, 2, 3, 2, 4, 5, 6, 9};
  int64_t dimension = 1;
  bool is_stable = true;
  auto comparator = [](absl::Span<const int64_t> args) {
    return (args[0] > args[1]);
  };
  std::vector<StorageT> output0_data(shapeR.NumElements());
  std::vector<StorageT> output1_data(shapeR.NumElements());
  std::vector<StorageT> output2_data(shapeR.NumElements());

  Tensor input0{.type = TensorType{.shape = shapeInput,
                                   .element_type = TypeParam::kStorage},
                .data = input0_data.data()};
  Tensor input1{.type = TensorType{.shape = shapeInput,
                                   .element_type = TypeParam::kStorage},
                .data = input1_data.data()};
  Tensor input2{.type = TensorType{.shape = shapeInput,
                                   .element_type = TypeParam::kStorage},
                .data = input2_data.data()};

  Tensor output0_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output0_data.data()};
  Tensor output1_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output1_data.data()};
  Tensor output2_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output2_data.data()};

  std::vector<StorageT> expected_output0_data = {5, 3, 2, 1, 5, 4, 3, 2};
  std::vector<StorageT> expected_output1_data = {3, 1, 2, 6, 3, 7, 1, 2};
  std::vector<StorageT> expected_output2_data = {5, 3, 2, 2, 6, 9, 4, 5};

  Tensor expected_output0_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = expected_output0_data.data()};
  Tensor expected_output1_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = expected_output1_data.data()};
  Tensor expected_output2_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = expected_output2_data.data()};
  std::vector<Tensor> inputs = {input0, input1, input2};
  std::vector<Tensor> outputs = {output0_tensor, output1_tensor,
                                 output2_tensor};

  auto op = Create(sortOp::Attributes{
      .dimension = dimension,
      .is_stable = is_stable,
      .comparator = comparator,
  });
  ASSERT_OK(Prepare(op, inputs, outputs));
  ASSERT_OK(Evaluate(op, inputs, outputs));

  EXPECT_EQ(output0_data, expected_output0_data);
  EXPECT_EQ(output1_data, expected_output1_data);
  EXPECT_EQ(output2_data, expected_output2_data);
}

TYPED_TEST(NonQuantizedkSI32SortTest, kSI32TestTypesTensorsWork06) {
  using StorageT = typename TypeParam::StorageT;
  const Shape shapeInput({2, 4});
  const Shape shapeR({2, 4});
  std::vector<StorageT> input0_data = {5, 2, 3, 1, 3, 2, 5, 4};
  std::vector<StorageT> input1_data = {3, 2, 1, 6, 1, 2, 3, 7};
  std::vector<StorageT> input2_data = {5, 2, 3, 2, 4, 5, 6, 9};
  int64_t dimension = 1;
  bool is_stable = false;
  auto comparator = [](absl::Span<const int64_t> args) {
    return (args[0] > args[1]);
  };
  std::vector<StorageT> output0_data(shapeR.NumElements());
  std::vector<StorageT> output1_data(shapeR.NumElements());
  std::vector<StorageT> output2_data(shapeR.NumElements());

  Tensor input0{.type = TensorType{.shape = shapeInput,
                                   .element_type = TypeParam::kStorage},
                .data = input0_data.data()};
  Tensor input1{.type = TensorType{.shape = shapeInput,
                                   .element_type = TypeParam::kStorage},
                .data = input1_data.data()};
  Tensor input2{.type = TensorType{.shape = shapeInput,
                                   .element_type = TypeParam::kStorage},
                .data = input2_data.data()};

  Tensor output0_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output0_data.data()};
  Tensor output1_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output1_data.data()};
  Tensor output2_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output2_data.data()};

  std::vector<StorageT> expected_output0_data = {5, 3, 2, 1, 5, 4, 3, 2};
  std::vector<StorageT> expected_output1_data = {3, 1, 2, 6, 3, 7, 1, 2};
  std::vector<StorageT> expected_output2_data = {5, 3, 2, 2, 6, 9, 4, 5};

  Tensor expected_output0_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = expected_output0_data.data()};
  Tensor expected_output1_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = expected_output1_data.data()};
  Tensor expected_output2_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = expected_output2_data.data()};
  std::vector<Tensor> inputs = {input0, input1, input2};
  std::vector<Tensor> outputs = {output0_tensor, output1_tensor,
                                 output2_tensor};

  auto op = Create(sortOp::Attributes{
      .dimension = dimension,
      .is_stable = is_stable,
      .comparator = comparator,
  });
  ASSERT_OK(Prepare(op, inputs, outputs));
  ASSERT_OK(Evaluate(op, inputs, outputs));

  EXPECT_EQ(output0_data, expected_output0_data);
  EXPECT_EQ(output1_data, expected_output1_data);
  EXPECT_EQ(output2_data, expected_output2_data);
}

}  // namespace
}  // namespace shlo_ref
