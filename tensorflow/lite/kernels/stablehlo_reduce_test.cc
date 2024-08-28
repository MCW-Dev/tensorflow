/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
         //
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>

#include <initializer_list>
#include <vector>

#include "Eigen/Core"
#include "subgraph_test_util.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/subgraph_test_util.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using subgraph_test_util::ReduceFunction;
using testing::ElementsAre;
using testing::ElementsAreArray;

class ReduceOpModel : public SingleOpModel {
 public:
  ReduceOpModel(const TensorData& inputs, const TensorData& init_value,
                const TensorData& outputs,
                const TfLiteStablehloReduceParams& params,
                ReduceFunction reduce_function, TfLiteType type) {
    inputs_ = AddInput(SymmetricInt16Scaling(inputs));
    init_value_ = AddInput(SymmetricInt16Scaling(init_value));
    outputs_ = AddOutput(SymmetricInt16Scaling(outputs));
    SetBuiltinOp(
        BuiltinOperator_STABLEHLO_REDUCE,
        BuiltinOptions2_StablehloReduceOptions,
        CreateStablehloReduceOptions(
            builder_,
            builder_.CreateVector(params.dimensions, params.num_dimensions), 1)
            .Union());
    BuildInterpreter({GetShape(inputs_), GetShape(init_value_)},
                     /*num_threads=*/-1, /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/false, /*allocate_and_delegate=*/false,
                     /*use_simple_allocator=*/false);

    AddSubgraphs(1, nullptr);
    subgraph_builder_.BuildReduceSubgraph(interpreter_->subgraph(1),
                                          reduce_function, type);

    AllocateAndDelegate(true);
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor<T>(inputs_, data);
  }

  template <typename T>
  void SetInitValue(std::initializer_list<T> data) {
    PopulateTensor<T>(init_value_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(outputs_);
  }

  template <typename QuantizedType>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<QuantizedType>(
        this->template ExtractVector<QuantizedType>(this->outputs_),
        GetScale(this->outputs_), GetZeroPoint(this->outputs_));
  }

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

  int input() { return inputs_; }
  int init_value() { return init_value_; }
  std::vector<int> GetOutputShape() { return GetTensorShape(outputs_); }

 protected:
  Subgraph* subgraph_;
  int inputs_;
  int init_value_;
  int outputs_;
  ReduceFunction reduce_function;
  subgraph_test_util::SubgraphBuilder subgraph_builder_;
};

}  // namespace
}  // namespace tflite
