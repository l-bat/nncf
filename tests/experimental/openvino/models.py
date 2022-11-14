"""
 Copyright (c) 2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from abc import ABC
import numpy as np
import openvino.runtime as ov

from tests.onnx.models import ONNXReferenceModel

from tests.onnx.models import LinearModel as ONNXLinearModel
from tests.onnx.models import MultiInputOutputModel as ONNXMultiInputOutputModel
from tests.onnx.models import OneConvolutionalModel as ONNXOneConvolutionalModel
from tests.onnx.models import OneInputPortQuantizableModel as ONNXOneInputPortQuantizableModel
from tests.onnx.models import ManyInputPortsQuantizableModel as ONNXManyInputPortsQuantizableModel


class OVReferenceModel(ABC):
    def __init__(self, onnx_reference_model: ONNXReferenceModel):
        self.input_shape = onnx_reference_model.input_shape
        core = ov.Core()
        self.ov_model = core.read_model(onnx_reference_model.onnx_model)


class OVLinearModel(OVReferenceModel):
    def __init__(self, onnx_reference_model=ONNXLinearModel()):
        super().__init__(onnx_reference_model)


class OVMultiInputOutputModel(OVReferenceModel):
    def __init__(self, onnx_reference_model=ONNXMultiInputOutputModel()):
        super().__init__(onnx_reference_model)


class OVOneConvolutionalModel(OVReferenceModel):
    def __init__(self, onnx_reference_model=ONNXOneConvolutionalModel()):
        super().__init__(onnx_reference_model)


class OVOneInputPortQuantizableModel(OVReferenceModel):
    def __init__(self, onnx_reference_model=ONNXOneInputPortQuantizableModel()):
        super().__init__(onnx_reference_model)


class OVManyInputPortsQuantizableModel(OVReferenceModel):
    def __init__(self, onnx_reference_model=ONNXManyInputPortsQuantizableModel()):
        super().__init__(onnx_reference_model)


class OVMultiResultModel:
    def __init__(self):
        self.input_shape = [1, 3, 4, 2]
        self.ov_model = self._create_model(self.input_shape)

    # def __call__(self) -> ov.Model:
    #     return self.ov_model

    def _create_model(self, input_shape):
        # input_1 -> Reshape -> Matmul(reshape, const) -> result_1
        #                |
        #                |-> Add(reshape, const) -> result_2
        input_1 = ov.opset9.parameter(input_shape, name="Input")
        reshape = ov.opset9.reshape(input_1, (1, 3, 2, 4), special_zero=False, name='Reshape')
        data = np.random.rand(1, 3, 4, 5).astype(np.float32)
        matmul = ov.opset9.matmul(reshape, data, transpose_a=False, transpose_b=False, name="MatMul")
        add = ov.opset9.add(reshape, np.random.rand(1, 3, 2, 4).astype(np.float32), name="Add")
        r1 = ov.opset9.result(matmul, name="Result_matmul")
        r2 = ov.opset9.result(add, name="Result_add")
        return ov.Model([r1, r2], [input_1])


class OVMatMulActModel:
    def __init__(self):
        self.input_shape = [1, 3, 4, 2]
        self.ov_model = self._create_model(self.input_shape)

    # def __call__(self) -> ov.Model:
    #     return self.ov_model

    def _create_model(self, input_shape):
        # input_1 -> Reshape -> Matmul(reshape, const) -> result_1
        #                |
        #                |-> Add(reshape, const) -> result_2
        input_1 = ov.opset9.parameter(input_shape, ov.Type.f32, name="Input1")
        input_2 = ov.opset9.parameter(input_shape, ov.Type.f32, name="Input2")
        matmul = ov.opset9.matmul(input_1, input_2, transpose_a=False, transpose_b=True, name="MatMul")
        r1 = ov.opset9.result(matmul, name="Result_matmul")
        return ov.Model([r1], [input_1, input_2])

