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

from typing import Union
from typing import List

import numpy as np

from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic


class QuantizerLayerParameters:
    """
    Class handles quantizer layer attributes.
    """

    def __init__(self, input_low: List[float], input_high: List[float], num_bits: int, mode: QuantizationMode):
        self.input_low = input_low
        self.input_high = input_high
        self.num_bits = num_bits
        self.mode = mode


def calculate_weight_quantizer_parameters(weight_tensor: np.ndarray, quantizer_config: QuantizerConfig) -> \
        QuantizerLayerParameters:
    """
    Calculates layer attributes for weight quantizer.
    """
    per_channel = quantizer_config.per_channel
    num_bits = quantizer_config.num_bits
    mode = quantizer_config.mode

    if per_channel:
        axes = tuple(range(len(weight_tensor.shape))[1:])
    else:
        axes = None

    input_high = np.amax(weight_tensor, axis=axes)
    input_low = np.amin(weight_tensor, axis=axes)
    return QuantizerLayerParameters(input_low.tolist(), input_high.tolist(), num_bits, mode)

def calculate_activation_quantizer_parameters(statistics: MinMaxTensorStatistic,
                                              quantizer_config: QuantizerConfig) -> QuantizerLayerParameters:
    """
    Calculates layer attributes for activation quantizer.
    """
    num_bits = quantizer_config.num_bits
    mode = quantizer_config.mode
    input_low = statistics.min_values
    input_high = statistics.max_values
    return QuantizerLayerParameters(input_low.tolist(), input_high.tolist(), num_bits, mode)




def get_scale_zp_from_input_low_input_high(level_low, level_high, input_low, input_high):
    levels = level_high - level_low + 1
    assert levels in [255, 256], "Can only export to INT8 256-level ONNX Quantize/Dequantize pairs"

    y_scale = (input_high - input_low) / (level_high - level_low)
    y_zero_point = (level_low * input_high - level_high * input_low) / (input_high - input_low)

    type_ = np.int8 if level_low < 0 else np.uint8
    level_low *= np.ones_like(y_zero_point).to(type_)
    level_high *= np.ones_like(y_zero_point).to(type_)
    level_low = level_low.to(y_zero_point.device)
    level_high = level_high.to(y_zero_point.device)
    y_zero_point = np.min(np.max(level_low, y_zero_point.to(type_)), level_high)

    y_scale = np.squeeze(y_scale)
    y_zero_point = np.squeeze(y_zero_point)
    return y_scale, y_zero_point


def calculate_scale_level(parameters: QuantizerLayerParameters) -> Union[float, np.ndarray]:
    """
    Calculates Quantizer/Dequantizer layer scale level.
    """
    min_val, max_val = parameters.input_low, parameters.input_high
    num_bits = parameters.num_bits
    if parameters.mode == QuantizationMode.SYMMETRIC:
        input_abs_max = np.maximum(np.abs(max_val), np.abs(min_val))
        return input_abs_max / ((2 ** num_bits - 1) / 2)
    return (max_val - min_val) / 2 ** num_bits



class ONNXQuantizerLayerParameters:
    """
    Class handles quantizer layer attributes.
    """

    def __init__(self, scales: List[float], zero_points: List[float], mode: QuantizationMode):
        self.scales = scales
        self.zero_points = zero_points
        self.mode = mode


def get_scale_zp_from_quantizer_parameters(parameters: QuantizerLayerParameters) -> ONNXQuantizerLayerParameters:
    """
    Calculates Quantizer/Dequantizer layer attributes for weight quantizer such as scale, zero_points and
    quantization mode: symmetric, asymmetric.
    """
    scales = calculate_scale_level(parameters)
    zero_points = np.zeros_like(scales, dtype=np.int64)
    return ONNXQuantizerLayerParameters(scales.tolist(), zero_points.tolist(), parameters.mode)


class OVQuantizerLayerParameters:
    """
    Class handles FakeQuantize op attributes.
    """
    def __init__(self,
                 input_low: np.ndarray,
                 input_high: np.ndarray,
                 output_low: np.ndarray,
                 output_high: np.ndarray,
                 levels: int):
        self.input_low = input_low
        self.input_high = input_high
        self.output_low = output_low
        self.output_high = output_high
        self.levels = levels


def calculate_fq_parameters(parameters: QuantizerLayerParameters) -> OVQuantizerLayerParameters:
    levels = 2 ** parameters.num_bits
    if parameters.mode == QuantizationMode.SYMMETRIC:
        output_low = np.full_like(parameters.input_low, fill_value=-levels / 2)
        output_high = np.full_like(parameters.input_high, fill_value=levels / 2 - 1)
    else:
        output_low = np.zeros_like(parameters.input_low)
        output_high = np.full_like(parameters.input_high, fill_value=levels - 1)

    return OVQuantizerLayerParameters(parameters.input_low, parameters.input_high, output_low, output_high, levels)