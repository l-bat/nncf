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

from typing import List

import openvino.runtime as ov
import numpy as np

from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizationMode

from nncf.experimental.openvino.statistics.collectors import OVMinMaxTensorStatistic


class FQParameters:
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


def calculate_fq_parameters(num_bits: int, input_low: np.ndarray, input_high: np.ndarray, is_symmetric: bool) -> \
        FQParameters:
    levels = 2 ** num_bits
    if is_symmetric:
        output_low = np.full_like(input_low, fill_value=-levels / 2)
        output_high = np.full_like(input_high, fill_value=levels / 2 - 1)
    else:
        output_low = np.zeros_like(input_low)
        output_high = np.full_like(input_high, fill_value=levels - 1)

    return FQParameters(input_low, input_high, output_low, output_high, levels)


def calculate_weight_quantizer_parameters(weight_tensor: np.ndarray, quantizer_config: QuantizerConfig) -> \
        FQParameters:
    """
    Calculates FakeQuantize op attributes for weight quantizer.
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

    return calculate_fq_parameters(num_bits, input_low, input_high, mode == QuantizationMode.SYMMETRIC)


def calculate_activation_quantizer_parameters(statistics: OVMinMaxTensorStatistic, num_bits: int) -> FQParameters:
    """
    Calculates FakeQuantize op attributes for activation quantizer.
    """
    input_low = np.array(statistics.min_values)
    input_high = np.array(statistics.max_values)
    return calculate_fq_parameters(num_bits, input_low, input_high, input_low < 0)


def find_ignored_scopes(disallowed_op_types: List[str], model: ov.Model) -> List[str]:
    """
    Find ignored_scopes from disallowed_op_types.
    For example, if disallowed_op_types = {"Mul", "Add"},
    all nodes which have op_type = "Mul" or "Add" will be added to ignored_scopes.
    """
    disallowed_op_types = set(disallowed_op_types)
    return [op.get_friendly_name() for op in model.get_ops() if op.get_type_name() in disallowed_op_types]
