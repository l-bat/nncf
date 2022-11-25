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

from copy import deepcopy
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
import numpy as np
import onnx

from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.patterns import HWFusedPatterns
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.hardware.config import HWConfig
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.common.utils.backend import BackendType
from nncf.common.utils.logger import logger as nncf_logger

from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import WEIGHT_LAYER_METATYPES
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXNonMaxSuppressionMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXTopKMetatype
from nncf.experimental.onnx.graph.model_transformer import ONNXModelTransformer
from nncf.experimental.onnx.graph.transformations.commands import ONNXQuantizerInsertionCommand
from nncf.experimental.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.experimental.onnx.hardware.config import ONNXHWConfig
from nncf.experimental.onnx.hardware.fused_patterns import ONNX_HW_FUSED_PATTERNS
from nncf.experimental.onnx.quantization.default_quantization import DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT
from nncf.experimental.onnx.statistics.collectors import ONNXMeanMinMaxStatisticCollector
from nncf.experimental.onnx.statistics.collectors import ONNXMinMaxStatisticCollector

from nncf.experimental.post_training.algorithms.quantization.min_max.backend import MinMaxAlgoBackend
from nncf.experimental.post_training.algorithms.quantization.min_max.backend import ALGO_BACKENDS
from nncf.experimental.post_training.algorithms.quantization.min_max.utils import QuantizerLayerParameters


@ALGO_BACKENDS.register(BackendType.ONNX)
class ONNXMinMaxAlgoBackend(MinMaxAlgoBackend):

    @property
    def layers_with_weights_metatypes(self) -> List[OperatorMetatype]:
        return WEIGHT_LAYER_METATYPES

    @property
    def post_processing_metatypes(self) -> List[OperatorMetatype]:
        return [ONNXTopKMetatype, ONNXNonMaxSuppressionMetatype]

    @property
    def hw_fused_patterns(self) -> HWFusedPatterns:
        return ONNX_HW_FUSED_PATTERNS

    @property
    def hw_config(self) -> HWConfig:
        return ONNXHWConfig

    @property
    def quant_trait_op_dict(self) -> Dict[int, OperatorMetatype]:
        return DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT

    @staticmethod
    def model_transformer(model: onnx.ModelProto) -> ONNXModelTransformer:
        return ONNXModelTransformer(model)

    @staticmethod
    def target_point(target_type: TargetType,
                     target_node_name: str = None,
                     port_id: int = None) -> ONNXTargetPoint:
        if target_type == TargetType.PRE_LAYER_OPERATION and port_id is None:
            raise RuntimeError('port_id must be specified for target_point with PRE_LAYER_OPERATION type')

        if target_type == TargetType.POST_LAYER_OPERATION:
            port_id = 0
        elif target_type == TargetType.OPERATION_WITH_WEIGHTS:
            port_id = 1

        input_tensor_names, _ = ONNXMinMaxAlgoBackend.get_tensor_names(node)
        edge_name = input_tensor_names[port_id]
        return ONNXTargetPoint(target_type, target_node_name, edge_name)

    @staticmethod
    def quantizer_insertion_command(target_point: ONNXTargetPoint,
                                    quantizer_config: QuantizerConfig,
                                    statistics: Union[MinMaxTensorStatistic, np.ndarray],
                                    ) -> ONNXQuantizerInsertionCommand:
        parameters = calculate_quantizer_parameters(statistics, quantizer_config)
        return ONNXQuantizerInsertionCommand(target_point, parameters)


class ONNXQuantizerLayerParameters:
    """
    Class handles Quantizer/Dequantizer layer attributes.
    """

    def __init__(self, scale: List[float], zero_point: List[int], mode: QuantizationMode):
        self.scale = scale
        self.zero_point = zero_point
        self.mode = mode

    @staticmethod
    def calculate_scale_zp(input_low: List[float],
                           input_high: List[float],
                           num_bits: int,
                           mode: QuantizationMode) -> Union[float, np.ndarray]:
        """
        Calculates Quantizer/Dequantizer layer scale level.
        """
        if mode == QuantizationMode.SYMMETRIC:
            input_abs_max = np.maximum(np.abs(input_high), np.abs(input_low))
            scales = input_abs_max / ((2 ** num_bits - 1) / 2)
            zero_point = np.zeros_like(scales, dtype=np.int64)
        else:
            scales = (input_high - input_low) / 2 ** num_bits
            zero_point = -input_low / scales
        return scales, zero_point


    # def calculate_weight_quantizer_parameters(weight_tensor: np.ndarray, quantizer_config: QuantizerConfig) -> \
    #         ONNXQuantizerLayerParameters:
    #     """
    #     Calculates layer attributes for weight quantizer.
    #     """
    #     num_bits = quantizer_config.num_bits
    #     mode = quantizer_config.mode

    #     per_channel = quantizer_config.per_channel
    #     if per_channel:
    #         axes = tuple(range(len(weight_tensor.shape))[1:])
    #     else:
    #         axes = None

    #     input_high = np.amax(weight_tensor, axis=axes)
    #     input_low = np.amin(weight_tensor, axis=axes)
    #     scale, zero_point = calculate_scale_zp(input_low,  input_high, num_bits, mode)
    #     return ONNXQuantizerLayerParameters(scale.tolist(), zero_point.tolist(), mode)

    # def calculate_activation_quantizer_parameters(statistics: MinMaxTensorStatistic,
    #                                             quantizer_config: QuantizerConfig) -> ONNXQuantizerLayerParameters:
    #     """
    #     Calculates layer attributes for activation quantizer.
    #     """
    #     num_bits = quantizer_config.num_bits
    #     mode = quantizer_config.mode
    #     input_low = statistics.min_values
    #     input_high = statistics.max_values
    #     scale, zero_point = calculate_scale_level(input_low,  input_high, num_bits, mode)
    #     return ONNXQuantizerLayerParameters(scale.tolist(), zero_point.tolist(), mode)

    @staticmethod
    def calculate_quantizer_parameters(statistics: Union[MinMaxTensorStatistic, np.ndarray],
                                       quantizer_config: QuantizerConfig) -> ONNXQuantizerLayerParameters:
        """
        Calculates layer attributes for quantizer.
        """
        num_bits = quantizer_config.num_bits
        mode = quantizer_config.mode

        if isinstance(statistics, MinMaxTensorStatistic):
            input_low = statistics.min_values
            input_high = statistics.max_values
        else:
            per_channel = quantizer_config.per_channel
            axes = tuple(range(len(statistics.shape))[1:]) if per_channel else None
            input_high = np.amax(statistics, axis=axes)
            input_low = np.amin(statistics, axis=axes)

        scale, zero_point = calculate_scale_zp(input_low, input_high, num_bits, mode)
        return ONNXQuantizerLayerParameters(scale.tolist(), zero_point.tolist(), mode)


    @staticmethod
    def minmax_statistic_collector(use_abs_max: bool,
                                   reduction_shape: ReductionShape,
                                   num_samples: int = None) -> ONNXMinMaxStatisticCollector:
        return ONNXMinMaxStatisticCollector(use_abs_max, reduction_shape,  num_samples)

    @staticmethod
    def mean_minmax_statistic_collector(use_per_sample_stats: bool,
                                        use_abs_max: bool,
                                        reduction_shape: ReductionShape,
                                        num_samples: int = None,
                                        window_size: int = None) -> ONNXMeanMinMaxStatisticCollector:
        return ONNXMeanMinMaxStatisticCollector(use_per_sample_stats,
                                                use_abs_max,
                                                reduction_shape,
                                                num_samples,
                                                window_size)

    @staticmethod
    def get_weight_tensor(model: onnx.ModelProto, port_id: int) -> np.ndarray:
        initializer_name = 
        for initializer in model.graph.initializer:
            if initializer.name == initializer_name:
                return onnx.numpy_helper.to_array(initializer)
        raise RuntimeError(
            'There is no initializer with the name {}'.format(initializer_name))

    @staticmethod
    def get_tensor_names(node: NNCFNode) -> Tuple[List[str], List[str]]:
        return node.layer_attributes.input_tensor_names, \
            node.layer_attributes.output_tensor_names

    @staticmethod
    def get_weight_config(config: QuantizerConfig, model: onnx.ModelProto) -> QuantizerConfig:
        config = deepcopy(config)
        if model.opset_import[0].version < 13:
            config.per_channel = False
            nncf_logger.warning(
                f"Model opset version is {model.opset_import[0].version} < 13. "
                "Per-channel quantization is not supported. "
                "Set weight_quantizer_config.per_channel = False"
            )

        return config
