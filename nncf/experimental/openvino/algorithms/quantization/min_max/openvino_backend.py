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

from typing import Dict
from typing import List
from typing import Tuple
import numpy as np
import openvino.runtime as ov

from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.patterns import HWFusedPatterns
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.hardware.config import HWConfig
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.common.utils.backend import BackendType

from nncf.experimental.openvino.graph.metatypes.ov_metatypes import GENERAL_WEIGHT_LAYER_METATYPES
from nncf.experimental.openvino.graph.model_transformer import OVModelTransformer
from nncf.experimental.openvino.graph.transformations.commands import OVQuantizerInsertionCommand
from nncf.experimental.openvino.graph.transformations.commands import OVTargetPoint
from nncf.experimental.openvino.hardware.config import OVHWConfig
from nncf.experimental.openvino.hardware.fused_patterns import OPENVINO_HW_FUSED_PATTERNS
from nncf.experimental.openvino.algorithms.quantization.default_quantization import DEFAULT_OV_QUANT_TRAIT_TO_OP_DICT
from nncf.experimental.openvino.statistics.collectors import OVMeanMinMaxStatisticCollector
from nncf.experimental.openvino.statistics.collectors import OVMinMaxStatisticCollector

from nncf.experimental.post_training.algorithms.quantization.min_max.backend import MinMaxAlgoBackend
from nncf.experimental.post_training.algorithms.quantization.min_max.backend import ALGO_BACKENDS
from nncf.experimental.post_training.algorithms.quantization.min_max.utils import QuantizerLayerParameters


@ALGO_BACKENDS.register(BackendType.OVNATIVE)
class OVMinMaxAlgoBackend(MinMaxAlgoBackend):

    @property
    def layers_with_weights_metatypes(self) -> List[OperatorMetatype]:
        return GENERAL_WEIGHT_LAYER_METATYPES

    @property
    def post_processing_metatypes(self) -> List[OperatorMetatype]:
        return []

    @property
    def hw_fused_patterns(self) -> HWFusedPatterns:
        return OPENVINO_HW_FUSED_PATTERNS

    @property
    def hw_config(self) -> HWConfig:
        return OVHWConfig

    @property
    def quant_trait_op_dict(self) -> Dict[int, OperatorMetatype]:
        return DEFAULT_OV_QUANT_TRAIT_TO_OP_DICT

    @staticmethod
    def model_transformer(model: ov.Model) -> OVModelTransformer:
        return OVModelTransformer(model)

    @staticmethod
    def target_point(target_type: TargetType,
                     target_node_name: str = None,
                     port_id: int = None) -> OVTargetPoint:
        if port_id is None:
            if target_type == TargetType.PRE_LAYER_OPERATION:
                raise RuntimeError('port_id must be specified for PRE_LAYER_OPERATION ')
            port_id = 0 if target_type == TargetType.POST_LAYER_OPERATION else 1
        return OVTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def quantizer_insertion_command(target_point: OVTargetPoint,
                                    parameters: QuantizerLayerParameters) -> OVQuantizerInsertionCommand:
        return OVQuantizerInsertionCommand(target_point, parameters)

    @staticmethod
    def minmax_statistic_collector(use_abs_max: bool,
                                   reduction_shape: ReductionShape,
                                   num_samples: int = None) -> OVMinMaxStatisticCollector:
        return OVMinMaxStatisticCollector(use_abs_max, reduction_shape,  num_samples)

    @staticmethod
    def mean_minmax_statistic_collector(use_per_sample_stats: bool,
                                        use_abs_max: bool,
                                        reduction_shape: ReductionShape,
                                        num_samples: int = None,
                                        window_size: int = None) -> OVMeanMinMaxStatisticCollector:
        return OVMeanMinMaxStatisticCollector(use_per_sample_stats,
                                              use_abs_max,
                                              reduction_shape,
                                              num_samples,
                                              window_size)

    @staticmethod
    def get_initializer_value(model: ov.Model, tensor_name: str) -> np.ndarray:
        for node in model.get_ops():
            if node.get_friendly_name() == tensor_name and node.get_type() == 'Const':
                return node.get_data()
        raise RuntimeError(f'There is no Constant node with the name {tensor_name}')

    @staticmethod
    def get_weight_config(config: QuantizerConfig, model: ov.Model) -> QuantizerConfig:
        return config
