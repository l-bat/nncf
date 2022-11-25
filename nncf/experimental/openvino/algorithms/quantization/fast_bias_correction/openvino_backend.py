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

from typing import Dict, Tuple
from typing import List
import numpy as np
import openvino.runtime as ov

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.common.utils.backend import BackendType
from nncf.common.utils.registry import Registry
from nncf.common.graph import NNCFNode

from nncf.experimental.openvino.graph.metatypes.ov_metatypes import LAYERS_WITH_BIAS_METATYPES
from nncf.experimental.openvino.graph.metatypes.ov_metatypes import OV_OPERATION_METATYPES
from nncf.experimental.openvino.graph.model_transformer import OVModelTransformer
from nncf.experimental.openvino.graph.transformations.commands import OVBiasCorrectionCommand
from nncf.experimental.openvino.graph.transformations.commands import OVModelExtractionCommand
from nncf.experimental.openvino.graph.transformations.commands import OVTargetPoint
from nncf.experimental.openvino.statistics.collectors import OVMeanStatisticCollector
from nncf.experimental.openvino.statistics.collectors import OVNNCFCollectorTensorProcessor
from nncf.experimental.openvino.tensor import OVNNCFTensor
from nncf.experimental.post_training.algorithms.quantization.fast_bias_correction.backend import ALGO_BACKENDS
from nncf.experimental.post_training.algorithms.quantization.fast_bias_correction.backend import FBCAlgoBackend


@ALGO_BACKENDS.register(BackendType.OVNATIVE)
class ONNXFBCAlgoBackend(FBCAlgoBackend):

    @property
    def operation_metatypes(self) -> Registry:
        return OV_OPERATION_METATYPES

    @property
    def layers_with_bias_metatypes(self) -> Registry:
        return LAYERS_WITH_BIAS_METATYPES

    @property
    def channel_axis_by_types(self) -> Dict[str, int]:
        return {'Conv': 1, 'Gemm': -1, 'ConvTranspose': 1}

    @property
    def tensor_processor(self) -> OVNNCFCollectorTensorProcessor:
        return OVNNCFCollectorTensorProcessor()

    @staticmethod
    def model_transformer(model: ov.Model) -> OVModelTransformer:
        return OVModelTransformer(model)

    @staticmethod
    def target_point(target_type: TargetType,
                     target_node_name: str = None,
                     port_id: int = None) -> OVTargetPoint:
        return OVTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def bias_correction_command(target_point: OVTargetPoint,
                                bias_value: np.ndarray,
                                threshold: float) -> OVBiasCorrectionCommand:
        return OVBiasCorrectionCommand(target_point, bias_value, threshold)

    @staticmethod
    def model_extraction_command(inputs: List[str], outputs: List[str]) -> OVModelExtractionCommand:
        return OVModelExtractionCommand(inputs, outputs)

    @staticmethod
    def mean_statistic_collector(reduction_shape: ReductionShape,
                                 num_samples: int = None,
                                 window_size: int = None) -> OVMeanStatisticCollector:
        return OVMeanStatisticCollector(reduction_shape,  num_samples, window_size)

    @staticmethod
    def nncf_tensor(tensor: np.ndarray) -> OVNNCFTensor:
        return OVNNCFTensor(tensor)

    # @staticmethod
    # def get_tensor_names(node: NNCFNode):
    #     return node.layer_attributes.input_tensor_names, \
    #         node.layer_attributes.output_tensor_names

    @staticmethod
    def create_blob(shape: Tuple[int], data: List[float]) -> np.ndarray:
        blob = np.zeros(shape)
        for i, value in enumerate(data):
            blob[:, i] = value
        blob = blob.astype(np.float32)
        return blob

    @staticmethod
    def get_initializer_value(model: ov.Model, tensor_name: str) -> np.ndarray:
        for node in model.get_ops():
            if node.get_friendly_name() == tensor_name and node.get_type() == 'Const':
                return node.get_data()
        raise RuntimeError(f'There is no Constant node with the name {tensor_name}')

    @staticmethod
    def process_model_output(raw_data: Dict, output_name: str) -> OVNNCFTensor:
        return raw_data[output_name]
