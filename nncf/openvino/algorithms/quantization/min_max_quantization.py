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
from typing import Set
from typing import Dict

import openvino.runtime as ov
# pylint: disable=no-member

from nncf.common.quantization.quantizer_propagation.solver import QuantizerPropagationSolver
from nncf.common.quantization.structs import QuantizableWeightedLayerNode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizationPoint
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.insertion_point_graph import InsertionPointGraph
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.common.utils.logger import logger as nncf_logger

from nncf.experimental.post_training.algorithms.quantization.min_max_quantization import MinMaxQuantization
from nncf.experimental.post_training.algorithms.quantization.min_max_quantization import MinMaxQuantizationParameters
from nncf.experimental.post_training.algorithms.quantization.min_max_quantization import RangeType
from nncf.experimental.post_training.algorithms.algorithm import PostTrainingAlgorithms
from nncf.experimental.post_training.statistics.statistic_point import StatisticPoint
from nncf.experimental.post_training.statistics.statistic_point import StatisticPointsContainer
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.graph.transformations.layout import OVTransformationLayout
from nncf.openvino.graph.metatypes.ov_metatypes import GENERAL_WEIGHT_LAYER_METATYPES
from nncf.openvino.graph.nncf_graph_builder import GraphConverter
from nncf.openvino.algorithms.quantization.default_quantization import DEFAULT_OV_QUANT_TRAIT_TO_OP_DICT
from nncf.openvino.graph.transformations.commands import OVQuantizerInsertionCommand
from nncf.openvino.engine import OVEngine
from nncf.openvino.statistics.collectors import OVMinMaxStatisticCollector
from nncf.openvino.statistics.collectors import OVMeanMinMaxStatisticCollector
from nncf.openvino.graph.model_transformer import OVModelTransformer
from nncf.openvino.hardware.fused_patterns import OPENVINO_HW_FUSED_PATTERNS
from nncf.openvino.algorithms.quantization.utils import calculate_activation_quantizer_parameters
from nncf.openvino.algorithms.quantization.utils import calculate_weight_quantizer_parameters
from nncf.openvino.hardware.config import OVHWConfig
from nncf.common.utils.backend import BackendType

QUANTIZATION_LAYER_METATYPES = GENERAL_WEIGHT_LAYER_METATYPES


class OpenVINOMinMaxQuantization(MinMaxQuantization):

    def __init__(self, parameters: MinMaxQuantizationParameters):
        super().__init__(parameters)
        self.nncf_graph = None
        # It prevents the duplicate weight quantizers from being added.
        # It can happen when you have layers that share the identical weight tensor.
        self._quantization_target_points = set()  # type: Set[OVTargetPoint]

    @property
    def available_backends(self) -> Dict[str, BackendType]:
        pass

    def generate_stat_collector(self, quantizer_config: QuantizerConfig) -> TensorStatisticCollectorBase:
        is_symmetric = quantizer_config.mode == QuantizationMode.SYMMETRIC
        axes = (0, 2, 3) if quantizer_config.per_channel else None
        if self.range_type == RangeType.MINMAX:
            return OVMinMaxStatisticCollector(use_abs_max=is_symmetric, reduction_shape=axes,
                                              num_samples=self.number_samples)
        if self.range_type == RangeType.MEAN_MINMAX:
            return OVMeanMinMaxStatisticCollector(use_per_sample_stats=False, use_abs_max=is_symmetric,
                                                  reduction_shape=axes, num_samples=self.number_samples)
        raise RuntimeError('This range type is not supported.')

    def _create_model_transformer(self, model: ov.Model) -> OVModelTransformer:
        return OVModelTransformer(model)

    def _get_quantizer_setup(self, model: ov.Model):
        self.nncf_graph = GraphConverter.create_nncf_graph(model) if self.nncf_graph is None else self.nncf_graph
        ip_graph = InsertionPointGraph(self.nncf_graph)
        pattern = OPENVINO_HW_FUSED_PATTERNS.get_full_pattern_graph()
        ip_graph = ip_graph.get_ip_graph_with_merged_hw_optimized_operations(pattern)

        weight_nodes = self.nncf_graph.get_nodes_by_metatypes(QUANTIZATION_LAYER_METATYPES)
        for node in weight_nodes:
            print(' ***** node', len(self.nncf_graph.get_input_edges(node)))
            if len(self.nncf_graph.get_input_edges(node)) == 2:  # op w/o weights
                weight_nodes.remove(node)

        quantizable_layer_nodes = [QuantizableWeightedLayerNode(weight_node, [QuantizerConfig()]) for weight_node in
                                   weight_nodes]

        hw_config_type = self.target_device
        hw_config_path = OVHWConfig.get_path_to_hw_config(hw_config_type)
        hw_config = OVHWConfig.from_json(hw_config_path)

        solver = QuantizerPropagationSolver(ignored_scopes=self.ignored_scopes,
                                            hw_config=hw_config,
                                            default_trait_to_metatype_map=DEFAULT_OV_QUANT_TRAIT_TO_OP_DICT,
                                            default_qconfig_list=[self._get_default_qconfig()],
                                            quantizable_layer_nodes=quantizable_layer_nodes,
                                            quantize_outputs=self.quantize_outputs)

        quantization_proposal = solver.run_on_ip_graph(ip_graph)
        multi_config_setup = quantization_proposal.quantizer_setup
        single_config_setup = multi_config_setup.select_first_qconfig_for_each_point()
        finalized_proposal = quantization_proposal.finalize(single_config_setup)
        final_setup = solver.get_final_quantizer_setup(finalized_proposal)
        return final_setup

    def _determine_weight_port(self, node):
        for i, inp in enumerate(node.input_values()):
            if inp.node.get_type_name() == 'Constant':
                return i
        raise RuntimeError

    def _add_weight_quantization_target_point(self, quantization_point: SingleConfigQuantizationPoint) -> None:
        # port_id = quantization_point.insertion_point.input_port_id
        print('FQ', quantization_point.insertion_point.target_node_name)
        weight_quantization_target_point = OVTargetPoint(TargetType.OPERATION_WITH_WEIGHTS,
                                                         quantization_point.insertion_point.target_node_name)
                                                        #  port_id=port_id)
        self._quantization_target_points.add(weight_quantization_target_point)

    def _add_activation_quantization_target_point(self, quantization_point: SingleConfigQuantizationPoint) -> None:
        node_name = quantization_point.insertion_point.target_node_name
        if NNCFGraphNodeType.INPUT_NODE in quantization_point.insertion_point.target_node_name:
            activation_quantization_target_point = OVTargetPoint(TargetType.POST_LAYER_OPERATION, node_name, port_id=0)
        elif quantization_point.insertion_point.input_port_id is not None:
            port_id = quantization_point.insertion_point.input_port_id
            activation_quantization_target_point = OVTargetPoint(TargetType.PRE_LAYER_OPERATION,
                                                                 node_name,
                                                                 port_id)
        else:
            activation_quantization_target_point = OVTargetPoint(TargetType.POST_LAYER_OPERATION, node_name, port_id=0)
        self._quantization_target_points.add(activation_quantization_target_point)

    def get_quantization_target_points(self, model: ov.Model) -> Set[OVTargetPoint]:
        """
        Returns Quantization Target Points.
        In the Compression Pipeline logic NNCF assumes that the compression pipeline works only on the single model.
        So for the optimization purpose if Quantization Target Points were computed before the function returns them,
        otherwise builds NNCFGraph from the 'model',
        finds the quantization setup and processes it to the Set of Quantization Target Points.
        :param model: OpenVINO model, for which Quantization Target Points are being seek.
        :return: Set of Quantization Target Points.
        """
        if self._quantization_target_points:
            return self._quantization_target_points
        quantizer_setup = self._get_quantizer_setup(model)
        for quantization_point in quantizer_setup.quantization_points.values():
            if quantization_point.is_weight_quantization_point():
                self._add_weight_quantization_target_point(quantization_point)
            elif quantization_point.is_activation_quantization_point():
                self._add_activation_quantization_target_point(quantization_point)
            else:
                raise RuntimeError('Incorrect quantization point')
        self._quantization_target_points = sorted(self._quantization_target_points)
        return self._quantization_target_points

    def _apply(self, model: ov.Model, engine: OVEngine,
               statistic_points: StatisticPointsContainer) -> ov.Model:
        model_transformer = self._create_model_transformer(model)
        name_to_node_mapping = model_transformer.name_to_node_mapping
        transformation_layout, transformation_commands = OVTransformationLayout(), []
        quantization_target_points = self.get_quantization_target_points(model)

        for quantization_target_point in quantization_target_points:
            target_node_name = quantization_target_point.target_node_name
            if quantization_target_point.type == TargetType.OPERATION_WITH_WEIGHTS:
                print('target_node_name', target_node_name)
                weight_node = name_to_node_mapping[target_node_name]
                weight_port_id = quantization_target_point.port_id
                print('weight_node', weight_node, weight_port_id)
                # const_weights = weight_node.input_value(weight_port_id).node
                weight_tensor = weight_node.get_vector().reshape(weight_node.shape)
                parameters = calculate_weight_quantizer_parameters(weight_tensor, self.weight_quantizer_config)

                command = OVQuantizerInsertionCommand(quantization_target_point, parameters)
                transformation_commands.append(command)
            elif quantization_target_point.type in [TargetType.PRE_LAYER_OPERATION, TargetType.POST_LAYER_OPERATION]:
                def filter_func(point):
                    return PostTrainingAlgorithms.MinMaxQuantization in point.algorithm_to_tensor_collectors and \
                           point.target_point.type == quantization_target_point.type

                for tensor_collector in statistic_points.get_algo_statistics_for_node(
                        target_node_name,
                        filter_func,
                        PostTrainingAlgorithms.MinMaxQuantization):
                    num_bits = self.activation_quantizer_config.num_bits
                    parameters = calculate_activation_quantizer_parameters(tensor_collector.get_statistics(),
                                                                           num_bits)
                    command = OVQuantizerInsertionCommand(quantization_target_point, parameters)
                    transformation_commands.append(command)
            else:
                raise RuntimeError('Inccorrect type of Quantization Target Point')

        for transformation_command in transformation_commands:
            transformation_layout.register(transformation_command)

        quantized_model = model_transformer.transform(transformation_layout)
        return quantized_model

    def get_statistic_points(self, model: ov.Model) -> StatisticPointsContainer:
        quantization_target_points = self.get_quantization_target_points(model)
        output = StatisticPointsContainer()
        for quantization_target_point in quantization_target_points:
            if quantization_target_point.type in [TargetType.PRE_LAYER_OPERATION, TargetType.POST_LAYER_OPERATION]:
                nncf_logger.debug(
                    'Adding {} Quantization Target Point to the Statistics Points,'
                    ' which outputs will be used for statistics collection'.format(
                        quantization_target_point.target_node_name))
                output.add_statistic_point(StatisticPoint(target_point=quantization_target_point,
                                                          tensor_collector=self.generate_stat_collector(
                                                              self.activation_quantizer_config),
                                                          algorithm=PostTrainingAlgorithms.MinMaxQuantization)
                                           )
            else:
                nncf_logger.debug(
                    'Skipping {} Quantization Target Point, which is used for weights quantization'.format(
                        quantization_target_point))
        return output

    def create_subalgorithms(self, backend: BackendType) -> None:
        return
