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
from typing import Tuple

import pytest
import numpy as np

from nncf.experimental.ov.engine import OVEngine
from nncf.experimental.ov.statistics.collectors import OVMinMaxStatisticCollector
from nncf.experimental.ov.statistics.aggregator import OVStatisticsAggregator
from nncf.experimental.ov.graph.model_transformer import OVModelTransformer
from nncf.experimental.ov.graph.transformations.layout import OVTransformationLayout
from nncf.experimental.ov.graph.transformations.commands import OVTargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.experimental.ov.graph.transformations.commands import OVOutputInsertionCommand
from nncf.experimental.ov.graph.transformations.commands import OVQuantizerInsertionCommand
from nncf.experimental.ov.tensor import OVNNCFTensor
from nncf.experimental.post_training.api.dataset import Dataset, NNCFData
from nncf.experimental.ov.samplers import OVBatchSampler
from nncf.experimental.post_training.algorithms.algorithm import PostTrainingAlgorithms
from nncf.experimental.post_training.statistics.statistic_point import StatisticPoint
from nncf.experimental.post_training.statistics.statistic_point import StatisticPointsContainer
from nncf.experimental.ov.algorithms.quantization.utils import calculate_activation_quantizer_parameters

from tests.ov.models import OVLinearModel
from tests.ov.models import OVMultiInputOutputModel
from tests.ov.models import OVMultiResultModel

TEST_MODELS = [
    # OVLinearModel,
    # OVMultiInputOutputModel,
    OVMultiResultModel,
]

TARGET_LAYERS = [['Add'], ['MatMul'], ['Add', 'MatMul']]
TARGET_POST_LAYERS_OUTPUT = [['Result_Add.0'], ['Result_MatMul.0'], ['Result_Add.0', 'Result_MatMul.0']]
TARGET_PRE_LAYERS_OUTPUT = [['Result_Reshape.0'], ['Result_Reshape.0'], ['Result_Reshape.0']]

INPUT_SHAPE = [3, 4, 2]
DATASET_SAMPLES = [(np.zeros(INPUT_SHAPE, dtype=np.float32), 0,),
                   (np.ones(INPUT_SHAPE, dtype=np.float32), 1),
                   (100 * np.ones(INPUT_SHAPE, dtype=np.float32), 2)]

REF_OUTPUT_SHAPES = {'Result_matmul': (1, 3, 2, 5), 'Result_add': (1, 3, 2, 4)}


class TestDataset(Dataset):
    def __init__(self, samples: List[Tuple[np.ndarray, int]], input_key: str = "input"):
        super().__init__(shuffle=False)
        self.samples = samples
        self.input_key = input_key

    def __getitem__(self, item) -> NNCFData:
        inputs, targets = self.samples[item]
        return {self.input_key: OVNNCFTensor(inputs), "targets": OVNNCFTensor(targets)}

    def __len__(self):
        return len(self.samples)


@pytest.mark.parametrize('model', TEST_MODELS)
def test_infer_original_model(model):
    model = model().ov_model
    input_data = {inp.get_friendly_name(): OVNNCFTensor(np.zeros(inp.shape, dtype=np.float32))
                  for inp in model.get_parameters()}

    engine = OVEngine()
    engine.set_model(model)
    outputs = engine.infer(input_data)
    for out_name, out in outputs.items():
        assert out.shape == REF_OUTPUT_SHAPES[out_name]


def create_transformed_model(model, target_layers, target_type):
    transformation_layout = OVTransformationLayout()
    for target_layer in target_layers:
        target_point = OVTargetPoint(target_type, target_layer, port_id=0)
        command = OVOutputInsertionCommand(target_point)
        transformation_layout.register(command)

    model_transformer = OVModelTransformer(model.ov_model)
    transformed_model = model_transformer.transform(transformation_layout)
    return transformed_model


def get_extra_outputs(original_model, transformed_model):
    extra_outputs = set()
    for out in transformed_model.get_results():
        extra_outputs.add(out.get_friendly_name())

    for out in original_model.get_results():
        extra_outputs.remove(out.get_friendly_name())

    return extra_outputs


@pytest.mark.parametrize('target_layers, target_layer_outputs', zip(TARGET_LAYERS, TARGET_PRE_LAYERS_OUTPUT))
def test_output_insertion_pre_layer(target_layers, target_layer_outputs):
    model = OVMultiResultModel()
    transformed_model = create_transformed_model(model, target_layers, TargetType.PRE_LAYER_OPERATION)
    extra_outputs = get_extra_outputs(model.ov_model, transformed_model)

    assert len(extra_outputs) == len(target_layer_outputs)
    for out_name in extra_outputs:
        assert out_name in target_layer_outputs


@pytest.mark.parametrize('target_layers, target_layer_outputs', zip(TARGET_LAYERS, TARGET_POST_LAYERS_OUTPUT))
def test_output_insertion_post_layer(target_layers, target_layer_outputs):
    model = OVMultiResultModel()
    transformed_model = create_transformed_model(model, target_layers, TargetType.POST_LAYER_OPERATION)
    extra_outputs = get_extra_outputs(model.ov_model, transformed_model)

    assert len(extra_outputs) == len(target_layer_outputs)
    for out_name in extra_outputs:
        assert out_name in target_layer_outputs

def create_quantized_model(model, statistics_aggregator, target_points):
    tensor_collector = OVMinMaxStatisticCollector(use_abs_max=True, reduction_shape=None, num_samples=3)
    statistic_points = StatisticPointsContainer()
    transformation_layout = OVTransformationLayout()

    for target_point in target_points:
        stat_point = StatisticPoint(target_point=target_point,
                                    tensor_collector=tensor_collector,
                                    algorithm=PostTrainingAlgorithms.MinMaxQuantization)
        statistic_points.add_statistic_point(stat_point)
    statistics_aggregator.register_stastistic_points(statistic_points)
    statistics_aggregator.collect_statistics(model)

    for algo_stat_points in statistic_points.values():
        for statistic_point in algo_stat_points:
            for tensor_collector in statistic_point.algorithm_to_tensor_collectors[PostTrainingAlgorithms.MinMaxQuantization]:
                parameters = calculate_activation_quantizer_parameters(tensor_collector.get_statistics(), num_bits=8)
                transformation_commands = OVQuantizerInsertionCommand(statistic_point.target_point, parameters)
                transformation_layout.register(transformation_commands)

    model_transformer = OVModelTransformer(model)
    quantized_model = model_transformer.transform(transformation_layout)
    return quantized_model


@pytest.mark.parametrize('target_layers', TARGET_LAYERS)
def test_fq_insertion_pre_layer(target_layers):
    model = OVMultiResultModel()

    engine = OVEngine()
    dataset = TestDataset(DATASET_SAMPLES)
    statistics_aggregator = OVStatisticsAggregator(engine, dataset)

    target_type = TargetType.PRE_LAYER_OPERATION
    port_id = 0
    target_points = [OVTargetPoint(target_type, op, port_id=port_id) for op in target_layers]

    quantized_model = create_quantized_model(model.ov_model, statistics_aggregator, target_points)

    for op in quantized_model.get_ops():
        if op.get_friendly_name() in target_layers:
            inp_node = op.input_value(port_id).get_node()
            assert inp_node.get_type_name() == 'FakeQuantize'


@pytest.mark.parametrize('target_layers', TARGET_LAYERS)
def test_fq_insertion_post_layer(target_layers):
    model = OVMultiResultModel()

    engine = OVEngine()
    dataset = TestDataset(DATASET_SAMPLES)
    statistics_aggregator = OVStatisticsAggregator(engine, dataset)

    target_type = TargetType.POST_LAYER_OPERATION
    port_id = 0
    target_points = [OVTargetPoint(target_type, op, port_id=port_id) for op in target_layers]

    quantized_model = create_quantized_model(model.ov_model, statistics_aggregator, target_points)

    for op in quantized_model.get_ops():
        if op.get_friendly_name() in target_layers:
            out_nodes = op.output(port_id).get_target_inputs()
            for out_node in out_nodes:
                assert out_node.get_type_name() == 'FakeQuantize'


# @pytest.mark.parametrize('target_layers, target_layer_outputs', zip(TARGET_LAYERS, TARGET_LAYERS_OUTPUT))
# def test_compute_statistics(target_layers, target_layer_outputs):
#     model = OVMultiResultModel()
#     engine = OVEngine()
#     dataset = TestDataset(DATASET_SAMPLES)

#     statistic_points = StatisticPointsContainer()
#     tensor_collector = OVMinMaxStatisticCollector(use_abs_max=True, reduction_shape=None, num_samples=2)
#     for target_layer in target_layers:
#         target_point = OVTargetPoint(TargetType.PRE_LAYER_OPERATION, target_layer)
#         statistic_points.add_statistic_point(StatisticPoint(target_point=target_point,
#                                                             tensor_collector=tensor_collector,
#                                                             algorithm=PostTrainingAlgorithms.MinMaxQuantization))

#     statistics_aggregator = OVStatisticsAggregator(engine, dataset)
#     statistics_aggregator.register_stastistic_points(statistic_points)
#     statistics_aggregator.collect_statistics(model.ov_model)
#     print('statistics_aggregator.statistic_points', statistics_aggregator.statistic_points)

#     def filter_func(point):
#         return PostTrainingAlgorithms.MinMaxQuantization in point.algorithm_to_tensor_collectors

#     for target_layer in target_layers:
#         print('target_layer', target_layer)
#         for tensor_collector in statistics_aggregator.statistic_points.iter_through_algorithm_tensor_collectors_in_target_node(
#                             target_layer, filter_func, PostTrainingAlgorithms.MinMaxQuantization):
#             statistics = tensor_collector.get_statistics()
#             input_low = statistics.min_values
#             input_high = statistics.max_values
#             print('low', input_low)
#             print('high', input_high)



    # sampler = OVBatchSampler(dataset)

    # engine.set_model(transformed_model)
    # engine.set_sampler(sampler)
    # engine.compute_statistics(statistic_points)


    # statistic_points = StatisticPointsContainer()
    # target_point = OVTargetPoint(TargetType.POST_LAYER_OPERATION, target_layers[0], port_id=0)
    # tensor_collector = OVMinMaxStatisticCollector(use_abs_max=True, reduction_shape=None, num_samples=2)
    # statistic_points.add_statistic_point(StatisticPoint(target_point=target_point, tensor_collector=tensor_collector, algorithm=PostTrainingAlgorithms.MinMaxQuantization))

    # engine = OVEngine()
    # dataset = TestDataset(DATASET_SAMPLES)
    # statistics_aggregator = OVStatisticsAggregator(engine, dataset)
    # statistics_aggregator.register_stastistic_points(statistic_points)
    # statistics_aggregator.collect_statistics(model.ov_model)

#  target_node_name = target_point.target_node_name

#     statistics_aggregator.statistic_points


# @pytest.mark.parametrize(("model_creator_func"),
#                          [TEST_MODELS])
# def test_stats_collector(model_creator_func, ref_metatypes):
#     model = model_creator_func()

#     # axes = (0, 2, 3) if quantizer_config.per_channel else None
#     axes = None
#     stat_collector = OVMinMaxStatisticCollector(use_abs_max=True,
#                                                 reduction_shape=axes,
#                                                 num_samples=10)


#     engine = OVEngine()
#     dataset = create_imagenet_torch_dataset(
#         dataset_path, input_name=input_name,
#         input_shape=input_shape, batch_size=batch_size, shuffle=shuffle)
#     statistics_aggregator = OVStatisticsAggregator(engine, dataset)

#     for algorithm in self.algorithms:
#         statistic_points = algorithm.get_statistic_points(modified_model)
#         statistics_aggregator.register_stastistic_points(statistic_points)

#     statistics_aggregator.collect_statistics(modified_model)

#     QUANTIZED_NODES = ["MatMul", "Add"]


# def get_statistic_points(model, quantized_types):
#     statistic_points = []
#     for op in model.get_ops():
#         # node_name = op.get_friendly_name()
#         if op.get_type_name() in quantized_types:
#             for inp in op.inputs():
#                 stats[node_name] = {}
#                 input_node = inp.get_source_output().node
#                 if input_node.get_type_name() == 'Constant':
#                     const_data = input_node.get_vector().reshape(input_node.shape)
#                     stats[node_name][inp] = min_per_tensor(const_data)
#                 else:
#                     index = inp.get_index()
#                     results.append(opset9.result(inp.get_node().output(index), name=f'Result_{node_name}_{index}'))

#     stats_model = ov.Model(results, model.get_parameters())