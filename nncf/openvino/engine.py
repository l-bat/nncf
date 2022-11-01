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

import numpy as np
import openvino.runtime as ov

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.definitions import NNCFGraphNodeType

from nncf.experimental.post_training.api.dataset import NNCFData
from nncf.experimental.post_training.api.engine import Engine
from nncf.experimental.post_training.api.sampler import Sampler
from nncf.experimental.post_training.statistics.statistic_point import StatisticPointsContainer
from nncf.openvino.samplers import create_ov_sampler
from nncf.openvino.tensor import OVNNCFTensor


class OVEngine(Engine):
    """
    Engine for OpenVINO backend using OpenVINO runtime to infer the model.
    """

    def __init__(self):
        super().__init__()
        self._inputs_transforms = lambda input_data: input_data.astype(np.float32)
        self.sess = None
        self.input_names = set()
        self.name_to_node_mapping = {}
        self.target_device = 'CPU'

    def get_sampler(self) -> Sampler:
        # TODO (Nikita Malinin): Replace range calling with the max length variable
        return self.sampler if self.sampler else create_ov_sampler(self.dataset, len(self.dataset))

    def set_model(self, model: ov.Model) -> None:
        """
        Creates CompiledModel and InferRequest for the OpenVINO model.

        :param model: ov.Model instance
        """
        super().set_model(model)

        ie = ov.Core()
        self.compiled_model = ie.compile_model(model=model, device_name=self.target_device)

        self.input_names.clear()
        for inp in model.get_parameters():
            self.input_names.add(inp.get_friendly_name())

        self.name_to_node_mapping.clear()
        for op in model.get_ops():
            self.name_to_node_mapping[op.get_friendly_name()] = op

    def infer(self, input_data: NNCFData) -> NNCFData:
        """
        Runs model on the provided input_data via OpenVINO runtime.
        Returns the dictionary of model outputs by node names.

        :param input_data: inputs for the model transformed with the inputs_transforms
        :return output_data: models output after outputs_transforms
        """
        model_outputs = self.compiled_model(
            {k: v.tensor for k, v in input_data.items() if k in self.input_names})

        return {
            out.get_node().get_friendly_name(): OVNNCFTensor(model_outputs[self.compiled_model.output(i)])
            for i, out in enumerate(model_outputs)
        }

    def _find_output_name_to_node_name_mapping(self, statistic_points: StatisticPointsContainer) -> Dict[str, str]:
        output_name_to_node_name = {}
        for node_name, _statistic_points in statistic_points.items():
            for statistic_point in _statistic_points:
                port_id = statistic_point.target_point.port_id
                if statistic_point.target_point.type == TargetType.POST_LAYER_OPERATION:
                    stat_node_name = node_name
                elif statistic_point.target_point.type == TargetType.PRE_LAYER_OPERATION:
                    node = self.name_to_node_mapping[node_name]
                    stat_node_name = node.input_value(port_id).get_node().get_friendly_name()
                else:
                    RuntimeError('The statistics should be collected only from the input of output edges of the node')
                output_name = f'Result_{stat_node_name}.{port_id}'
                output_name_to_node_name[output_name] = output_name_to_node_name.get(output_name, []) + [node_name]
        return output_name_to_node_name

    def _register_statistics(self, outputs: NNCFData, statistic_points: StatisticPointsContainer) -> None:
        """
        Registers 'outputs' tensors from inferred model and register them to the correspondence 'statistic_points'.
        """
        output_name_to_node_name = self._find_output_name_to_node_name_mapping(statistic_points)
        for output_name, output_tensor in outputs.items():
            if output_name in output_name_to_node_name:
                for node_name in output_name_to_node_name[output_name]:
                    for statistic_point in statistic_points[node_name]:
                        statistic_point.register_tensor(output_tensor)
