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

from typing import TypeVar

import openvino.runtime as ov

from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.transformations.commands import TargetType
from nncf.openvino.graph.transformations.layout import OVTransformationLayout
from nncf.openvino.graph.transformations.commands import OVQuantizerInsertionCommand
from nncf.openvino.graph.transformations.commands import OVOutputInsertionCommand

ModelType = TypeVar('ModelType')


# pylint: disable=no-member

class OVModelTransformer(ModelTransformer):
    def __init__(self, model: ov.Model):
        super().__init__(model)
        self.transformed_model = model.clone()
        self.quantizer_insertion_commands = []  # type: List[OVQuantizerInsertionCommand]
        self.output_insertion_commands = []  # type: List[OVOutputInsertionCommand]
        self.name_to_node_mapping = {op.get_friendly_name(): op for op in self.transformed_model.get_ops()}

    def transform(self, transformation_layout: OVTransformationLayout) -> ov.Model:
        for transformation in transformation_layout.transformations:
            if isinstance(transformation, OVQuantizerInsertionCommand):
                self._add_quantizer_insertion_transformation(transformation)
            elif isinstance(transformation, OVOutputInsertionCommand):
                self._add_output_transformation(transformation)
        self._apply_transformations()
        return self.transformed_model

    def _add_quantizer_insertion_transformation(self, transformation: OVQuantizerInsertionCommand) -> None:
        self.quantizer_insertion_commands.append(transformation)

    def _add_output_transformation(self, transformation: OVOutputInsertionCommand) -> None:
        self.output_insertion_commands.append(transformation)

    def _apply_transformations(self) -> None:
        if self.quantizer_insertion_commands:
            self._apply_quantizer_insertion_transformations()
            self.quantizer_insertion_commands = []
        if self.output_insertion_commands:
            self._apply_outputs_transformations()
            self.output_insertion_commands = []

    def _apply_outputs_transformations(self):
        extra_model_outputs = []
        for transformation in self.output_insertion_commands:
            node_name = transformation.target_point.target_node_name
            node = self.name_to_node_mapping[node_name]
            port_id = transformation.target_point.port_id
            if transformation.target_point.type == TargetType.POST_LAYER_OPERATION:
                output = node.output(port_id)
            elif transformation.target_point.type == TargetType.PRE_LAYER_OPERATION:
                output = node.input_value(port_id)
            else:
                raise RuntimeError

            output_name = output.get_node().get_friendly_name()
            result = ov.opset9.result(output, name=f'Result_{output_name}.{port_id}')
            extra_model_outputs.append(result)

        model_outputs = self.transformed_model.get_results() + extra_model_outputs
        self.transformed_model = ov.Model(model_outputs, self.transformed_model.get_parameters())

    def _apply_quantizer_insertion_transformations(self) -> None:
        # TODO: optimize: could be insertion of quantizers done in one operations
        for transformation in self.quantizer_insertion_commands:
            self._insert_fake_quantize_op(transformation)

    def _insert_fake_quantize_op(self, transformation: OVQuantizerInsertionCommand) -> None:
        fq_params = transformation.quantizer_parameters
        input_low = fq_params.input_low
        input_high = fq_params.input_high
        output_low = fq_params.output_low
        output_high = fq_params.output_high
        levels = fq_params.levels

        # TODO: Add FQ name f'FakeQuantize_{transformation.target_point.target_node_name}.{port_id}'
        if transformation.target_point.type == TargetType.PRE_LAYER_OPERATION:
            target_node = self.name_to_node_mapping[transformation.target_point.target_node_name]
            port_id = transformation.target_point.port_id
            inp_node = target_node.input(port_id)
            input_node_output = inp_node.get_source_output()
            fq = ov.opset9.fake_quantize(input_node_output, input_low, input_high, output_low, output_high, levels)
            inp_node.replace_source_output(fq.output(0))
        elif transformation.target_point.type == TargetType.POST_LAYER_OPERATION:
            target_node = self.name_to_node_mapping[transformation.target_point.target_node_name]
            port_id = transformation.target_point.port_id
            output = target_node.output(port_id)
            fq = ov.opset9.fake_quantize(output, input_low, input_high, output_low, output_high, levels)
            for inp_node in output.get_target_inputs():
                inp_node.replace_source_output(fq.output(0))
        else:
            raise RuntimeError