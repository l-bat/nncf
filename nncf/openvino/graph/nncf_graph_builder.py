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
import openvino.runtime as ov

from nncf.common.graph import NNCFGraph
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.operator_metatypes import UnknownMetatype
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.definitions import MODEL_INPUT_OP_NAME
from nncf.common.graph.definitions import MODEL_OUTPUT_OP_NAME
from nncf.common.graph.operator_metatypes import InputNoopMetatype
from nncf.common.graph.operator_metatypes import OutputNoopMetatype
from nncf.common.utils.logger import logger as nncf_logger

from nncf.openvino.graph.metatypes.ov_metatypes import OV_OPERATION_METATYPES
from nncf.openvino.graph.metatypes.ov_metatypes import OVConstantMetatype
from nncf.openvino.graph.metatypes.ov_metatypes import OVResultMetatype


class GraphConverter:
    """
    Builds the NNCFGraph from an OpenVINO model.
    """

    @staticmethod
    def _is_valid_openvino_metatype(node: ov.Node) -> bool:
        """
        Checks whether the node has the metatype which should be added to the NNCFGraph.
        :param node: Node to be checked.
        :return: True if the metatype is valid and False if not.
        """
        node_type = node.get_type_name()
        metatype = OV_OPERATION_METATYPES.get_operator_metatype_by_op_name(node_type)
        if metatype == OVConstantMetatype:  # We don't need to quantize Constants
            nncf_logger.debug('The metatype is OVConstantMetatype, which in not quantizable. Skipping this node.')
            return False
        if metatype == OVResultMetatype:  # We don't need to add Results to nncf_graph
            nncf_logger.debug('The metatype is OVResultMetatype, which in not quantizable. Skipping this node.')
            return False
        if metatype == UnknownMetatype:
            node_name = node.get_friendly_name()
            nncf_logger.warning(
                'The node with name {} with type {} was mapped to UnknownMetatype,'
                ' which means that there was not registered such NNCF metatype. '
                'Please, Inform the NNCF developers about this message.'.format(
                    node_name, node_type))
        return True

    @staticmethod
    def _add_nncf_input_nodes(model: ov.Model, nncf_graph: NNCFGraph) -> None:
        """
        Adds special NNCF Input nodes to NNCFGraph.
        For all the OpenVINO model inputs, the special NNCF Input node is placed and then corresponding edges are added.
        :param model: OpenVINO model.
        :param nncf_graph: NNCFGraph, in which the new nodes will be added.
        :return: None.
        """
        for i, _input in enumerate(model.get_parameters()):
            input_node = nncf_graph.add_nncf_node(node_name=MODEL_INPUT_OP_NAME + '_' + str(i),
                                                  node_type=NNCFGraphNodeType.INPUT_NODE,
                                                  node_metatype=InputNoopMetatype,
                                                  layer_attributes=None)
            input_node_node_id = input_node.node_id

            to_nodes = []
            for out in _input.outputs():
                to_nodes.extend(out.get_target_inputs())

            tensor_shape = _input.get_output_shape()
            print('tensor_shape', tensor_shape)
            ov_dtype = _input.get_element_type()
            nncf_dtype = GraphConverter.convert_ov_dtype_to_nncf_dtype(ov_dtype)
            output_port_id = 0
            for inp in filter(GraphConverter._is_valid_openvino_metatype, to_nodes):
                to_node_id = nncf_graph.get_node_by_name(inp.get_node().get_friendly_name()).node_id
                input_port_id = inp.get_index()
                nncf_graph.add_edge_between_nncf_nodes(
                    from_node_id=input_node_node_id,
                    to_node_id=to_node_id,
                    tensor_shape=tensor_shape,
                    input_port_id=input_port_id,
                    output_port_id=output_port_id,
                    dtype=nncf_dtype
                )
                output_port_id += 1

    @staticmethod
    def _add_nncf_output_nodes(model: ov.Model, nncf_graph: NNCFGraph) -> None:
        """
        Adds special NNCF Output nodes to NNCFGraph.
        For all the OpenVINO model results, the special NNCF Output node is placed and then corresponding edges are added.
        :param model: OpenVINO model.
        :param nncf_graph: NNCFGraph, in which the new nodes will be added.
        :return: None.
        """
        for i, _output in enumerate(model.get_results()):
            output_node = nncf_graph.add_nncf_node(node_name=MODEL_OUTPUT_OP_NAME + '_' + str(i),
                                                   node_type=NNCFGraphNodeType.OUTPUT_NODE,
                                                   node_metatype=OutputNoopMetatype,
                                                   layer_attributes=None)
            output_node_node_id = output_node.node_id

            from_nodes = _output.input_values()
            output_shape = _output.get_output_shape(0)
            print('output_shape', output_shape)
            raise ""
            ov_dtype = _output.get_element_type()
            nncf_dtype = GraphConverter.convert_ov_dtype_to_nncf_dtype(ov_dtype)
            input_port_id = 0
            for inp in filter(GraphConverter._is_valid_openvino_metatype, from_nodes):
                from_node_id = nncf_graph.get_node_by_name(inp.get_node().get_friendly_name()).node_id
                output_port_id = onnx_graph.get_output_port_id_for_node_before_output(output_name, node)

                nncf_graph.add_edge_between_nncf_nodes(
                    from_node_id=from_node_id,
                    to_node_id=output_node_node_id,
                    tensor_shape=output_shape,
                    input_port_id=input_port_id,
                    output_port_id=output_port_id,
                    dtype=nncf_dtype
                )
                input_port_id += 1

    @staticmethod
    def convert_ov_dtype_to_nncf_dtype(ov_dtype: str) -> Dtype:
        """
        Converts the primitive types from the OpenVINO domain to the NNCF domain.
        :param ov_dtype: OpenVINO primitive typename.
        :return: NNCF primitive type.
        """
        conversion_map = {
            'f16': 'float',
            'f32': 'float',
            'f64': 'float',
            'i16': 'int',
            'i32': 'int',
            'i4': 'int',
            'i64': 'int',
            'i8': 'int',
            'u1': 'int',
            'u16': 'int',
            'u32': 'int',
            'u4': 'int',
            'u64': 'int',
            'u8': 'int',
        }
        return Dtype(conversion_map.get(ov_dtype, 'int'))

    @staticmethod
    def create_nncf_graph(model: ov.Model) -> NNCFGraph:
        """
        Creates NNCFGraph from OpenVINO Model.
        All nodes from model which have valid metatype are added to NNCFGraph.
        Then, corresponding edges are added to the NNCFGraph with shape, type, output and input port ids.
        In the last step, special NNCF Input and Output nodes are added.
        :param model: OpenVINO model.
        :return: NNCFGraph.
        """
        nncf_graph = NNCFGraph()

        visited = set()
        nodes = model.get_parameters()
        while nodes:
            node = nodes[0]
            nodes = nodes[1:]
            if node.get_friendly_name() not in visited and GraphConverter._is_valid_openvino_metatype(node):
                visited.add(node.get_friendly_name())
                ov_dtype = node.get_element_type()
                nncf_dtype = GraphConverter.convert_ov_dtype_to_nncf_dtype(ov_dtype)
                node_type = node.get_type_name()
                metatype = OV_OPERATION_METATYPES.get_operator_metatype_by_op_name(node_type)

                # print('* Add to nncf graph:', node.get_friendly_name())
                nncf_graph.add_nncf_node(node_name=node.get_friendly_name(),
                                        node_type=nncf_dtype,
                                        node_metatype=metatype,
                                        layer_attributes=None)

                for out in node.outputs():
                    for inp in out.get_target_inputs():
                        nodes.append(inp.get_node())

        # for node in nncf_graph.get_all_nodes():
        #     print(' --- node.node_name', node.node_name)

        for op in filter(GraphConverter._is_valid_openvino_metatype, model.get_ordered_ops()):
            in_node_id = nncf_graph.get_node_by_name(op.get_friendly_name()).node_id
            for output_port_id, out in enumerate(op.outputs()):
                for inp in out.get_target_inputs():
                    out_node = inp.get_node()
                    if GraphConverter._is_valid_openvino_metatype(out_node):
                        tensor_shape = list(out.shape)
                        # print('£ Connection:',  op.get_friendly_name(), '->', out_node.get_friendly_name())
                        # print('inp port:', inp.get_index(), 'out port:', output_port_id)
                        output_node_id = nncf_graph.get_node_by_name(out_node.get_friendly_name()).node_id

                        ov_dtype = op.get_element_type().get_type_name()
                        nncf_dtype = GraphConverter.convert_ov_dtype_to_nncf_dtype(ov_dtype)
                        nncf_graph.add_edge_between_nncf_nodes(
                            from_node_id=in_node_id,
                            to_node_id=output_node_id,
                            tensor_shape=tensor_shape,
                            input_port_id=inp.get_index(),
                            output_port_id=output_port_id,
                            dtype=Dtype(nncf_dtype)
                        )

        return nncf_graph
