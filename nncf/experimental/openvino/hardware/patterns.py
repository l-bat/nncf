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

from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.patterns import GraphPattern

from nncf.experimental.openvino.graph.metatypes.ov_metatypes import OVAddLayerMetatype
from nncf.experimental.openvino.graph.metatypes.ov_metatypes import OVSubLayerMetatype
from nncf.experimental.openvino.graph.metatypes.ov_metatypes import OVMulLayerMetatype


def create_input_preprocessing_pattern() -> GraphPattern:
    pattern = GraphPattern()

    model_input_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_INPUT',
                                             GraphPattern.METATYPE_ATTR: NNCFGraphNodeType.INPUT_NODE})
    add_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                     GraphPattern.METATYPE_ATTR: OVAddLayerMetatype})
    mul_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MUL',
                                     GraphPattern.METATYPE_ATTR: OVMulLayerMetatype})

    pattern.add_edge(model_input_node_1, add_node_1)
    pattern.add_edge(add_node_1, mul_node_1)

    model_input_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_INPUT',
                                             GraphPattern.METATYPE_ATTR: NNCFGraphNodeType.INPUT_NODE})
    mul_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MUL',
                                     GraphPattern.METATYPE_ATTR: OVMulLayerMetatype})
    add_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                     GraphPattern.METATYPE_ATTR: OVAddLayerMetatype})

    pattern.add_edge(model_input_node_2, mul_node_2)
    pattern.add_edge(mul_node_2, add_node_2)

    model_input_node_3 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_INPUT',
                                             GraphPattern.METATYPE_ATTR: NNCFGraphNodeType.INPUT_NODE})
    add_node_3 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                     GraphPattern.METATYPE_ATTR: OVAddLayerMetatype})

    pattern.add_edge(model_input_node_3, add_node_3)

    model_input_node_4 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_INPUT',
                                             GraphPattern.METATYPE_ATTR: NNCFGraphNodeType.INPUT_NODE})
    mul_node_4 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MUL',
                                     GraphPattern.METATYPE_ATTR: OVMulLayerMetatype})

    pattern.add_edge(model_input_node_4, mul_node_4)

    return pattern


def create_scale_shift() -> GraphPattern:
    pattern = GraphPattern()

    model_input_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: '*INPUT_NODE*',
                                             GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE})
    mul_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MUL',
                                     GraphPattern.METATYPE_ATTR: OVMulLayerMetatype})
    add_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                     GraphPattern.METATYPE_ATTR: OVAddLayerMetatype})

    pattern.add_edge(model_input_node_1, mul_node_1)
    pattern.add_edge(mul_node_1, add_node_1)

    model_input_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: '*INPUT_NODE*',
                                             GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE})
    mul_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MUL',
                                     GraphPattern.METATYPE_ATTR: OVMulLayerMetatype})
    add_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SUB',
                                     GraphPattern.METATYPE_ATTR: OVSubLayerMetatype})

    pattern.add_edge(model_input_node_2, mul_node_2)
    pattern.add_edge(mul_node_2, add_node_2)

    return pattern
