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

from nncf.common.graph.patterns import merge_two_types_of_operations
from nncf.common.graph.graph_matching import GraphPattern
# from nncf.experimental.openvino.graph.metatypes.ov_metatypes import OVConvWeightsSubtype
# from nncf.experimental.openvino.graph.metatypes.ov_metatypes import OVConvActsSubtype
# from nncf.experimental.openvino.graph.metatypes.ov_metatypes import OVConvBackpropDataWeightsSubtype
# from nncf.experimental.openvino.graph.metatypes.ov_metatypes import OVConvBackpropDataActsSubtype
# from nncf.experimental.openvino.graph.metatypes.ov_metatypes import OVMatMulWeightsSubtype
# from nncf.experimental.openvino.graph.metatypes.ov_metatypes import OVMatMulActsSubtype

from nncf.experimental.openvino.graph.metatypes.ov_metatypes import OVConvolutionMetatype
from nncf.experimental.openvino.graph.metatypes.ov_metatypes import OVConvolutionBackpropDataMetatype
from nncf.experimental.openvino.graph.metatypes.ov_metatypes import OVMatMulMetatype

from nncf.experimental.openvino.graph.metatypes.ov_metatypes import OVBatchNormMetatype
from nncf.experimental.openvino.graph.metatypes.ov_metatypes import OVReluMetatype
from nncf.experimental.openvino.graph.metatypes.ov_metatypes import OVEluMetatype
from nncf.experimental.openvino.graph.metatypes.ov_metatypes import OVPReluMetatype
from nncf.experimental.openvino.graph.metatypes.ov_metatypes import OVSigmoidMetatype
from nncf.experimental.openvino.graph.metatypes.ov_metatypes import OVHardSigmoidMetatype
from nncf.experimental.openvino.graph.metatypes.ov_metatypes import OVAddLayerMetatype
from nncf.experimental.openvino.graph.metatypes.ov_metatypes import OVMulLayerMetatype
from nncf.experimental.openvino.graph.metatypes.ov_metatypes import OVDivLayerMetatype
from nncf.experimental.openvino.graph.metatypes.ov_metatypes import OVSubLayerMetatype

LINEAR_OPERATIONS = {GraphPattern.METATYPE_ATTR: [OVConvolutionMetatype,
                                                  OVConvolutionBackpropDataMetatype,
                                                  OVMatMulMetatype,
                                                  ],
                     GraphPattern.LABEL_ATTR: 'LINEAR'}

BATCH_NORMALIZATION_OPERATIONS = {GraphPattern.METATYPE_ATTR: [OVBatchNormMetatype],
                                  GraphPattern.LABEL_ATTR: 'BATCH_NORMALIZATION'}

RELU_OPERATIONS = {GraphPattern.METATYPE_ATTR: [OVReluMetatype],
                   GraphPattern.LABEL_ATTR: 'RELU'}

NON_RELU_ACTIVATIONS_OPERATIONS = {GraphPattern.METATYPE_ATTR: [OVEluMetatype,
                                                                OVPReluMetatype,
                                                                OVSigmoidMetatype,
                                                                OVHardSigmoidMetatype,
                                                                ],
                                   GraphPattern.LABEL_ATTR: 'NON_RELU_ACTIVATIONS'}

ATOMIC_ACTIVATIONS_OPERATIONS = merge_two_types_of_operations(RELU_OPERATIONS,
                                                              NON_RELU_ACTIVATIONS_OPERATIONS,
                                                              'ATOMIC_ACTIVATIONS')

ARITHMETIC_OPERATIONS = {GraphPattern.METATYPE_ATTR: [OVAddLayerMetatype,
                                                      OVSubLayerMetatype,
                                                      OVMulLayerMetatype,
                                                      OVDivLayerMetatype,
                                                      ],
                         GraphPattern.LABEL_ATTR: 'ARITHMETIC'}

MATMUL_OPERATIONS = {GraphPattern.METATYPE_ATTR: [OVMatMulMetatype,
                                                  ],
                     GraphPattern.LABEL_ATTR: 'MATMUL'}
