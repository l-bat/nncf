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

from nncf.common.quantization.quantizer_propagation.structs import QuantizationTrait
from nncf.openvino.graph.metatypes.ov_metatypes import OVConvolutionMetatype
from nncf.openvino.graph.metatypes.ov_metatypes import OVConvolutionBackpropDataMetatype
from nncf.openvino.graph.metatypes.ov_metatypes import OVMatMulMetatype
from nncf.openvino.graph.metatypes.ov_metatypes import OVSigmoidMetatype
from nncf.openvino.graph.metatypes.ov_metatypes import OVHardSigmoidMetatype
from nncf.openvino.graph.metatypes.ov_metatypes import OVAveragePoolMetatype
from nncf.openvino.graph.metatypes.ov_metatypes import OVAddLayerMetatype
from nncf.openvino.graph.metatypes.ov_metatypes import OVSubMetatype
from nncf.openvino.graph.metatypes.ov_metatypes import OVMulLayerMetatype
from nncf.openvino.graph.metatypes.ov_metatypes import OVConcatMetatype
from nncf.openvino.graph.metatypes.ov_metatypes import OVBatchNormMetatype
from nncf.openvino.graph.metatypes.ov_metatypes import OVInterpolateMetatype
from nncf.openvino.graph.metatypes.ov_metatypes import OVSoftmaxMetatype
from nncf.openvino.graph.metatypes.ov_metatypes import OVExpMetatype

from nncf.common.graph.operator_metatypes import UnknownMetatype

DEFAULT_OV_QUANT_TRAIT_TO_OP_DICT = {
    QuantizationTrait.INPUTS_QUANTIZABLE: [
        OVConvolutionMetatype,
        OVConvolutionBackpropDataMetatype,
        OVMatMulMetatype,
        OVAveragePoolMetatype,
        OVAddLayerMetatype,
        OVSubMetatype,
        OVMulLayerMetatype,
        OVBatchNormMetatype,
        OVHardSigmoidMetatype,
        OVInterpolateMetatype,
    ],
    QuantizationTrait.NON_QUANTIZABLE: [OVSigmoidMetatype,
                                        OVSoftmaxMetatype,
                                        OVExpMetatype,
                                        UnknownMetatype],
    QuantizationTrait.CONCAT: [OVConcatMetatype],
    QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS: []
}
