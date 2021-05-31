"""
 Copyright (c) 2021 Intel Corporation
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

from nncf.common.utils.backend import Backend


def is_depthwise_conv(node):
    if Backend.get() == 'Torch':
        from nncf.torch.pruning.utils import is_depthwise_conv as is_depthwise_conv_pt
        is_depthwise_conv_fn = is_depthwise_conv_pt(node)
    elif Backend.get() == 'TensorFlow':
        from beta.nncf.tensorflow.pruning.utils import is_depthwise_conv as is_depthwise_conv_tf
        is_depthwise_conv_fn = is_depthwise_conv_tf(node)
    return is_depthwise_conv_fn


def is_conv_with_downsampling(node):
    if Backend.get() == 'Torch':
        from nncf.torch.pruning.utils import is_conv_with_downsampling as is_conv_with_downsampling_pt
        is_depthwise_conv_fn = is_conv_with_downsampling_pt(node)
    elif Backend.get() == 'TensorFlow':
        from beta.nncf.tensorflow.pruning.utils import is_conv_with_downsampling as is_conv_with_downsampling_tf
        is_depthwise_conv_fn = is_conv_with_downsampling_tf(node)
    return is_depthwise_conv_fn


def get_module_identifier(node):
    if Backend.get() == 'Torch':
        from nncf.torch.graph.graph import get_module_identifier as get_module_identifier_pt
        get_module_identifier_fn = get_module_identifier_pt(node)
    elif Backend.get() == 'TensorFlow':
        from beta.nncf.tensorflow.graph.utils import get_layer_identifier as get_module_identifier_tf
        get_module_identifier_fn = get_module_identifier_tf(node)
    return get_module_identifier_fn


def should_consider_scope(scope_str, target_scopes, ignored_scopes):
    if Backend.get() == 'Torch':
        from nncf.torch.utils import should_consider_scope as should_consider_scope_pt
        get_module_identifier_fn = should_consider_scope_pt(scope_str, target_scopes, ignored_scopes)
    elif Backend.get() == 'TensorFlow':
        from beta.nncf.tensorflow.utils.scopes_handle import should_consider_scope as should_consider_scope_tf
        get_module_identifier_fn = should_consider_scope_tf(scope_str, target_scopes, ignored_scopes)
    return get_module_identifier_fn
