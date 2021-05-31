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


def create_compressed_model(model, config, *args):
    Backend.init(model)
    if Backend.get() == 'Torch':
         from nncf.torch.model_creation import create_compressed_model \
              as create_compressed_model_pt
         create_compressed_model_fn = create_compressed_model_pt(model, config, *args)
    elif Backend.get() == 'TensorFlow':
        from beta.nncf.tensorflow.helpers.model_creation import create_compressed_model  \
            as create_compressed_model_tf
        create_compressed_model_fn = create_compressed_model_tf(model, config, *args)
    return create_compressed_model_fn
