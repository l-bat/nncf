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

# from nncf.api.compression import ModelType


class Backend:
    """
    A backend determine original framework based on provided model.

    :param framework: string with original framework name.
    :param preferable_backend: string with one of the installed frameworks.
    """
    __framework = None
    __preferable_backend = None

    @classmethod
    def get(cls):
        if cls.__framework is None:
            raise RuntimeError('Backend is not initialized')
        return cls.__framework

    @classmethod
    def set_preferable_backend(cls, backend_name):
        cls.__preferable_backend = backend_name

    @classmethod
    def get_preferable_backend(cls):
        if cls.__preferable_backend is None:
            raise RuntimeError('Preferable backend is not set')
        return cls.__preferable_backend

    @classmethod
    # def init(cls, model: ModelType):
    def init(cls, model):
        try:
            import torch
        except ImportError:
            torch = None

        try:
            import tensorflow as tf
        except ImportError:
            tf = None

        if torch and isinstance(model, torch.nn.Module):
            cls.__framework = 'Torch'
        elif tf and isinstance(model, tf.keras.Model):
            cls.__framework = 'TensorFlow'
        else:
            raise RuntimeError('Could not determine the model framework.')
