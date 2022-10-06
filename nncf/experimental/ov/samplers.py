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
from typing import Union
from collections import defaultdict

from nncf.common.utils.logger import logger as nncf_logger
from nncf.experimental.ov.statistics.collectors import OVNNCFCollectorTensorProcessor

from nncf.experimental.post_training.api.dataset import Dataset, NNCFData

from nncf.experimental.post_training.samplers import BatchSampler
from nncf.experimental.post_training.samplers import RandomBatchSampler


def _post_process(nncf_data_list: List[NNCFData], stack_tensors: bool = True) -> NNCFData:
    outputs = defaultdict(list)

    for nncf_data in nncf_data_list:
        for k, v in nncf_data.items():
            outputs[k] += [v]

    for k in outputs:
        if stack_tensors:
            outputs[k] = OVNNCFCollectorTensorProcessor.stack(outputs[k])
        elif not stack_tensors and len(outputs[k]) == 1:
            outputs[k] = outputs[k][0]
        else:
            raise RuntimeError(f"{k} has more than one non-stacked tensor.")

    return outputs


class OVBatchSampler(BatchSampler):
    def form_batch(self, start_i: int, end_i: int) -> NNCFData:
        return _post_process(
            nncf_data_list=[self.dataset[i] for i in range(start_i, end_i)],
            stack_tensors=self.has_batch_dim
        )


class OVRandomBatchSampler(RandomBatchSampler):
    def form_batch(self, start_i: int, end_i: int) -> NNCFData:
        return _post_process(
            nncf_data_list=[self.dataset[self.random_permutated_indices[i]]
                            for i in range(start_i, end_i)],
            stack_tensors=self.has_batch_dim
        )


def create_ov_sampler(dataset: Dataset,
                      sample_indices: int) -> Union[OVBatchSampler, OVRandomBatchSampler]:
    if dataset.shuffle:
        nncf_logger.info('Using Shuffled dataset')
        return OVRandomBatchSampler(dataset, sample_indices=sample_indices)
    nncf_logger.info('Using Non-Shuffled dataset')
    return OVBatchSampler(dataset, sample_indices=sample_indices)
