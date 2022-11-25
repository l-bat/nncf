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

from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.utils.logger import logger as nncf_logger
from nncf.experimental.post_training.api.dataset import Dataset
from nncf.experimental.post_training.statistics.aggregator import StatisticsAggregator
from nncf.experimental.post_training.api.sampler import Sampler
from nncf.experimental.openvino.samplers import OVBatchSampler
from nncf.experimental.openvino.samplers import OVRandomBatchSampler
from nncf.experimental.openvino.engine import OVNativeEngine
from nncf.experimental.post_training.statistics.statistic_point import StatisticPointsContainer
from nncf.experimental.openvino.graph.transformations.commands import OVOutputInsertionCommand


class OVStatisticsAggregator(StatisticsAggregator):
    def __init__(self, engine: OVNativeEngine, dataset: Dataset):
        super().__init__(engine, dataset)

    def _create_sampler(self, dataset: Dataset,
                        sample_indices: int) -> Sampler:
        if dataset.shuffle:
            nncf_logger.info('Using Shuffled dataset')
            return OVRandomBatchSampler(dataset, sample_indices=sample_indices)
        nncf_logger.info('Using Non-Shuffled dataset')
        return OVBatchSampler(dataset, sample_indices=sample_indices)

    def _get_transformation_layout_extra_outputs(
            self,
            statistic_points: StatisticPointsContainer) -> TransformationLayout:
        transformation_layout = TransformationLayout()
        transformation_commands = []
        for _statistic_points in statistic_points.values():
            for _statistic_point in _statistic_points:
                transformation_commands.append(OVOutputInsertionCommand(_statistic_point.target_point))

        for transformation_command in transformation_commands:
            transformation_layout.register(transformation_command)

        return transformation_layout
