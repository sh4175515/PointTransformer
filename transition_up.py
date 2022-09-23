# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
PointTransformer Transition Up blocks
"""

import mindspore.nn as nn
import mindspore.ops as ops
from mindvision.ms3d.models.blocks.pointnet2_fp import PointNetFeaturePropagation

class TransitionUp(nn.Cell):
    """
    TransitionUp Block
    """
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Cell):
            def __init__(self):
                super(SwapAxes, self).__init__()
                self.random = ops.Transpose()

            def construct(self, x):
                new_features = self.random(x, (0, 2, 1))
                return new_features

        super(TransitionUp, self).__init__()
        self.random = ops.Transpose()
        self.f1 = nn.Dense(dim1, dim_out)
        self.Swap = SwapAxes()
        self.BN = nn.BatchNorm2d(dim_out, momentum=0.1, affine=True)
        self.relu = nn.ReLU()
        self.f2 = nn.Dense(dim2, dim_out)
        self.fp = PointNetFeaturePropagation(-1, [])
        self.transpose = ops.Transpose()
        #self.feats1 = self.relu(ops.Squeeze(-1)(self.BN(ops.ExpandDims()(self.Swap(self.f1()), -1))))
        #self.feats2 = self.relu(ops.Squeeze(-1)(self.BN(ops.ExpandDims()(self.Swap(self.f2()), -1))))

    def construct(self, xyz1, point1, xyz2, point2):
        """
        TransitionUp Block
        """
        #feats1 = self.feats1(point1)
        #feats2 = self.feats2(point2)
        feats1 = self.relu(ops.Squeeze(-1)(self.BN(ops.ExpandDims()(self.Swap(self.f1(point1)), -1))))
        feats2 = self.relu(ops.Squeeze(-1)(self.BN(ops.ExpandDims()(self.Swap(self.f2(point2)), -1))))
        mind = self.fp(xyz2, xyz1, None, feats1)
        return feats2 + mind

