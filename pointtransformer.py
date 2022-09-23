"""
PointTransformer Backbone Module

"""
import transformer, transition_down, transition_up
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
import mindspore


class Backbone(nn.Cell):
    def __init__(self):
        super(Backbone, self).__init__()
        npoints = 1024
        nblocks = 4
        nneighbor = 16
        d_points = 22
        self.fc1 = nn.SequentialCell(

            nn.Dense(d_points, 32),
            nn.ReLU(),
            nn.Dense(32, 32)

        )
        self.transpose = ops.Transpose()
        self.transformer1 = transformer.Transformer(32, 512, nneighbor)
        self.transition_downs = nn.CellList()
        self.transformers = nn.CellList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(
                transition_down.Transitiondown(npoints // 4 ** (i + 1), nneighbor,
                                               [channel // 2 + 3, channel, channel]))
            self.transformers.append(transformer.Transformer(channel, 512, nneighbor))
        self.nblocks = nblocks

    def construct(self, x):
        # x=x.squeeze(1)
        # x = self.transpose(x, (0, 2, 1))
        xyz = x[..., :3]

        y = self.fc1(x)
        points = self.transformer1(xyz, y)[0]
        xyz_and_feat = [(xyz, points)]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feat.append((xyz, points))
        return points, xyz_and_feat


class pointtransformercls(nn.Cell):
    def __init__(self):
        super(pointtransformercls, self).__init__()
        self.backbone = Backbone()
        self.fc2 = nn.SequentialCell(
            nn.Dense(32 * 2 ** 4, 256),
            nn.ReLU(),
            nn.Dense(256, 64),
            nn.ReLU(),
            nn.Dense(64, 40)
        )
        self.reducemean = ops.ReduceMean()

    # self.random = ops.transpose()

    def construct(self, x):
        point, _ = self.backbone(x)
        point = self.reducemean(point, 1)
        res = self.fc2(point)
        # res = self.random(res,(1,0))

        return res


class pointtransformerseg(nn.Cell):
    def __init__(self):
        super(pointtransformerseg, self).__init__()
        self.backbone = Backbone()
        npoints, nblocks, nneighbor, n_c, d_points = 1024, 4, 16, 50, 21
        self.fc1 = nn.SequentialCell(
            nn.Dense(32 * 2 ** nblocks, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 32 * 2 ** nblocks)
        )
        self.transformer2 = transformer.Transformer(32 * 2 ** nblocks, 512, nneighbor)
        self.nblocks = nblocks
        self.transition_ups = nn.CellList()
        self.transformers = nn.CellList()
        for i in reversed(range(nblocks)):
            channel = 32 * 2 ** i
            self.transition_ups.append(transition_up.TransitionUp(channel * 2, channel, channel))
            self.transformers.append(transformer.Transformer(channel, 512, nneighbor))
        self.fc3 = nn.SequentialCell(
            nn.Dense(32, 64),
            nn.ReLU(),
            nn.Dense(64, 64),
            nn.ReLU(),
            nn.Dense(64, n_c)
        )
        self.transpore = ops.Transpose()

    def construct(self, x):
        points, xyz_and_feats = self.backbone(x)
        xyz = xyz_and_feats[-1][0]
        points = self.transformer2(xyz, self.fc1(points))[0]
        for i in range(self.nblocks):
            points = self.transition_ups[i](xyz, points, xyz_and_feats[- i - 2][0], xyz_and_feats[- i - 2][1])
            xyz = xyz_and_feats[- i - 2][0]

            points = self.transpore(points, (0, 2, 1))
            points = self.transformers[i](xyz, points)[0]

        return self.fc3(points).view(-1, 50)


if __name__ == "__main__":
    import numpy as np

    C = pointtransformerseg()
    x_dim = mindspore.Tensor(np.random.random((2, 1024, 22)), dtype=mindspore.float32)
    print(C(x_dim).shape)
