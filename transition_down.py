
"""
    TransitionDown Module
"""

import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from pointnet2_util import sample_and_group_all, farthest_point_sample,index_points

def Index_points(points, idx):
    raw_size = idx.shape
    reshape = ops.Reshape()
    idx = reshape(idx, (raw_size[0], -1))
    expand = ops.BroadcastTo((-1, -1, points.shape[-1]))
    output = expand(idx[..., None])
    res = ops.GatherD()(points, 1, output)
    # res_output = res(points, 1, output)
    res_ = reshape(res, (*raw_size, -1))
    return res_

def Square_distance(src, dst):
    out = (src[:, :, None] - dst[:, None]) ** 2
    sum = ops.ReduceSum()
    return sum(out, -1)


def sample_and_group(npoint, radius, nsample, xyz, points, knn=True):
    b, N, c = xyz.shape
    s = npoint
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    sort = ops.Sort()
    dist = Square_distance(new_xyz, xyz)
    idx = sort(dist)[1][:, :, :nsample]
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(b, s, 1, c)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = ops.Concat(-1)((grouped_xyz_norm, grouped_points))
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points



class PointNet2SetAbstraction(nn.Cell):
    """
    SA_ssg  module.
    Input:
        npoint(int):points sampled in farthest point sampling.
        radius(float):search radius in local region,0~1.
        nsample(int): how many points in each local region.
        in_channel(int): Input characters of points.
        mlp(array):output size for MLP on each point.
        group_all(bool): if True choose  pointnet2_group.SampleGroup.sample_and_group_all
                    if False  choose  pointnet2_group.SampleGroup.sample_and_group

        xyz: input points position data, [B, N, C]
        points: input points data, [B, D, N]
    Return:
        new_xyz: sampled points position data, [B, S, C]
        new_points_concat: sample points feature data, [B, D', S]

    Examples:
        >> l1_xyz= Tensor(np.ones((24, 512, 3)),ms.float32)
        >> l1_points= Tensor(np.ones((24,128, 512)),ms.float32)
        >> sa2 = PointNet2SetAbstraction(npoint=128, radius=0.4, nsample=64,
                                        in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)

        >> l2_xyz, l2_points = sa2.construct(l1_xyz, l1_points)
        >> print(l2_xyz.shape, l2_points.shape)

    """

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, knn):
        super(PointNet2SetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.knn = knn
        self.mlp_convs = nn.CellList()
        self.mlp_bns = nn.CellList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, has_bias=True, weight_init='HeUniform'))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel, momentum=0.1, ))
            last_channel = out_channel

        self.relu = ops.ReLU()
        self.transpose = ops.Transpose()
        self.reduce_max = ops.ArgMaxWithValue(axis=2, keep_dims=False)

    def construct(self, xyz, points):
        """SA construct"""

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, knn=self.knn)

        new_points = self.transpose(new_points, (0, 3, 2, 1))
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = self.relu(bn(conv(new_points)))
        new_points = self.reduce_max(new_points)[1]
        new_points = self.transpose(new_points, (0, 2, 1))

        return new_xyz, new_points


class Transitiondown(nn.Cell):
    """
    TransitionDown Block

    Input:
        xyz: input points position data, [B, C, N]
        points: input points data, [B, D, N]

    Return:
        new_xyz: sampled points position data, [B, C, S]
        new_points_concat: sample points feature data, [B, D', S]
    """

    def __init__(self, k, nneighbor, channels):
        super(Transitiondown, self).__init__()
        self.sa = PointNet2SetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)

    def construct(self, xyz, points):
        """
        TransitionDown construct
        """
        after_sa = self.sa(xyz, points)
        return after_sa

