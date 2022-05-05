import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv1d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv1d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class PointNetFeature(nn.Module):
    def __init__(self):
        super(PointNetFeature, self).__init__()
        self.conv1 = Conv1d(3, 64, 1)
        self.conv2 = Conv1d(64, 128, 1)
        self.gn1 = nn.GroupNorm(32, 64)
        self.gn2 = nn.GroupNorm(32, 128)

        self.conv3 = Conv1d(128, 256, 1)
        self.conv4 = Conv1d(256, 1024, 1)
        self.gn3 = nn.GroupNorm(32, 256)
        self.gn4 = nn.GroupNorm(32, 1024)

    def forward(self, x):
        x = F.leaky_relu(self.gn1(self.conv1(x)))
        m = F.leaky_relu(self.gn2(self.conv2(x)))

        x = F.leaky_relu(self.gn3(self.conv3(m)))
        x = F.leaky_relu(self.gn4(self.conv4(x)))
        x = torch.max(x, dim=-1, keepdim=True)[0]
        return x, m


class PointNetSegment(nn.Module):
    def __init__(self):
        super(PointNetSegment, self).__init__()
        self.conv1 = Conv1d(1024 * 2 + 128, 1024, 1)
        self.conv2 = Conv1d(1024, 512, 1)
        self.conv3 = Conv1d(512, 256, 1)
        self.conv4 = nn.Conv1d(256, 1, 1)
        self.gn1 = nn.GroupNorm(32, 1024)
        self.gn2 = nn.GroupNorm(32, 512)
        self.gn3 = nn.GroupNorm(32, 256)

    def forward(self, x, feat):
        x = torch.cat([feat.repeat(1, 1, x.size()[2]), x], dim=1)
        x = F.leaky_relu(self.gn1(self.conv1(x)))
        x = F.leaky_relu(self.gn2(self.conv2(x)))
        x = F.leaky_relu(self.gn3(self.conv3(x)))
        x = self.conv4(x)
        return x


class LinearPrediction(nn.Module):
    def __init__(self, output_dim):
        super(LinearPrediction, self).__init__()
        self.layer1 = Conv1d(1024 * 2, 1024, 1)
        self.layer2 = Conv1d(1024, 512, 1)
        self.layer3 = nn.Conv1d(512, output_dim, 1)
        self.gn1 = nn.GroupNorm(32, 1024)
        self.gn2 = nn.GroupNorm(32, 512)

    def forward(self, x):
        x = F.leaky_relu(self.gn1(self.layer1(x)))
        x = F.leaky_relu(self.gn2(self.layer2(x)))
        x = self.layer3(x)
        x = x[:, :, 0]
        return x


class JointPrediction(nn.Module):
    def __init__(self):
        super(JointPrediction, self).__init__()

        self.pc_feature = PointNetFeature()
        self.pc_segment = PointNetSegment()
        self.joint_predictor = LinearPrediction(output_dim=10)

    def forward(self, pc_start, pc_end):
        pc_start_feature, m = self.pc_feature(pc_start)
        pc_end_feature, _ = self.pc_feature(pc_end)
        combined_feature = torch.cat([pc_start_feature, pc_end_feature], dim=1)

        segment = self.pc_segment(m, combined_feature)
        joint = self.joint_predictor(combined_feature)

        return segment, joint
