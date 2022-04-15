import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetFeature(nn.Module):
    def __init__(self):
        super(PointNetFeature, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.gn1 = nn.GroupNorm(1, 64)
        self.gn2 = nn.GroupNorm(1, 128)

        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.gn3 = nn.GroupNorm(1, 256)
        self.gn4 = nn.GroupNorm(1, 512)

    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.relu(self.gn3(self.conv3(x)))
        x = F.relu(self.gn4(self.conv4(x)))
        x = torch.max(x, dim=-1)[0]
        return x


class LinearPrediction(nn.Module):
    def __init__(self, output_dim):
        super(LinearPrediction, self).__init__()
        self.layer1 = nn.Linear(1024, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, output_dim)

        self.gn1 = nn.GroupNorm(1, 512)
        self.gn2 = nn.GroupNorm(1, 512)

    def forward(self, x):
        x = F.relu(self.gn1(self.layer1(x)))
        x = F.relu(self.gn2(self.layer2(x)))
        x = self.layer3(x)
        return x


class JointPrediction(nn.Module):
    def __init__(self):
        super(JointPrediction, self).__init__()

        self.pc_feature = PointNetFeature()
        self.joint_predictor = LinearPrediction(output_dim=10)

    def forward(self, pc_start, pc_end):
        pc_start_feature = self.pc_feature(pc_start)
        pc_end_feature = self.pc_feature(pc_end)
        x = torch.cat([pc_start_feature, pc_end_feature], dim=1)

        screw = self.joint_predictor(x)
        screw[:, 0] = torch.sigmoid(screw[:, 0])

        return screw
