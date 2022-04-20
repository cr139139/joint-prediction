import os
import numpy as np

import torch
import torch.utils.data as data


class ArticulatedDataset(data.Dataset):
    def __init__(self, data_path, sample_num_points=16384):
        self.data_path = data_path
        self.ids = os.listdir(data_path)
        self.sample_num_points = sample_num_points

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        file_data = np.load(self.data_path + self.ids[idx])

        pc_start = file_data['pc_start']  # [file_data['pc_seg_start'] == True]
        pc_end = file_data['pc_end']  # [file_data['pc_seg_end'] == True]
        pc_seg_start = file_data['pc_seg_start']

        if pc_start.shape[0] > self.sample_num_points:
            choice_start = np.random.choice(pc_start.shape[0], self.sample_num_points, replace=False)
            pc_start = pc_start[choice_start]
            pc_seg_start = pc_seg_start[choice_start]
        if pc_end.shape[0] > self.sample_num_points:
            choice_end = np.random.choice(pc_end.shape[0], self.sample_num_points, replace=False)
            pc_end = pc_end[choice_end]

        pc_seg_start = np.expand_dims(pc_seg_start, axis=0)

        joint_type = file_data['joint_type'].astype(np.int32)
        if (float(file_data['state_end']) - float(file_data['state_start'])) >= 0:
            screw = np.concatenate((file_data['screw_axis'], file_data['screw_moment']), axis=0)
        else:
            screw = np.concatenate((-file_data['screw_axis'], -file_data['screw_moment']), axis=0)

        pc_start = pc_start.astype(np.float32)
        pc_end = pc_end.astype(np.float32)
        pc_seg_start = pc_seg_start.astype(np.float32)
        joint_type = joint_type.astype(np.float32)
        screw = screw.astype(np.float32)

        return np.transpose(pc_start), np.transpose(pc_end), pc_seg_start, joint_type, screw
