import os
import numpy as np

import torch
import torch.utils.data as data


class ArticulatedDataset(data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.ids = os.listdir(data_path)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        file_data = np.load(self.data_path + self.ids[idx])

        # pc_start = file_data['pc_start']
        # pc_seg_start = file_data['pc_seg_start'].astype(int)
        # pc_end = file_data['pc_end']
        # pc_seg_end = file_data['pc_seg_end'].astype(int)

        # pc_start = np.concatenate((pc_start, np.expand_dims(pc_seg_start, axis=1)), axis=1)
        # pc_end = np.concatenate((pc_end, np.expand_dims(pc_seg_end, axis=1)), axis=1)

        pc_start = file_data['pc_start'][file_data['pc_seg_start'] == True]
        pc_end = file_data['pc_end'][file_data['pc_seg_end'] == True]

        joint_type = file_data['joint_type'].astype(np.int32)
        if (float(file_data['state_end'])-float(file_data['state_start'])) >= 0:
            screw = np.concatenate((file_data['screw_axis'], file_data['screw_moment']), axis=0)
        else:
            screw = np.concatenate((-file_data['screw_axis'], -file_data['screw_moment']), axis=0)

        pc_start = pc_start.astype(np.float32)
        pc_end = pc_end.astype(np.float32)
        joint_type = joint_type.astype(np.float32)
        screw = screw.astype(np.float32)

        return np.transpose(pc_start), np.transpose(pc_end), joint_type, screw
