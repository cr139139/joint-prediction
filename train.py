import argparse
import os
import csv
import time
import random

import torch
import torch.optim as optim
import torch.utils.data

from datasets import ArticulatedDataset
from model import JointPrediction
from losses import degree_error_distance, origin_error_distance

import numpy as np
from visdom import Visdom

vis = Visdom()
line = vis.line(np.arange(10))

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--weight', type=str, default='../data/weights', help='trained weight folder')
parser.add_argument('--model', type=str, default='', help='model path')

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = ArticulatedDataset("../data/cabinet_train_1K/scenes/")
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
valid_dataset = ArticulatedDataset('../data/cabinet_val_50/scenes/')
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)
print(len(train_dataset), len(valid_dataset))

# create weights folder if does not exist
try:
    os.makedirs(opt.weight)
except OSError:
    pass

model = JointPrediction()
if opt.model != '':
    model.load_state_dict(torch.load(opt.model))
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.99 ** epoch)
model.to(device)

num_train_dataset = len(train_dataset)
num_valid_dataset = len(valid_dataset)
batch_size = 25

seg_distance = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4.88))
type_distance = torch.nn.BCELoss()
mse_distance = torch.nn.MSELoss()
seg_distance.to(device)
type_distance.to(device)
mse_distance.to(device)

start_time = time.time()
time_loss = []
best_valid_loss = 10000

for epoch in range(opt.nepoch):
    train_sum_loss = 0
    train_sum_step = 0
    valid_sum_loss = 0
    valid_sum_step = 0
    train_segment_correctness = 0
    valid_segment_correctness = 0
    train_type_correctness = 0
    valid_type_correctness = 0

    # training steps
    for i, data in enumerate(train_dataloader, 0):
        pc_start, pc_end, pc_seg_start, joint_type, joint_screw = data
        pc_start, pc_end, pc_seg_start, joint_type, joint_screw = pc_start.to(device), pc_end.to(device), \
                                                                  pc_seg_start.to(device), \
                                                                  joint_type.to(device), joint_screw.to(device)

        # optimizer.zero_grad()
        model = model.train()
        segment_pred, joint_pred = model(pc_start, pc_end)

        # calculate losses
        segment_loss = seg_distance(segment_pred, pc_seg_start)
        type_loss = type_distance(joint_pred[0, :1], joint_type)
        if joint_type < 0.5:
            orientation_loss = degree_error_distance(joint_pred[:, 1:4], joint_screw[:, :3])
            origin_loss = mse_distance(joint_pred[:, 4:7], joint_screw[:, 3:]) \
                          + origin_error_distance(joint_pred[:, 1:4], joint_pred[:, 4:7],
                                                  joint_screw[:, :3], joint_screw[:, 3:])
        else:
            orientation_loss = degree_error_distance(joint_pred[:, 7:], joint_screw[:, :3])
            origin_loss = 0

        loss = (segment_loss + type_loss + orientation_loss + origin_loss) / batch_size
        loss.backward()

        if (joint_pred[0, :1] >= 0.5).int() == joint_type:
            train_type_correctness += 1

        correctness_sum = torch.sum(((segment_pred >= 0.5).int() == pc_seg_start.int()).int())
        train_segment_correctness += correctness_sum / pc_start.size()[2]

        if i % batch_size == batch_size - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
            optimizer.step()
            optimizer.zero_grad()
            print('[%d: %4d/%4d] train loss: %.3f (seg: %.3f, type: %.3f, ori: %.3f, orig: %.3f, seg acc: %.2f, '
                  'type acc: %.2f) '
                  % (epoch, i + 1, num_train_dataset, loss.item(), segment_loss, type_loss, orientation_loss,
                     origin_loss, train_segment_correctness / batch_size, train_type_correctness / batch_size))

            train_segment_correctness = 0
            train_type_correctness = 0

        # store training loss and step
        train_sum_loss += loss.item() * pc_start.size(0) * batch_size
        train_sum_step += pc_start.size(0)

    # validation steps
    for i, data in enumerate(valid_dataloader, 0):
        pc_start, pc_end, pc_seg_start, joint_type, joint_screw = data
        pc_start, pc_end, pc_seg_start, joint_type, joint_screw = pc_start.to(device), pc_end.to(device), \
                                                                  pc_seg_start.to(device), \
                                                                  joint_type.to(device), joint_screw.to(device)

        model = model.eval()
        segment_pred, joint_pred = model(pc_start, pc_end)

        # calculate losses
        segment_loss = seg_distance(segment_pred, pc_seg_start)
        type_loss = type_distance(joint_pred[0, :1], joint_type)
        if joint_type < 0.5:
            orientation_loss = degree_error_distance(joint_pred[:, 1:4], joint_screw[:, :3])
            origin_loss = mse_distance(joint_pred[:, 4:7], joint_screw[:, 3:]) \
                          + origin_error_distance(joint_pred[:, 1:4], joint_pred[:, 4:7],
                                                  joint_screw[:, :3], joint_screw[:, 3:])
        else:
            orientation_loss = degree_error_distance(joint_pred[:, 7:], joint_screw[:, :3])
            origin_loss = 0

        if (joint_pred[0, :1] >= 0.5).int() == joint_type:
            valid_type_correctness += 1

        correctness_sum = torch.sum(((segment_pred >= 0.5).int() == pc_seg_start.int()).int())
        valid_segment_correctness += correctness_sum / pc_start.size()[2]

        loss = (segment_loss + type_loss + orientation_loss + origin_loss)

        # store validation loss and step
        valid_sum_loss += loss.item() * pc_start.size(0)
        valid_sum_step += pc_start.size(0)

        blue = lambda x: '\033[94m' + x + '\033[0m'
        if i + 1 != num_valid_dataset:
            print_end = '\r'
        else:
            print_end = '\n'
        print('[%d: %4d/%4d] %s loss: %.3f (seg: %.3f, type: %.3f, ori: %.3f, orig: %.3f, seg acc: %.2f, type acc: '
              '%.2f) '
              % (epoch, i + 1, num_valid_dataset, blue('valid'), valid_sum_loss / valid_sum_step,
                 segment_loss, type_loss, orientation_loss, origin_loss, valid_segment_correctness / valid_sum_step,
                 valid_type_correctness / valid_sum_step),
              end=print_end)

    # save a weight with lowest loss
    if best_valid_loss > valid_sum_loss / valid_sum_step:
        best_valid_loss = valid_sum_loss / valid_sum_step
        torch.save(model.state_dict(), '%s/model_best_valid.pth' % opt.weight)

    torch.save(model.state_dict(), '%s/model_last_train.pth' % opt.weight)

    time_loss.append([time.time() - start_time, train_sum_loss / train_sum_step, valid_sum_loss / valid_sum_step])

    time_loss_array = np.array(time_loss)
    vis.line(X=time_loss_array[:, 0],
             Y=time_loss_array[:, 1:],
             win=line,
             opts=dict(legend=["Train", "Valid"]))

    # reduce the learning rate after each epochs
    scheduler.step()

with open('loss_track.csv', 'w', newline='') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(['time', 'train loss', 'valid loss'])
    write.writerows(time_loss)
    f.close()
