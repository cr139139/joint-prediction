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

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--weight', type=str, default='weights', help='trained weight folder')
parser.add_argument('--model', type=str, default='', help='model path')

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = ArticulatedDataset("./cabinet_train_1K/scenes/")
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
valid_dataset = ArticulatedDataset('./cabinet_val_50/scenes/')
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
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
model.to(device)

num_batch = len(train_dataset)
batch_size = 25

typeloss = torch.nn.BCELoss()
mseloss = torch.nn.MSELoss()
typeloss.to(device)
mseloss.to(device)

start_time = time.time()
time_loss = []
valid_loss = 10000

for epoch in range(opt.nepoch):
    sum_loss = 0
    sum_step = 0
    valid_sum_loss = 0
    valid_sum_step = 0
    type_correctness = 0

    # training steps
    for i, data in enumerate(train_dataloader, 0):
        pc_start, pc_end, joint_type, joint_screw = data
        pc_start, pc_end, joint_type, joint_screw = pc_start.to(device), pc_end.to(device), \
                                                    joint_type.to(device), joint_screw.to(device)

        # optimizer.zero_grad()
        model = model.train()
        prediction = model(pc_start, pc_end)

        # calculate losses
        type_loss = typeloss(prediction[0, :1], joint_type)

        if joint_type < 0.5:
            screw_orientation_loss = degree_error_distance(prediction[:, 1:4], joint_screw[:, :3])
            screw_origin_loss = mseloss(prediction[:, 4:7], joint_screw[:, 3:]) \
                                + origin_error_distance(prediction[:, 1:4], prediction[:, 4:7],
                                                        joint_screw[:, :3], joint_screw[:, 3:])

        else:
            screw_orientation_loss = degree_error_distance(prediction[:, 7:], joint_screw[:, :3])
            screw_origin_loss = 0

        if (prediction[0, :1] >= 0.5).int() == joint_type:
            type_correctness += 1
        loss = (type_loss + screw_orientation_loss + screw_origin_loss) / batch_size
        loss.backward()

        if i % batch_size == batch_size - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
            optimizer.step()
            optimizer.zero_grad()
            print('[%d: %4d/%d] train loss: %.3f, type loss: %.3f, axis loss: %.3f, origin loss: %.3f type accuracy: '
                  '%.2f'
                  % (epoch, i + 1, num_batch, loss.item(), type_loss, screw_orientation_loss, screw_origin_loss,
                     type_correctness / batch_size))

            type_correctness = 0

        # store training loss and step
        sum_loss += loss.item() * pc_start.size(0)
        sum_step += pc_start.size(0)

    type_correctness = 0

    # validation steps
    for i, data in enumerate(valid_dataloader, 0):
        pc_start, pc_end, joint_type, joint_screw = data
        pc_start, pc_end, joint_type, joint_screw = pc_start.to(device), pc_end.to(device), \
                                                    joint_type.to(device), joint_screw.to(device)

        model = model.eval()
        prediction = model(pc_start, pc_end)

        # calculate losses
        type_loss = typeloss(prediction[0, :1], joint_type)

        if joint_type < 0.5:
            screw_orientation_loss = degree_error_distance(prediction[:, 1:4], joint_screw[:, :3])
            screw_origin_loss = mseloss(prediction[:, 4:7], joint_screw[:, 3:]) \
                                + origin_error_distance(prediction[:, 1:4], prediction[:, 4:7],
                                                        joint_screw[:, :3], joint_screw[:, 3:])

        else:
            screw_orientation_loss = degree_error_distance(prediction[:, 7:], joint_screw[:, :3])
            screw_origin_loss = 0

        if (prediction[0, :1] >= 0.5).int() == joint_type:
            type_correctness += 1

        loss = (type_loss + screw_orientation_loss + screw_origin_loss)

        # store validation loss and step
        valid_sum_loss += loss.item() * pc_start.size(0)
        valid_sum_step += pc_start.size(0)

    new_valid_loss = valid_sum_loss / valid_sum_step

    blue = lambda x: '\033[94m' + x + '\033[0m'
    print('[%d: 0/0] %s loss: %f type accuracy: %.2f' % (epoch, blue('valid'), new_valid_loss, type_correctness / 50))

    # save a weight with lowest loss
    if valid_loss > new_valid_loss:
        valid_loss = new_valid_loss
        torch.save(model.state_dict(), '%s/model_valid_best.pth' % opt.weight)

    torch.save(model.state_dict(), '%s/model_last.pth' % opt.weight)

    time_loss.append([time.time() - start_time, sum_loss / sum_step, new_valid_loss])

    # reduce the learning rate after each epochs
    scheduler.step()

with open('loss_track.csv', 'w', newline='') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(['time', 'train loss', 'valid loss'])
    write.writerows(time_loss)
