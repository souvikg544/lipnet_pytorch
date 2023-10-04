import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import os
import sys
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
from dataset import MyDataset
import numpy as np
import time
from model import LipNet
import torch.optim as optim
import glob
opt = __import__('options')
# g=glob.glob(os.path.join("/home2/souvikg544/souvik/exp2_l2s/sample_gridcorpus/faces","**/*"), 
#                    recursive = False)
# print(g)

dataset = MyDataset(opt.video_path,
            opt.anno_path,
            opt.vid_padding,
            opt.txt_padding)
dataset_size=len(dataset)
train_ratio = 0.8  # Split ratio for training data
train_size = int(train_ratio * dataset_size)
eval_size = dataset_size - train_size

# Split dataset into training and evaluation subsets
train_indices = list(range(train_size))
eval_indices = list(range(train_size, dataset_size))

# Create SubsetRandomSampler for training and evaluation
train_sampler = SubsetRandomSampler(train_indices)
eval_sampler = SubsetRandomSampler(eval_indices)

# Create data loaders with the defined samplers
train_dataloader = DataLoader(dataset, batch_size=opt.batch_size, sampler=train_sampler,num_workers=opt.num_workers)
eval_dataloader = DataLoader(dataset, batch_size=opt.batch_size, sampler=eval_sampler,num_workers=opt.num_workers)

print(len(train_dataloader))