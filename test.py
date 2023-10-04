import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import os
import sys
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

dataset = MyDataset(opt.video_path,opt.anno_path,opt.val_list,opt.vid_padding,opt.txt_padding,'test')
d=dataset.__getitem__(0)
print(d)