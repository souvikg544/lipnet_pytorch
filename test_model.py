import torch
import torch.nn as nn
from landmarknetmodelgru import landmarknet_model
import torch
import torch.optim as optim
import torch.nn as nn
from dataloader import landmarknet_data
from landmarknetmodelgru import landmarknet_model
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
import os
import pickle

# # Define the input size
# input_size = 366  # Assuming each row has 468 features
# time_dim = 75  # Assuming there are 75 rows in the sequence
# hidden_size = 256  # Number of GRU units
# output_size = 53

# model = landmarknet_model(input_size=366, num_classes=output_size)
# model=model.to("cuda")

# # Print the model architecture
# #print(model)

# # Verify the sizes of input and output
# input_example = torch.randn(256, time_dim, input_size)  # Example input tensor with batch size 2
# input_example=input_example.to("cuda")
# output_example = model(input_example)
# print("Input size:", input_example.size())
# print("Output size:", output_example.size())
video_no=425
root_folder = '/ssd_scratch/cvit/souvikg544/gridcorpus/'
dataset = landmarknet_data(root_folder)
dataset_size=len(dataset)
a,b=dataset.__getitem__(video_no)
print(b)
print(a)