import torch
import torch.optim as optim
import torch.nn as nn
from dataloader import lipnet_data
from lipnetmodel import lipnet_model
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
import os
import pickle



file_name = 'vocab_dict.pkl'

with open(file_name, 'rb') as file:
    word_label_dict = pickle.load(file)



model_path="/ssd_scratch/cvit/souvikg544/checkpoints_lipnet/exp6_big/model_179.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_folder = '/ssd_scratch/cvit/souvikg544/gridcorpus/'
dataset = lipnet_data(root_folder)
dataset_size=len(dataset)

video_no=dataset_size-60
#video_no=1000

num_classes=52
model = lipnet_model(num_classes)
model.load_state_dict(torch.load(model_path)['model_state_dict'])
print(torch.load(model_path)['epoch'])
print(torch.load(model_path)['loss'])
model.to(device)
a,b=dataset.__getitem__(video_no)

a=a.unsqueeze(0)
a=a.to(device)
pred=model.forward(a)

softmax=nn.Softmax(dim=1)
pred=softmax(pred)
print(pred)
predicted_labels = torch.argmax(pred, dim=2)
print(predicted_labels)
actual_words=[list(word_label_dict.keys())[list(word_label_dict.values()).index(x)] for x in b]
pred_words=[list(word_label_dict.keys())[list(word_label_dict.values()).index(x)] for x in predicted_labels[0]]
print(actual_words)
print(pred_words)
