import os
import torch
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
from torchvision import transforms
import cv2
import numpy as np
import glob
from torchtext.vocab import vocab
from collections import Counter, OrderedDict
import pickle
import warnings
warnings.filterwarnings('ignore', message='.*CPU', )

class landmarknet_data(Dataset):
    def __init__(self, root_folder,l_size=366):
        self.root_folder_landmarks =  os.path.join(root_folder,"landmarks")
        self.root_folder_transcription= os.path.join(root_folder,"transcription")
        self.frames_folder=glob.glob(os.path.join(self.root_folder_landmarks,"**/*"), 
                   recursive = False)
        
        file_name_vocab = 'vocab_dict1.pkl'
        file_name_landmark = 'landmark_index_lip.pkl'
        file_name_blend = 'blend_index.pkl'

        # Open the file in binary read mode
        with open(file_name_vocab, 'rb') as file:
            self.v1 = pickle.load(file)            
        with open(file_name_landmark, 'rb') as file:
            self.coordinates = pickle.load(file)
        with open(file_name_blend, 'rb') as file:
            self.blend_index = pickle.load(file)    
        #print(self.frames_folder)
        self.l_size=l_size        
    def __len__(self):        
        return len(self.frames_folder)
    
    
    def __getitem__(self, idx):
        video_frames = self.frames_folder[idx]
        video_name=  video_frames.split("/")[-1]
        speaker_name = video_frames.split("/")[-2]

        #print(video_name,speaker_name)
        
        align_path = os.path.join(self.root_folder_transcription,speaker_name, f"{video_name}.align")
        
        frames=np.empty((0, self.l_size), dtype=float)
        for i in range(1,76,1):            
            res_path=os.path.join(video_frames,f"{i}_landmark.npy")
            blend_path=os.path.join(video_frames,f"{i}_blend.npy")
            
            # if not os.path.exists(im_path):
            #     if i!=0:
            #         im_path=os.path.join(video_frames,f"{i-1}.jpg")
            #     else:
            #         im_path=os.path.join(video_frames,f"{i+1}.jpg")
                    
            
            res=np.load(res_path)
            blend=np.load(blend_path)            
            res1=np.empty((0, 3), dtype=float)
            for j in range(len(res)):
                k=res[j]
                
                if(j in self.coordinates):
                    row=np.array(k)
                    res1 = np.vstack((res1, row))
                    
            res1=res1.flatten()

            # for j in range(len(blend)):
            #     score=blend[j]
            #     if(j in self.blend_index):
            #         res1=np.append(res1,score)
            
            frames=np.vstack((frames,res1))    

        align_data = []
        with open(align_path, 'r') as align_file:
            for line in align_file:
                _,timestamp, label = line.strip().split()
                label=label.lower()
                if(label=="sil"):
                    continue
                #print(label)
                align_data.append(self.v1[label])
        
        align_data = torch.tensor(align_data)
        frames=torch.tensor(frames)
        #frames=frames.permute(3, 0, 1, 2)
        return frames.float(), align_data.float()
