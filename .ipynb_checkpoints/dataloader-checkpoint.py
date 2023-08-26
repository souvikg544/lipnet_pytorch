import os
import torch
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
from torchvision import transforms
import cv2
import numpy as np
import glob
from torchtext.vocab import vocab
from collections import Counter, OrderedDict

class lipnet_data(Dataset):
    def __init__(self, root_folder):
        self.root_folder_faces =  os.path.join(root_folder,"faces")
        self.root_folder_transcription= os.path.join(root_folder,"transcription")
        self.frames_folder=glob.glob(os.path.join(self.root_folder_faces,"**/*"), 
                   recursive = False)
        
        self.transform = transforms.Compose([transforms.Resize((100,100)),
                                             transforms.ToTensor()])
        self.getwordlist()

    def getwordlist(self):
        label_files=glob.glob(os.path.join(self.root_folder_transcription,"**/*"), 
                   recursive = False)

        self.words=[]
        for f in label_files:
            with open(f, 'r') as align_file:
                for line in align_file:
                    _,timestamp, label = line.strip().split()
                    label=label.lower()
                    if(label=="sil"):
                        continue
                    self.words.append(label)
        self.counter = Counter(self.words)
        sorted_by_freq_tuples = sorted(self.counter.items(), key=lambda x: x[1], reverse=False)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
        self.v1 = vocab(ordered_dict)
        #print(len(self.v1))
        #self.v1.set_default_index(-1)
                
    def __len__(self):        
        return len(self.frames_folder)
    
    def __getitem__(self, idx):
        video_frames = self.frames_folder[idx]
        video_name=  video_frames.split("/")[-1]
        speaker_name = video_frames.split("/")[-2]
        
        align_path = os.path.join(self.root_folder_transcription,speaker_name, f"{video_name}.align")
        frames=[]
        for i in range(1,76,1):
            im_path=os.path.join(video_frames,f"{i}.jpg")
            if not os.path.exists(im_path):
                if i!=0:
                    im_path=os.path.join(video_frames,f"{i-1}.jpg")
                else:
                    im_path=os.path.join(video_frames,f"{i+1}.jpg")
            image=cv2.imread(im_path,cv2.COLOR_BGR2RGB)            
            height, width, _ = image.shape
            start_y = height // 2
            end_y = height           
            cropped_image = image[start_y:end_y, :]
            cropped_image=cv2.resize(cropped_image, (128, 64))
            cropped_image=cropped_image/255.
            #cropped_image=self.transform(cropped_image)
            frames.append(cropped_image)
       

        align_data = []
        with open(align_path, 'r') as align_file:
            for line in align_file:
                _,timestamp, label = line.strip().split()
                label=label.lower()
                if(label=="sil"):
                    continue
                align_data.append(self.v1[label])
        
        align_data = torch.tensor(align_data)
        frames=torch.tensor(np.array(frames))
        frames=frames.permute(3, 0, 1, 2)
        return frames.float(), align_data.float()
