# encoding: utf-8
import numpy as np
import glob
import time
import cv2
import os
from torch.utils.data import Dataset
from cvtransforms import *
import torch
import glob
import re
import copy
import json
import random
import editdistance
import glob

    
class MyDataset(Dataset):
    letters = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    def __init__(self, video_path, anno_path, file_list, vid_pad, txt_pad, phase):
        self.anno_path = anno_path
        self.vid_pad = vid_pad
        self.txt_pad = txt_pad
        self.phase = phase
        
        # with open(file_list, 'r') as f:
        #     self.videos = [os.path.join(video_path, line.strip()) for line in f.readlines()]
        self.videos=glob.glob(os.path.join("/home2/souvikg544/souvik/exp2_l2s/sample_gridcorpus/faces","**/*"), 
                   recursive = False)
            
        self.data = []
        for vid in self.videos:
            items = vid.split(os.path.sep)            
            self.data.append((vid, items[-2], items[-1]))
            
        file_name_landmark = 'landmark_index_lip.pkl'
        with open(file_name_landmark, 'rb') as file:
            self.coordinates = pickle.load(file)
                
    def __getitem__(self, idx):
        (vid, spk, name) = self.data[idx]
        #vid = self._load_vid(vid)
        vid = self._load_land(vid)
        anno = self._load_anno(os.path.join(self.anno_path, spk, name + '.align'))

        # if(self.phase == 'train'):
        #     vid = HorizontalFlip(vid)
          
        # vid = ColorNormalize(vid)                   
        
        vid_len = vid.shape[0]
        anno_len = anno.shape[0]
        #vid = self._padding(vid, self.vid_pad)
        anno = self._padding(anno, self.txt_pad)
        
        return {
            'vid': torch.FloatTensor(vid.transpose(3, 0, 1, 2)), 
            'txt': torch.LongTensor(anno),
            'txt_len': anno_len,
            'vid_len': vid_len,
            #'vid_size':torch.FloatTensor(vid.transpose(3, 0, 1, 2)).size(),
            #'txt_size':torch.LongTensor(anno).size()
        }
            
    def __len__(self):
        return len(self.data)
        
    def _load_vid(self, p): 
        files = os.listdir(p)
        files = list(filter(lambda file: file.find('.jpg') != -1, files))
        files = sorted(files, key=lambda file: int(os.path.splitext(file)[0]))
        array = [cv2.imread(os.path.join(p, file)) for file in files]
        array = list(filter(lambda im: not im is None, array))
        array = [cv2.resize(im, (128, 64), interpolation=cv2.INTER_LANCZOS4) for im in array]
        array = np.stack(array, axis=0).astype(np.float32)
        return array

    def _load_land(self,p):
        frames=np.empty((0, self.l_size), dtype=float)
        for i in range(1,76,1):            
            res_path=os.path.join(video_frames,f"{i}_landmark.npy")
            blend_path=os.path.join(video_frames,f"{i}_blend.npy")
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
    
    def _load_anno(self, name):
        with open(name, 'r') as f:
            lines = [line.strip().split(' ') for line in f.readlines()]
            txt = [line[2] for line in lines]
            txt = list(filter(lambda s: not s.upper() in ['SIL', 'SP'], txt))
        return MyDataset.txt2arr(' '.join(txt).upper(), 1)
    
    def _padding(self, array, length):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)
    
    @staticmethod
    def txt2arr(txt, start):
        arr = []
        for c in list(txt):
            arr.append(MyDataset.letters.index(c) + start)
        return np.array(arr)
        
    @staticmethod
    def arr2txt(arr, start):
        txt = []
        for n in arr:
            if(n >= start):
                txt.append(MyDataset.letters[n - start])     
        return ''.join(txt).strip()
    
    @staticmethod
    def ctc_arr2txt(arr, start):
        pre = -1
        txt = []
        for n in arr:
            if(pre != n and n >= start):                
                if(len(txt) > 0 and txt[-1] == ' ' and MyDataset.letters[n - start] == ' '):
                    pass
                else:
                    txt.append(MyDataset.letters[n - start])                
            pre = n
        return ''.join(txt).strip()
            
    @staticmethod
    def wer(predict, truth):        
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return wer
        
    @staticmethod
    def cer(predict, truth):        
        cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
        return cer
