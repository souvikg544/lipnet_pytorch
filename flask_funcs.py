import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import os
import torch
import torch.optim as optim
import torch.nn as nn
from lipnetmodel import lipnet_model
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
import os
import pickle



file_name = 'vocab_dict.pkl'

with open(file_name, 'rb') as file:
    word_label_dict = pickle.load(file)


# Accepts 75 frames from opencv. Returns Finds face from each of frames , appends it and permutes and sends

face_model_path = 'detector.tflite'
base_options = python.BaseOptions(model_asset_path=face_model_path)
options = vision.FaceDetectorOptions(base_options=base_options,min_detection_confidence=0.5)
detector = vision.FaceDetector.create_from_options(options)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path="/ssd_scratch/cvit/souvikg544/checkpoints_lipnet/exp6_big/model_179.pth"
num_classes=52
model = lipnet_model(num_classes)

checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

def process_frame(images):
    # saves the file
    frames=[]
    for f in images:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=f)
        detection_result = detector.detect(mp_image)
        bbox=detection_result.detections[0].bounding_box
        x,y = bbox.origin_x, bbox.origin_y
        x1,y1 = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        c_image=f[y:y1+15, x:x1]
        height, width, _ = c_image.shape
        start_y = height // 2
        end_y = height           
        cropped_image = c_image[start_y:end_y, :]
        cropped_image=cv2.resize(cropped_image, (128, 64))
        cropped_image=cropped_image/255.
        frames.append(cropped_image)
    frames=torch.tensor(np.array(frames))
    frames=frames.permute(3, 0, 1, 2) # 75,64,128,3 ---> 3,75,64,128
    frames=frames.float()
    frames=frames.unsqueeze(0)
    frames=frames.to(device)
    return frames


# Given the batch frames , returns the sente
def inference(batch_frames):
    pred=model.forward(batch_frames)
    softmax=nn.Softmax(dim=1)
    pred=softmax(pred)
    print(pred)
    predicted_labels = torch.argmax(pred, dim=2)
    pred_words=[list(word_label_dict.keys())[list(word_label_dict.values()).index(x)] for x in predicted_labels[0]]
    #print(pred_words)
    return " ".join(pred_words)
    