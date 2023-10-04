import torch
import torch.optim as optim
import torch.nn as nn
from dataloader import landmarknet_data
from landmarknetmodelgru import landmarknet_model
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
import os
import pickle
import numpy as np
from itertools import groupby

def ctc_beam_search_decoder(probs, blank=0, beam_width=10):
    """
    CTC Beam Search Decoding for PyTorch.

    Args:
        probs (Tensor): The softmax output probabilities for each batch and time step.
        blank (int): Index of the blank label (typically 0).
        beam_width (int): Width of the beam for beam search.

    Returns:
        List of decoded sequences for each batch (list of lists of integers).
    """
    batch_size, T, C = probs.shape  # Batch size, T: Number of time steps, C: Number of classes

    # Initialize the beams for each batch with the empty sequence.
    beams = [[([], 0.0)] * beam_width for _ in range(batch_size)]

    for t in range(T):
        new_beams = [[] for _ in range(batch_size)]

        for b in range(batch_size):
            for beam, beam_score in beams[b]:
                # Extend with the blank label.
                extended_beam = beam + [blank]
                new_beams[b].append((extended_beam, beam_score + np.log(probs[b, t, blank].item())))

                # Extend with the argmax label.
                argmax_label = torch.argmax(probs[b, t, 1:]) + 1  # Exclude the blank label.
                extended_beam = beam + [argmax_label]
                new_beams[b].append((extended_beam, beam_score + np.log(probs[b, t, argmax_label].item())))

                # Extend repetitions of labels.
                if len(beam) > 0 and beam[-1] != blank and beam[-1] == argmax_label:
                    extended_beam = beam + [argmax_label]
                    new_beams[b].append((extended_beam, beam_score + np.log(probs[b, t, argmax_label].item())))

            # Prune and keep the top beam_width beams for this batch.
            new_beams[b].sort(key=lambda x: x[1], reverse=True)
            beams[b] = new_beams[b][:beam_width]

    decoded_sequences = []

    for b in range(batch_size):
        # Select the best path (highest probability) for each batch element.
        best_beam = beams[b][0][0]

        # Remove consecutive duplicates and blanks.
        decoded_sequence = [label for label, _ in groupby(best_beam) if label != blank]

        decoded_sequences.append(decoded_sequence)

    return decoded_sequences



file_name = 'vocab_dict1.pkl'

with open(file_name, 'rb') as file:
    word_label_dict = pickle.load(file)


model_path="/home2/souvikg544/souvik/checkpoints/exp8_gru/best_train_model.pth"
#model_path="/ssd_scratch/cvit/souvikg544/checkpoints_landmarknet/exp8_gru/model_60.pth"
device = "cpu"

root_folder = '/ssd_scratch/cvit/souvikg544/gridcorpus/'
dataset = landmarknet_data(root_folder)
dataset_size=len(dataset)
video_no=416

num_classes=53
model = landmarknet_model(366,num_classes)

t=torch.load(model_path,map_location=torch.device('cpu'))
#t=torch.load(model_path)
print("1")
model.load_state_dict(t['model_state_dict'])
model.eval()
print("Epoch -----",t['epoch'])
print("Loss -------",t['loss'])
model.to(device)
a,b=dataset.__getitem__(video_no)

a=a.unsqueeze(0)
a=a.to(device)

#print(a)
print(b)
pred=model.forward(a)

softmax=nn.LogSoftmax(dim=2)
pred1=softmax(pred)
# decoded_sequences = ctc_beam_search_decoder(pred1)
# print("Decoded Sequences:", decoded_sequences)

#print(pred.size())
predicted_labels = torch.argmax(pred, dim=2)
print(predicted_labels)

pred1=[]
for item in predicted_labels[0]:
    if item not in pred1 and item!=0:
        pred1.append(item)
        
actual_words=[list(word_label_dict.keys())[list(word_label_dict.values()).index(x)] for x in b]
pred_words=[list(word_label_dict.keys())[list(word_label_dict.values()).index(x)] for x in pred1]
print("Actual Words ---------",actual_words)
print("Predicted words -------",pred_words)
