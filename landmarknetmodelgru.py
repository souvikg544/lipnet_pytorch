import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from torch.nn import init

class landmarknet_model(nn.Module):
    def __init__(self,input_size, num_classes):
        super(landmarknet_model, self).__init__()
        self.gru_blocks= nn.ModuleList([
                 nn.GRU(input_size=input_size, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True),
                 nn.GRU(input_size=512, hidden_size=256,  num_layers=2, bidirectional=True, batch_first=True),
                 nn.GRU(input_size=512, hidden_size=256,  num_layers=2, bidirectional=True, batch_first=True), 
                 nn.GRU(input_size=512, hidden_size=256,  num_layers=2, bidirectional=True, batch_first=True),
                 nn.GRU(input_size=512, hidden_size=256,  num_layers=2, bidirectional=True, batch_first=True),
                 nn.GRU(input_size=512, hidden_size=256,  num_layers=2, bidirectional=True, batch_first=True),
        ])
        self.relu=nn.ReLU()
        
        #self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.30)
        
        rnn_size=256
        for m in self.gru_blocks:
            stdv = math.sqrt(2 / (96 * 3 * 6 + rnn_size))
            for i in range(0, rnn_size * 3, rnn_size):
                init.uniform_(m.weight_ih_l0[i: i + rnn_size],
                            -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0[i: i + rnn_size])
                init.constant_(m.bias_ih_l0[i: i + rnn_size], 0)
                init.uniform_(m.weight_ih_l0_reverse[i: i + rnn_size],
                            -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0_reverse[i: i + rnn_size])
                init.constant_(m.bias_ih_l0_reverse[i: i + rnn_size], 0)
        

    def forward(self, x):
        for f in self.gru_blocks:
            x, _= f(x)
            #x=self.relu(x)
            x=self.dropout(x)

        x=self.fc2(x)

        return x