import torch
import torch.nn as nn
import torch.nn.functional as F
from conv import Conv1d,Conv1dTranspose

class landmarknet_model(nn.Module):
    def __init__(self,input_size, num_classes):
        super(landmarknet_model, self).__init__()
        
        self.conv_blocks=nn.ModuleList([            
            nn.Sequential(
                Conv1d(in_channels=input_size, out_channels=32, kernel_size=7, stride=5, padding=1),
                Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1,residual=True),
                nn.Dropout(0.20)
            ),
            nn.Sequential(
                Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=5, padding=1),
                Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,residual=True),
                nn.Dropout(0.20)
            ),
            nn.Sequential(
                 Conv1dTranspose(in_channels=64,out_channels=96,kernel_size=2,stride=2,padding=0),
                 Conv1d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1,residual=True),
                 nn.Dropout(0.20)
             )
        ])

        self.gru_blocks= nn.ModuleList([
                 nn.GRU(input_size=96, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True),
                 nn.GRU(input_size=512, hidden_size=256,  num_layers=2, bidirectional=True, batch_first=True),
                 nn.GRU(input_size=512, hidden_size=256,  num_layers=2, bidirectional=True, batch_first=True),
        ])
        self.relu=nn.ReLU()
        
        #self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.20)    
        

    def forward(self, x):

        x=x.permute(0,2,1)    
        
        for f in self.conv_blocks:
            x = f(x)
            
        x=x.permute(0,2,1)
        
        for f in self.gru_blocks:
            x, _= f(x)

        #print(x.size())
            
        #x = self.fc1(x)
        #x=self.dropout(x)
        x=self.fc2(x)

        return x