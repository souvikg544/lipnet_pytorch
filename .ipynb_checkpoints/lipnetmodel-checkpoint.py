import torch
import torch.nn as nn
import torch.nn.functional as F
from conv import Conv3d,Conv3dTranspose

class lipnet_model(nn.Module):
    def __init__(self, num_classes):
        super(lipnet_model, self).__init__()
        
        self.conv_blocks=nn.ModuleList([
            nn.Sequential(Conv3d(in_channels=3, out_channels=32, kernel_size=(3, 5, 5), stride=(5, 2, 2), padding=(1, 2, 2))),
            nn.Sequential(
                Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 5, 5), stride=(5, 2, 2), padding=(1, 2, 2)),
                Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2),residual=True)
            ),
            nn.Sequential(
                Conv3d(in_channels=64, out_channels=96, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
                Conv3d(in_channels=96, out_channels=96, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
                Conv3d(in_channels=96, out_channels=96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),residual=True)
            ),
            nn.Sequential(
                Conv3d(in_channels=96, out_channels=96, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
            ),
            nn.Sequential(
                 Conv3dTranspose(in_channels=96,out_channels=96,kernel_size=(2, 1, 1),stride=(2, 1, 1),padding=(0, 0, 0))
             )
        ])

        self.gru_blocks= nn.ModuleList([
                 nn.GRU(input_size=96 * 2 * 4, hidden_size=256, bidirectional=True, batch_first=True),
                 nn.GRU(input_size=512, hidden_size=256, bidirectional=True, batch_first=True)
        ])
        self.fc1 = nn.Linear(512, num_classes)
            

        
        self.bi_gru1 = nn.GRU(input_size=96 * 2 * 4, hidden_size=256, bidirectional=True, batch_first=True)
        self.bi_gru2 = nn.GRU(input_size=512, hidden_size=256, bidirectional=True, batch_first=True)
        

    def forward(self, x):
    
        for f in self.conv_blocks:
            x = f(x)
            
        batch_size, num_channels,seq_len, height, width = x.size()
        #print(x.size())
        x = x.view(batch_size, seq_len, num_channels * height * width)  
        #print(x.size())
        
        for f in self.gru_blocks:
            x, _= f(x)
            
        x = self.fc1(x)

        return x