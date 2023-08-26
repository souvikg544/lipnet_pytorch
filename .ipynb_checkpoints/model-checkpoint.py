import torch
import torch.nn as nn
import torch.nn.functional as F

class lipnet_model(nn.Module):
    def __init__(self, num_classes):
        super(lipnet_model, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(3, 5, 5), stride=(5, 2, 2), padding=(1, 2, 2))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 5, 5), stride=(5, 2, 2), padding=(1, 2, 2))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=96, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv_transpose = nn.ConvTranspose3d(in_channels=96,out_channels=96,kernel_size=(2, 1, 1),stride=(2, 1, 1),padding=(0, 0, 0))

        
        self.bi_gru1 = nn.GRU(input_size=96 * 1 * 2, hidden_size=256, bidirectional=True, batch_first=True)
        self.bi_gru2 = nn.GRU(input_size=512, hidden_size=256, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        #print(x.size())
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        #print(x.size())
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        #batch_size, num_channels,seq_len, height, width = x.size()
        #print(x.size())
        x=self.conv_transpose(x)
        batch_size, num_channels,seq_len, height, width = x.size()
        #print(x.size())
        x = x.view(batch_size, seq_len, num_channels * height * width)  # Reshape for GRU input
        x, _ = self.bi_gru1(x)
        x, _ = self.bi_gru2(x)
        #print(x.size())
        x = self.fc1(x)
        #print(x.size())
        return x