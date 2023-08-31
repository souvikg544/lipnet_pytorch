from lipnetmodel import lipnet_model
import torch

batch_size = 16
num_frames = 75
num_channels = 3
height, width = 64,128
num_classes = 52

# Create a random input tensor
input_data = torch.randn(batch_size,num_channels,num_frames,height, width)
print("Input shape:", input_data.shape)
# Initialize the STCNN model
stcnn_model = lipnet_model(num_classes)

# Get predictions from the model
predictions = stcnn_model.forward(input_data)


print("Output shape (predictions):", predictions.shape)