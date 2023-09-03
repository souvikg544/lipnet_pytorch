import torch
import torch.optim as optim
import torch.nn as nn
from dataloader import lipnet_data
from lipnetmodel import lipnet_model
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size=256


# -------------------- Initializing Dataloader -------------------------

print("-------------------- Initializing Dataloader -------------------------")

root_folder = '/ssd_scratch/cvit/souvikg544/gridcorpus/'

dataset = lipnet_data(root_folder)
dataset_size=len(dataset)
train_ratio = 0.9  # Split ratio for training data
train_size = int(train_ratio * dataset_size)
eval_size = dataset_size - train_size

# Split dataset into training and evaluation subsets
train_indices = list(range(train_size))
eval_indices = list(range(train_size, dataset_size))

# Create SubsetRandomSampler for training and evaluation
train_sampler = SubsetRandomSampler(train_indices)
eval_sampler = SubsetRandomSampler(eval_indices)

# Create data loaders with the defined samplers
train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
eval_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=eval_sampler)

# --------------------- Initializing Model -----------------------------

print("--------------------- Initializing Model -----------------------------")
num_classes=52
model = lipnet_model(num_classes).to(device)
#model = nn.DataParallel(model)

# -------------------------Loss Functions and Optimizers ---------------
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
k=0


save_epoch=[60,120,180,240]

# ----------------------------- Checkpoint paths and Evals --------------
checkpoint_dir = '/ssd_scratch/cvit/souvikg544/checkpoints_lipnet/exp2'
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)



best_eval_loss = float(3)

start_epoch=0
end_epoch=250

print("----------------------------- Starting Training ------------------------")
print(f"-----------------------------Device = {device} ------------------------")

for epoch in range(start_epoch,end_epoch,1):
    train_accuracy_sum=0
    eval_accuracy_sum=0
    # Training phase
    model.train()
    k=0
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{end_epoch}", leave=False)
    for batch_images, batch_align in train_dataloader:
        progress_bar.set_description(f"Epoch {epoch+1}/{end_epoch} - Progress: {k}/{len(train_dataloader)}")
        k+=1
        batch_images = batch_images.to(device)
        batch_align = batch_align.to(device)

        optimizer.zero_grad()
        predictions = model.forward(batch_images)
        predictions_reshaped = predictions.permute(0,2,1)
        targets_reshaped = batch_align.long()
        loss = loss_function(predictions_reshaped, targets_reshaped)
        loss.backward()
        optimizer.step()
        # if(k%10==0):
        #     print(".", end=".")

        predicted_labels = torch.argmax(predictions, dim=2)
        correct_predictions = torch.sum(predicted_labels == batch_align)
        total_predictions = predicted_labels.numel()  # Total number of elements in the tensor
        accuracy = (correct_predictions.item() / total_predictions) * 100
        train_accuracy_sum+=accuracy
    
    epoch_accuracy=train_accuracy_sum/len(train_dataloader)        
   

    # Evaluation phase
    model.eval()
    eval_loss = 0.0
    with torch.no_grad():
        for eval_images, eval_align in eval_dataloader:
            eval_images = eval_images.to(device)
            eval_align = eval_align.to(device)

            eval_predictions = model.forward(eval_images)
            eval_predictions_reshaped = eval_predictions.permute(0,2,1)
            eval_targets_reshaped = eval_align.long()
            eval_batch_loss = loss_function(eval_predictions_reshaped, eval_targets_reshaped)
            eval_loss += eval_batch_loss.item()
            
            eval_predicted_labels = torch.argmax(eval_predictions, dim=2)            
            eval_correct_predictions = torch.sum(eval_predicted_labels == eval_align)
            eval_total_predictions = eval_predicted_labels.numel()  # Total number of elements in the tensor
            eval_accuracy = (eval_correct_predictions.item() / eval_total_predictions) * 100
            eval_accuracy_sum+=eval_accuracy
            
    epoch_eval_accuracy=eval_accuracy_sum/len(eval_dataloader)
    eval_loss /= len(eval_dataloader)

    if (epoch + 1) in save_epoch:
        checkpoint_model_path=os.path.join(checkpoint_dir, f'model_{epoch + 1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_model_path)

        print(f"Checkpoint saved at epoch {epoch+1}")

    # Save the best model based on evaluation loss
    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        best_model_path = os.path.join(checkpoint_dir, f'best_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, best_model_path)
        print(f"Best model saved at epoch - {epoch+1}")

    tqdm.write(f"Epoch [{epoch+1}/{end_epoch}], Training Loss: {loss.item():.4f}, Training accuracy: {epoch_accuracy:.4f}, Eval Loss: {eval_loss:.4f}, Eval Accuracy: {epoch_eval_accuracy:.4f}")
    tqdm.write(" ")