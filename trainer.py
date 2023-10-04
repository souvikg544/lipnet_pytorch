import torch
import torch.optim as optim
import torch.nn as nn
from dataloader import landmarknet_data
from landmarknetmodelgru import landmarknet_model
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
import os

torch.backends.cudnn.deterministic = True
def save_model(save_checkpoint_path,model1,optimizer1,loss1,epoch1):    
    torch.save({
            'epoch': epoch1,
            'model_state_dict': model1.state_dict(),
            'optimizer_state_dict': optimizer1.state_dict(),
            'loss': loss1,
        }, save_checkpoint_path)

    print(f"Checkpoint saved at epoch {epoch1+1}")
    print("")
    


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#batch_size=512
batch_size=2
checkpoint_dir = '/ssd_scratch/cvit/souvikg544/checkpoints_landmarknet/exp10_gru_small'
home_check_dir = '/home2/souvikg544/souvik/checkpoints/exp10_gru_small'
load_checkpoint_path = '/home2/souvikg544/souvik/checkpoints/exp10_gru_small/best_train_model.pth'
load_checkpoint=True
start_epoch=0
end_epoch=100

# -------------------- Initializing Dataloader -------------------------

print("-------------------- Initializing Dataloader -------------------------")

#root_folder = '/ssd_scratch/cvit/souvikg544/gridcorpus/'
root_folder = '/home2/souvikg544/souvik/exp2_l2s/sample_gridcorpus'

dataset = landmarknet_data(root_folder)
dataset_size=len(dataset)
train_ratio = 0.8  # Split ratio for training data
train_size = int(train_ratio * dataset_size)
eval_size = dataset_size - train_size

# Split dataset into training and evaluation subsets
train_indices = list(range(train_size))
eval_indices = list(range(train_size, dataset_size))

# Create SubsetRandomSampler for training and evaluation
train_sampler = SubsetRandomSampler(train_indices)
eval_sampler = SubsetRandomSampler(eval_indices)

# Create data loaders with the defined samplers
train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,num_workers=4)
eval_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=eval_sampler,num_workers=4)

# --------------------- Initializing Model -----------------------------

print("--------------------- Initializing Model -----------------------------")
input_size=366
num_classes=53

model = landmarknet_model(input_size,num_classes).to(device)
#model = nn.DataParallel(model)

# -------------------------Loss Functions and Optimizers ---------------
#loss_function = nn.CrossEntropyLoss()
loss_function=nn.CTCLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001,betas=(0.9, 0.999))
k=0




# ----------------------------- Checkpoint paths and Evals --------------

if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

if not os.path.exists(home_check_dir):
    os.mkdir(home_check_dir)



best_eval_loss = float(4)
best_train_loss=float(4)

if load_checkpoint == True:
    checkpoint = torch.load(load_checkpoint_path)
    print(f"------------------------------------ Loaded Model from - {load_checkpoint_path} ---------------------------------------")
    # Get the saved model state dictionary, optimizer state dictionary, and other information
    start_epoch = checkpoint['epoch']
    end_epoch=start_epoch+200
    
    
    model_state_dict = checkpoint['model_state_dict']
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    best_train_loss = checkpoint['loss']
    
    # Load the model and optimizer states
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)

save_epoch=list(range(start_epoch,end_epoch+1,end_epoch//10))

print("----------------------------- Starting Training ------------------------")
print(f"-----------------------------Device = {device} ------------------------")

# input_lengths = torch.full(size=(batch_size,), fill_value=75, dtype=torch.int32)
# target_lengths = torch.full(size=(batch_size,), fill_value=6, dtype=torch.int32)

for epoch in range(start_epoch,end_epoch,1):
    train_accuracy_sum=0
    eval_accuracy_sum=0
    # Training phase
    
    k=1
    j=1
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{end_epoch}", leave=False)

    model.train()
    for batch_images, batch_align in train_dataloader:
        progress_bar.set_description(f"Epoch {epoch+1}/{end_epoch} - Training - Progress: {k}/{len(train_dataloader)}")
        k+=1
        batch_images = batch_images.to(device)
        batch_align = batch_align.to(device)

        optimizer.zero_grad()
        predictions = model.forward(batch_images)

        # -------------uncomment below for cnn training --------------------
        
        #predictions_reshaped = predictions.permute(0,2,1)
        #targets_reshaped = batch_align.long()
        # -----------------cross entropy loss -----------------------------------
        #loss = loss_function(predictions_reshaped, targets_reshaped)

        # --------------- uncomment for gru training -----------------------
        
        predictions_reshaped=predictions.permute(1,0,2).log_softmax(2)
        targets_reshaped=batch_align.to(torch.int32)
        input_lengths = torch.full(size=(predictions_reshaped.size()[1],), fill_value=75, dtype=torch.int32)
        target_lengths = torch.full(size=(predictions_reshaped.size()[1],), fill_value=6, dtype=torch.int32)
        # ------------------ctc loss ----------------
        loss = loss_function(predictions_reshaped, targets_reshaped, input_lengths, target_lengths)
        
        loss.backward()
        optimizer.step()
        # if(k%10==0):
        #     print(".", end=".")

        # predicted_labels = torch.argmax(predictions, dim=2)
        # correct_predictions = torch.sum(predicted_labels == batch_align)
        # total_predictions = predicted_labels.numel()  # Total number of elements in the tensor
        # accuracy = (correct_predictions.item() / total_predictions) * 100
        # train_accuracy_sum+=accuracy
    
    #epoch_accuracy=train_accuracy_sum/len(train_dataloader)        
    epoch_accuracy=0

    # Evaluation phase
    model.eval()
    eval_loss = 0.0
    with torch.no_grad():
        for eval_images, eval_align in eval_dataloader:
            progress_bar.set_description(f"Epoch {epoch+1}/{end_epoch} - Evaluation - Progress: {j}/{len(eval_dataloader)}")
            j+=1
            eval_images = eval_images.to(device)
            eval_align = eval_align.to(device)

            eval_predictions = model.forward(eval_images)
            
            # eval_predictions_reshaped = eval_predictions.permute(0,2,1)
            # eval_targets_reshaped = eval_align.long()
             #eval_batch_loss = loss_function(eval_predictions_reshaped, eval_targets_reshaped)
            
            eval_predictions_reshaped=eval_predictions.permute(1,0,2).log_softmax(2)
            eval_targets_reshaped =eval_align.to(torch.int32)
            input_lengths = torch.full(size=(eval_predictions_reshaped.size()[1],), fill_value=75, dtype=torch.int32)
            target_lengths = torch.full(size=(eval_predictions_reshaped.size()[1],), fill_value=6, dtype=torch.int32)
            
            eval_batch_loss = loss_function(eval_predictions_reshaped, eval_targets_reshaped, input_lengths, target_lengths)
            
            eval_loss += eval_batch_loss.item()
            
            # eval_predicted_labels = torch.argmax(eval_predictions, dim=2)            
            # eval_correct_predictions = torch.sum(eval_predicted_labels == eval_align)
            # eval_total_predictions = eval_predicted_labels.numel()  # Total number of elements in the tensor
            # eval_accuracy = (eval_correct_predictions.item() / eval_total_predictions) * 100
            # eval_accuracy_sum+=eval_accuracy
            
    #epoch_eval_accuracy=eval_accuracy_sum/len(eval_dataloader)
    epoch_eval_accuracy=0
    eval_loss /= len(eval_dataloader)

    if (epoch + 1) in save_epoch:
        checkpoint_model_path=os.path.join(checkpoint_dir, f'model_{epoch + 1}.pth')
        save_model(checkpoint_model_path,model,optimizer,loss,epoch)
       

    # Save the best model based on evaluation loss
    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        best_eval_model_path = os.path.join(home_check_dir, f'best_eval_model.pth')
        save_model(best_eval_model_path,model,optimizer,loss,epoch)

    if loss.item() < best_train_loss:
        best_train_loss=loss.item()
        best_train_model_path = os.path.join(home_check_dir, f'best_train_model.pth')
        save_model(best_train_model_path,model,optimizer,loss,epoch)
        

    #tqdm.write(f"Epoch [{epoch+1}/{end_epoch}], Training Loss: {loss.item():.4f}, Training accuracy: {epoch_accuracy:.4f}, Eval Loss: {eval_loss:.4f}, Eval Accuracy: {epoch_eval_accuracy:.4f}")

    tqdm.write(f"Epoch [{epoch+1}/{end_epoch}], Training Loss: {loss.item():.4f},Eval Loss: {eval_loss:.4f}")
    tqdm.write(" ")