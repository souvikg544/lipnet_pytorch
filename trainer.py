import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import os
import sys
from dataset import MyDataset
import numpy as np
import time
from landmarknetmodelgru import landmarknet_model
import torch.optim as optim
import re
import json
from tensorboardX import SummaryWriter


if(__name__ == '__main__'):
    opt = __import__('options')
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu    
    if(not os.path.exists(opt.log_dir)): os.makedirs(opt.log_dir)
    writer = SummaryWriter(opt.log_dir)

dataset = MyDataset(opt.video_path,
            opt.anno_path,
            opt.vid_padding,
            opt.txt_padding)

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
train_dataloader = DataLoader(dataset, batch_size=opt.batch_size, sampler=train_sampler,num_workers=opt.num_workers)
eval_dataloader = DataLoader(dataset, batch_size=opt.batch_size, sampler=eval_sampler,num_workers=opt.num_workers)




def save_model(save_checkpoint_path,model1,optimizer1,loss1,epoch1):    
    torch.save({
            'epoch': epoch1,
            'model_state_dict': model1.state_dict(),
            'optimizer_state_dict': optimizer1.state_dict(),
            'loss': loss1,
        }, save_checkpoint_path)

    print(f"Checkpoint saved at epoch {epoch1+1}")
    print("")

def dataset2dataloader(dataset, num_workers=opt.num_workers, shuffle=True):
    return DataLoader(dataset,
        batch_size = opt.batch_size, 
        shuffle = shuffle,
        num_workers = num_workers,
        drop_last = False)

def show_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return np.array(lr).mean()  

def ctc_decode(y):
    result = []
    y = y.argmax(-1)
    return [MyDataset.ctc_arr2txt(y[_], start=1) for _ in range(y.size(0))]
    
def test(model):

    with torch.no_grad():
        # dataset = MyDataset(opt.video_path,
        #     opt.anno_path,
        #     opt.val_list,
        #     opt.vid_padding,
        #     opt.txt_padding,
        #     'test')
            
        print('num_test_data:{}'.format(len(dataset.data)))  
        model.eval()
        #loader = dataset2dataloader(dataset, shuffle=False)
        loss_list = []
        wer = []
        cer = []
        crit = nn.CTCLoss()
        tic = time.time()
        for (i_iter, input) in enumerate(eval_dataloader):            
            vid = input.get('vid').cuda()
            txt = input.get('txt').cuda()
            vid_len = input.get('vid_len').cuda()
            txt_len = input.get('txt_len').cuda()
            
            y = model(vid)
            
            loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1)).detach().cpu().numpy()
            loss_list.append(loss)
            pred_txt = ctc_decode(y)
            
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            wer.extend(MyDataset.wer(pred_txt, truth_txt)) 
            cer.extend(MyDataset.cer(pred_txt, truth_txt))              
            if(i_iter % opt.display == 0):
                v = 1.0*(time.time()-tic)/(i_iter+1)
                eta = v * (len(eval_dataloader)-i_iter) / 3600.0
                
                print(''.join(101*'-'))                
                print('{:<50}|{:>50}'.format('predict', 'truth'))
                print(''.join(101*'-'))                
                for (predict, truth) in list(zip(pred_txt, truth_txt))[:10]:
                    print('{:<50}|{:>50}'.format(predict, truth))                
                print(''.join(101 *'-'))
                print('test_iter={},eta={},wer={},cer={}'.format(i_iter,eta,np.array(wer).mean(),np.array(cer).mean()))                
                print(''.join(101 *'-'))
                
        return (np.array(loss_list).mean(), np.array(wer).mean(), np.array(cer).mean())
    
def train(model):
    
    # dataset = MyDataset(opt.video_path,
    #     opt.anno_path,
    #     opt.train_list,
    #     opt.vid_padding,
    #     opt.txt_padding,
    #     'train')
    best_train_loss=4.0    
    if(not os.path.exists(opt.save_prefix)): os.makedirs(opt.save_prefix)
    #loader = dataset2dataloader(dataset) 
    optimizer = optim.Adam(model.parameters(),
                lr = opt.base_lr,
                weight_decay = 0.,
                amsgrad = True)
                
    print('num_train_data:{}'.format(len(dataset.data)))    
    crit = nn.CTCLoss()
    tic = time.time()
    
    train_wer = []
    for epoch in range(opt.start_epoch,opt.max_epoch):
        for (i_iter, input) in enumerate(train_dataloader):
            model.train()
            vid = input.get('vid').cuda()
            txt = input.get('txt').cuda()
            vid_len = input.get('vid_len').cuda()
            txt_len = input.get('txt_len').cuda()
            #print(vid.size(),txt.size(),vid_len,txt_len)
            
            optimizer.zero_grad()
            y = model(vid)
            #print(y.size())
            loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1))
            loss.backward()
            if(opt.is_optimize):
                optimizer.step()
            
            tot_iter = i_iter + epoch*len(train_dataloader)
            
            pred_txt = ctc_decode(y)
            
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            train_wer.extend(MyDataset.wer(pred_txt, truth_txt))
            
            if(tot_iter % opt.display == 0):
                v = 1.0*(time.time()-tic)/(tot_iter+1)
                eta = (len(train_dataloader)-i_iter)*v/3600.0
                
                writer.add_scalar('train loss', loss, tot_iter)
                writer.add_scalar('train wer', np.array(train_wer).mean(), tot_iter)              
                print(''.join(101*'-'))                
                print('{:<50}|{:>50}'.format('predict', 'truth'))                
                print(''.join(101*'-'))
                
                for (predict, truth) in list(zip(pred_txt, truth_txt))[:3]:
                    print('{:<50}|{:>50}'.format(predict, truth))
                print(''.join(101*'-'))                
                print('epoch={},total_iter={},eta={},loss={},train_wer={}'.format(epoch, tot_iter, eta, loss, np.array(train_wer).mean()))
                print(''.join(101*'-'))
                
            if(tot_iter % opt.test_step == 0):                
                (eval_loss, wer, cer) = test(model)
                print('i_iter={},lr={},loss={},wer={},cer={}'
                    .format(tot_iter,show_lr(optimizer),eval_loss,wer,cer))
                writer.add_scalar('val loss', eval_loss, tot_iter)                    
                writer.add_scalar('wer', wer, tot_iter)
                writer.add_scalar('cer', cer, tot_iter)
                # savename = '{}_loss_{}_wer_{}_cer_{}.pt'.format(opt.save_prefix, loss, wer, cer)
                # (path, name) = os.path.split(savename)
                # if(not os.path.exists(path)): os.makedirs(path)
                # torch.save(model.state_dict(), savename)
                if(not opt.is_optimize):
                    exit()
                    
        if loss.item() < best_train_loss:
            best_train_loss=loss.item()
            best_train_model_path = os.path.join(opt.save_prefix, f'best_train_model.pth')
            save_model(best_train_model_path,model,optimizer,loss,epoch)
            
            
        if (epoch + 1) in opt.save_epoch:
            checkpoint_model_path=os.path.join(opt.save_prefix, f'model_{epoch + 1}.pth')
            save_model(checkpoint_model_path,model,optimizer,loss,epoch)
        
if(__name__ == '__main__'):
    print("Loading options...")
    input_size,num_classes=366,28
    model = landmarknet_model(input_size,num_classes)
    model = model.cuda()
    #net = nn.DataParallel(model).cuda()

    if(hasattr(opt, 'weights')):
        pretrained_dict = torch.load(opt.weights)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed_all(opt.random_seed)
    train(model)
        
