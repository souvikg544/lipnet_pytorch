gpu = '0'
random_seed = 0
data_type = 'unseen'
video_path = '/ssd_scratch/cvit/souvikg544/gridcorpus/landmarks'
train_list = f'data/{data_type}_train.txt'
val_list = f'data/{data_type}_val.txt'

anno_path = '/ssd_scratch/cvit/souvikg544/gridcorpus/transcription'
vid_padding = 75
txt_padding = 200
batch_size = 2
base_lr = 2e-4
num_workers = 4
start_epoch=0
max_epoch = 100
display = 200
test_step = 1000
save_prefix = f'/ssd_scratch/cvit/souvikg544/checkpoints_landmarknet/exp10_big'
is_optimize = True
log_dir='/ssd_scratch/cvit/souvikg544/tensorboard/logs_exp10'
save_epoch=list(range(start_epoch,max_epoch+1,max_epoch//10))

#weights = 'pretrain/LipNet_unseen_loss_0.44562849402427673_wer_0.1332580699113564_cer_0.06796452465503355.pt'
