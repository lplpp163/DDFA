import numpy as np
import time
import os
import gc
from tqdm import tqdm
import argparse

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from Dataloader import FocalStackDDFFH5Reader, FS6_dataset, HCI_dataset, FlyingThings3d
from metrics import *
from model import DDFA


# args
parser = argparse.ArgumentParser(description='Train code')
parser.add_argument('--saveroot', default="exp/", type=str,help='save_root')
parser.add_argument('--dataset', default="ddff", type=str,help='ddff, def, hci, fly')
parser.add_argument('--lr', default=1e-4, type=float,help='learning rate')
parser.add_argument('--max_epoch',default=3000,type=int,help='max epoch')
parser.add_argument('--load_epoch',default=0,type=int,help='load epoch')
parser.add_argument('--batch_size',default=4,type=int,help='batch size')
parser.add_argument('--cpus',default=8,type=int,help='num_workers')
args = parser.parse_args()

root = args.saveroot
os.makedirs(os.path.join(root, 'models'), exist_ok=True)
writer = SummaryWriter(log_dir=os.path.join(root,'logs'))

# param
max_epoch = args.max_epoch
load_epoch = args.load_epoch
lr = args.lr
batch_size = args.batch_size

#loss
smooth_loss = nn.SmoothL1Loss()

# model
model=DDFA()
model=model.cpu()
model = nn.DataParallel(model)
model=model.cuda()

# dataloader

if args.dataset == 'ddff':
    dataroot = 'Datasets/DDFF/ddff-dataset-trainval.h5'
    train_dataset=FocalStackDDFFH5Reader(dataroot)
    valid_dataset=FocalStackDDFFH5Reader(dataroot,'stack_val','disp_val')
    
elif args.dataset == 'def':
    train_dataset=FS6_dataset('train')
    valid_dataset=FS6_dataset('test')
    
elif args.dataset == 'hci':
    hdf5_filename='Datasets/HCI/HCI_FS_trainval.h5'
    train_dataset=HCI_dataset(hdf5_filename)
    valid_dataset=HCI_dataset(hdf5_filename,'stack_val','disp_val')
    
elif args.dataset == 'fly':
    train_dataset=FlyingThings3d('train')
    valid_dataset=FlyingThings3d('val')

dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=args.cpus,pin_memory=True)
valid_dataloader=DataLoader(valid_dataset,1,num_workers=args.cpus,pin_memory=True)

num_train=len(train_dataset)
num_val=len(valid_dataset)

# optim
optimizer=torch.optim.Adam(model.parameters(),lr=lr)
scaler = GradScaler()

# load checkpoints
if load_epoch != 0:
    model_path = os.path.join(root, 'models', f'{load_epoch}.pth') 
    if os.path.isfile(model_path):
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt)

# for saving best
best_eval = float('inf')

for epoch in range(load_epoch,max_epoch+1):#chang validation part
    gc.collect()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    # Validation
    if load_epoch != 0:
        model.eval()
        with torch.no_grad():
            
            avg_abs_rel=0.0
            avg_sq_rel=0.0
            avg_mse=0.0
            avg_rmse=0.0
            avg_rmse_log=0.0
            avg_Bump=0.0
            avg_accuracy_1=0.0
            avg_accuracy_2=0.0
            avg_accuracy_3=0.0
            avg_mae=0.0
            
            eval_loss=0.0 # for saving
            
            for idx, samples in enumerate(tqdm(valid_dataloader,desc='Valid')):
                inp, gt , fp, mask = samples
                fp = fp.cuda()
                gt = gt.numpy().squeeze() # b h w
                mask = mask.data.cpu().numpy().squeeze()
                
                #inference
                with autocast():
                    prob = model(inp) # b n h w
                    dpt = torch.sum(fp*prob,dim=1) # b h w
                    dpt = dpt.data.cpu().numpy().squeeze()
                    #crop
                    h,w = gt.shape
                    dpt=dpt[:h,:w]
                    
                    #loss
                    avg_mae += mask_mae(dpt,gt,mask)
                    avg_mse += mask_mse(dpt,gt,mask)
                    avg_abs_rel += mask_abs_rel(dpt,gt,mask)
                    avg_sq_rel += mask_sq_rel(dpt,gt,mask)
                    avg_rmse += mask_rmse(dpt,gt,mask)
                    avg_rmse_log += mask_rmse_log(dpt,gt,mask)
                    avg_Bump += get_bumpiness(gt,dpt,mask)
                    avg_accuracy_1 += mask_accuracy_k(dpt,gt,1,mask)
                    avg_accuracy_2 += mask_accuracy_k(dpt,gt,2,mask)
                    avg_accuracy_3 += mask_accuracy_k(dpt,gt,3,mask)
                    
            print("avg_mae(" +str(epoch)+") : " ,avg_mae/num_val)        
            print("avg_mse(" +str(epoch)+") : " ,avg_mse/num_val)
            print("avg_abs_rel(" +str(epoch)+") : " ,avg_abs_rel/num_val)
            print("avg_sq_rel(" +str(epoch)+") : " ,avg_sq_rel/num_val)
            print("avg_rmse(" +str(epoch)+") : " ,avg_rmse/num_val)
            print("avg_rmse_log(" +str(epoch)+") : " ,avg_rmse_log/num_val)
            print("avg_Bump(" +str(epoch)+") : " ,avg_Bump/num_val)
            print("avg_accuracy_1(" +str(epoch)+") : " ,avg_accuracy_1/num_val)
            print("avg_accuracy_2(" +str(epoch)+") : " ,avg_accuracy_2/num_val)
            print("avg_accuracy_3(" +str(epoch)+") : " ,avg_accuracy_3/num_val)
            print()
            writer.add_scalar("Validate/avg_mae",avg_mae/num_val,epoch)
            writer.add_scalar("Validate/avg_mse",avg_mse/num_val,epoch)
            writer.add_scalar("Validate/avg_abs_rel",avg_abs_rel/num_val,epoch)
            writer.add_scalar("Validate/avg_sq_rel",avg_sq_rel/num_val,epoch)
            writer.add_scalar("Validate/avg_rmse",avg_rmse/num_val,epoch)
            writer.add_scalar("Validate/avg_rmse_log",avg_rmse_log/num_val,epoch)
            writer.add_scalar("Validate/avg_accuracy_1",avg_accuracy_1/num_val,epoch)
            writer.add_scalar("Validate/avg_accuracy_2",avg_accuracy_2/num_val,epoch)
            writer.add_scalar("Validate/avg_accuracy_3",avg_accuracy_3/num_val,epoch)
            
            eval_loss = avg_mse/num_val

            # init best loss
            if (epoch == load_epoch):
                best_eval = eval_loss
            
            # save best
            if(eval_loss < best_eval and epoch !=load_epoch):
                best_eval = eval_loss
                path= os.path.join(root, 'models', str(epoch)+'.pth') 
                torch.save(model.state_dict(),path)
                    
    # Training
    if epoch == max_epoch: break
    model.train()

    train_loss = 0.0

    for idx, samples in enumerate(tqdm(dataloader,desc='Train')):
        inp, gt , fp, mask = samples
        inp=inp.cuda(non_blocking=True)
        gt=gt.cuda(non_blocking=True)
        fp=fp.cuda(non_blocking=True)
        mask=mask.cuda(non_blocking=True)

        #amp
        with autocast():
            prob = model(inp) # b n h w
            dpt = torch.sum(fp*prob,dim=1) # b h w
            loss = smooth_loss(dpt[mask], gt[mask])

        # backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        train_loss += loss
    
    print(f'Train Loss({epoch+1}) : {train_loss.item()/(num_train)}\n')
    writer.add_scalar(f"Train/Loss", train_loss.item()/(num_train), epoch)