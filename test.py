import time
import os
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Dataloader import FS6_dataset, HCI_dataset, Middlebury, DDFF12dataset_benchmark
from metrics import *
from model import DDFA

import matplotlib.pyplot as plt

# args
parser = argparse.ArgumentParser(description='Test code')
parser.add_argument('--dataset', default="ddff", type=str, help='ddff, def, hci, fly')
args = parser.parse_args()


# data
if args.dataset == 'ddff':
    h5file = 'Datasets/DDFF/ddff-dataset-test.h5'
    test_set=DDFF12dataset_benchmark(h5file)
    root = "Results/DDFF/"
elif args.dataset == 'def':
    test_set=FS6_dataset('test')
    root = "Results/DefocusNet/"
elif args.dataset == 'hci':
    h5file='Datasets/HCI/HCI_FS_trainval.h5'
    test_set=HCI_dataset(h5file,stack_key='stack_val',disp_key='disp_val')
    root = "Results/4D_Light_Field/"
elif args.dataset == 'fly': 
    test_set=Middlebury()
    test_set2=FS6_dataset('test')
    root = "Results/FlyingThings3D/Middlebury/"
    root2 = "Results/FlyingThings3D/DefocusNet/"
    
test_loader = DataLoader(test_set)
num_test = len(test_set)

# model
model=DDFA()
model=model.cpu()
model = nn.DataParallel(model)
model=model.cuda()

# load ckpt
model.load_state_dict(torch.load(os.path.join(root, 'ckpt.pth')))


model.eval()
with torch.no_grad():
    
    test_time=0
    
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

    for idx, samples in enumerate(tqdm(test_loader)):
        
        if args.dataset == 'ddff':
            inp, fp = samples
            fp = fp.cuda()
        else:
            inp, gt , fp, mask = samples     
            fp = fp.cuda()      
            gt = gt.numpy().squeeze() # b h w
            mask = mask.data.cpu().numpy().squeeze()
        
        #inference
        start_time = time.time()
        prob = model(inp) # b n h w
        test_time += time.time() - start_time
        
        dpt = torch.sum(fp*prob,dim=1) # b h w
        dpt = dpt.cpu().numpy().squeeze()

        if args.dataset != 'ddff':
            h,w = gt.shape
            dpt=dpt[:h,:w]
        
        # save fig
        plt.imsave(os.path.join(root, 'Depth', f'{idx}.jpg'), dpt, cmap='jet')

        # matrics
        if args.dataset != 'ddff':
            avg_mae = avg_mae + mask_mae(dpt,gt,mask)
            avg_mse = avg_mse + mask_mse(dpt,gt,mask)
            avg_abs_rel = avg_abs_rel + mask_abs_rel(dpt,gt,mask)
            avg_sq_rel = avg_sq_rel + mask_sq_rel(dpt,gt,mask)
            avg_rmse = avg_rmse + mask_rmse(dpt,gt,mask)
            avg_rmse_log = avg_rmse_log + mask_rmse_log(dpt,gt,mask)
            avg_Bump = avg_Bump + get_bumpiness(gt,dpt,mask)
            avg_accuracy_1 = avg_accuracy_1 + mask_accuracy_k(dpt,gt,1,mask)
            avg_accuracy_2 = avg_accuracy_2 + mask_accuracy_k(dpt,gt,2,mask)
            avg_accuracy_3 = avg_accuracy_3 + mask_accuracy_k(dpt,gt,3,mask)

if args.dataset != 'ddff':
    print("avg_mae : " ,avg_mae/num_test)
    print("avg_mse : " ,avg_mse/num_test)
    print("avg_abs_rel : " ,avg_abs_rel/num_test)
    print("avg_sq_rel : " ,avg_sq_rel/num_test)
    print("avg_rmse : " ,avg_rmse/num_test)
    print("avg_rmse_log : " ,avg_rmse_log/num_test)
    print("avg_Bump : " ,avg_Bump/num_test)
    print("avg_accuracy_1 : " ,avg_accuracy_1/num_test)
    print("avg_accuracy_2 : " ,avg_accuracy_2/num_test)
    print("avg_accuracy_3 : " ,avg_accuracy_3/num_test)

print("avg_time:",test_time/num_test)


# fly for defocusnet dataset
if args.dataset == 'fly':
    test_loader = DataLoader(test_set2)
    num_test = len(test_set2)
    model.load_state_dict(torch.load(os.path.join(root2, 'ckpt.pth')))
    
    with torch.no_grad():
        
        test_time=0
        
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

        for idx, samples in enumerate(tqdm(test_loader)):
            
            inp, gt , fp, mask = samples     
            fp = fp.cuda()      
            gt = gt.numpy().squeeze() # b h w
            mask = mask.data.cpu().numpy().squeeze()
            
            #inference
            start_time = time.time()
            prob = model(inp) # b n h w
            test_time += time.time() - start_time
            
            dpt = torch.sum(fp*prob,dim=1) # b h w
            dpt = dpt.cpu().numpy().squeeze()

            h,w = gt.shape
            dpt=dpt[:h,:w]
            
            # save fig
            plt.imsave(os.path.join(root2, 'Depth', f'{idx}.jpg'), dpt, cmap='jet')
            
            # matrics
            avg_mae = avg_mae + mask_mae(dpt,gt,mask)
            avg_mse = avg_mse + mask_mse(dpt,gt,mask)
            avg_abs_rel = avg_abs_rel + mask_abs_rel(dpt,gt,mask)
            avg_sq_rel = avg_sq_rel + mask_sq_rel(dpt,gt,mask)
            avg_rmse = avg_rmse + mask_rmse(dpt,gt,mask)
            avg_rmse_log = avg_rmse_log + mask_rmse_log(dpt,gt,mask)
            avg_Bump = avg_Bump + get_bumpiness(gt,dpt,mask)
            avg_accuracy_1 = avg_accuracy_1 + mask_accuracy_k(dpt,gt,1,mask)
            avg_accuracy_2 = avg_accuracy_2 + mask_accuracy_k(dpt,gt,2,mask)
            avg_accuracy_3 = avg_accuracy_3 + mask_accuracy_k(dpt,gt,3,mask)
            
    print("avg_mae : " ,avg_mae/num_test)
    print("avg_mse : " ,avg_mse/num_test)
    print("avg_abs_rel : " ,avg_abs_rel/num_test)
    print("avg_sq_rel : " ,avg_sq_rel/num_test)
    print("avg_rmse : " ,avg_rmse/num_test)
    print("avg_rmse_log : " ,avg_rmse_log/num_test)
    print("avg_Bump : " ,avg_Bump/num_test)
    print("avg_accuracy_1 : " ,avg_accuracy_1/num_test)
    print("avg_accuracy_2 : " ,avg_accuracy_2/num_test)
    print("avg_accuracy_3 : " ,avg_accuracy_3/num_test)

    print("avg_time:",test_time/num_test)