import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2,h5py,os
from os import listdir
from os.path import isfile, join
import OpenEXR

class FocalStackDDFFH5Reader(Dataset):
    
    def __init__(self, hdf5_filename, stack_key="stack_train", disp_key="disp_train", norm=True):
        self.norm = norm
        self.hdf5 = h5py.File(hdf5_filename, 'r')
        self.stack_key = stack_key
        self.disp_key = disp_key
        focal_length = 521.4052
        K2 = 1982.0250823695178
        flens = 7317.020641763665
        baseline = K2 / flens * 1e-3
        focus_dists = np.linspace(baseline * focal_length / 0.5,baseline * focal_length / 7,num=10)
        focus_dists = focus_dists.reshape((focus_dists.shape[0],1,1))
        self.focus_dists = torch.Tensor(np.tile(focus_dists,[1,224,224]))
        self.min_dist = np.min(focus_dists)
        self.max_dist = np.max(focus_dists)

        if self.norm:
            self.focus_dists = (self.focus_dists-self.min_dist)/(self.max_dist - self.min_dist)

    def __len__(self):
        return self.hdf5[self.stack_key].shape[0]

    def __getitem__(self, idx):
        
        # image
        FS=self.hdf5[self.stack_key][idx] # n h w c
        
        # depth
        gt=self.hdf5[self.disp_key][idx].astype(np.float32)

        # tensor
        FS= torch.from_numpy(np.transpose(FS,(0,3,1,2))) # n c h w
        FS=FS/255
        
        mask=torch.from_numpy(np.where(gt==0.0,0.,1.).astype(np.bool_))
        if self.norm:
            gt = (gt - self.min_dist)  / (self.max_dist - self.min_dist)
        gt = torch.from_numpy(gt)

        return FS, gt , self.focus_dists, mask

class DDFF12dataset_benchmark(Dataset):
    def __init__(self, h5file):
        
        self.hdf5 = h5py.File(h5file, 'r') # ddff-dataset-test.h5
        self.stack_key = "stack_test"
        focal_length = 521.4052
        K2 = 1982.0250823695178
        flens = 7317.020641763665
        baseline = K2 / flens * 1e-3
        focus_dists = np.linspace(baseline * focal_length / 0.5, baseline * focal_length / 7,num=10)
        focus_dists = focus_dists.reshape((focus_dists.shape[0],1,1))
        self.focus_dists = torch.Tensor(np.tile(focus_dists,[1,384,576]))   

    def __len__(self):
        return self.hdf5[self.stack_key].shape[0]

    def __getitem__(self, idx):
        #Create sample dict
        FS=self.hdf5[self.stack_key][idx]
        FS = FS/255
        FS = torch.Tensor(np.transpose(FS,(0,3,1,2))) # n c h w
        
        N,C,H,W = FS.shape

        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        FS  = F.pad(FS, ( 0,pad_w,  0,pad_h) )
        
        return FS, self.focus_dists

class FS6_dataset(Dataset):
    def __init__(self,mode):
        
        self.mode=mode
        self.root = "Datasets/fs_6/" + mode +"/"
        self.imglist_all = [f for f in listdir(self.root) if isfile(join(self.root, f)) and f[-7:] == "All.tif"]
        self.imglist_dpt = [f for f in listdir(self.root) if isfile(join(self.root, f)) and f[-7:] == "Dpt.exr"]
        self.imglist_all.sort()
        self.imglist_dpt.sort()
        
        focus_dists = np.array([0.1,0.15,0.3,0.7,1.5])
        focus_dists = focus_dists.reshape((focus_dists.shape[0],1,1))
        self.focus_dists = torch.Tensor(np.tile(focus_dists,[1,256,256]))
        
        self.jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.5)
        self.affine = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=3),
            ])
        
    def __len__(self):
        return int(len(self.imglist_dpt))

    def __getitem__(self, index):
        
        # imgs
        FS= np.concatenate([np.expand_dims(cv2.imread(self.root + self.imglist_all[(index*5) + i]),axis=3) for i in range(5)],3) # h w c n
        FS = torch.Tensor(FS.transpose(3,2,0,1)) # n c h w
        FS=FS/255

        # depth
        gt = self.read_dpt (self.root + self.imglist_dpt[index]) 
        gt[ gt < (0.0 if self.mode=="train" else 0.1) ] = 0.0
        gt[ gt > (2.0 if self.mode=="train" else 1.5) ] = 0.0
        gt = torch.Tensor(gt)

        # aug
        if self.mode == 'train':
            FS=self.jitter(FS)
            
            gt = gt[None,None].repeat(1,3,1,1)
            FS=torch.cat([FS, gt], dim=0)
            FS=self.affine(FS)
            
            gt = FS[-1,0]
            FS = FS[:-1]

        #mask
        mask=torch.from_numpy(np.where(gt==0.0,0.,1.).astype(np.bool_))
 
        return FS, gt, self.focus_dists, mask
    
    def read_dpt(self,gt_path):
        dpt_img = OpenEXR.InputFile(gt_path)
        dw = dpt_img.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        (r, g, b) = dpt_img.channels("RGB")
        dpt = np.fromstring(r, dtype=np.float16)
        dpt.shape = (size[1], size[0])
        return dpt

class HCI_dataset(Dataset):
    
    def __init__(self, hdf5_filename,stack_key="stack_train", disp_key="disp_train"):

        self.hdf5 = h5py.File(hdf5_filename, 'r')
        self.stack_key = stack_key
        self.disp_key = disp_key
        
        self.input_size = (512,512)
        self.size = (256,256) if stack_key == "stack_train" else (512,512)

        focus_dists = self.hdf5['focus_position_disp']
        focus_dists = np.array(focus_dists).reshape((focus_dists.shape[1],1,1))
        self.focus_dists = torch.Tensor(np.tile(focus_dists,[1,self.size[0],self.size[1]]))
        
        self.min_dist = np.min(focus_dists)
        self.max_dist = np.max(focus_dists) 
        
        # aug
        self.crop = transforms.RandomCrop(self.size)
        self.jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.5)
        self.flip = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=3),
            ])
        
    def __len__(self):
        return self.hdf5[self.stack_key].shape[0]


    def __getitem__(self, idx):

        # image
        FS=self.hdf5[self.stack_key][idx] # n h w c
        FS=torch.Tensor(np.transpose(FS,(0,3,1,2))) # n c h w
        FS = FS/255 # [0,1]

        # disp & fd
        gt=self.hdf5[self.disp_key][idx].astype(np.float32)
        gt = torch.from_numpy(gt)
        re_focus_dists = self.focus_dists

        #augmentation
        if self.stack_key=="stack_train":

            # concat
            gt = gt[None,None].repeat(1,3,1,1) # 1 3 h w
            FS = torch.cat([FS, gt], dim=0) # n+1 3 h w

            # aug
            FS = self.crop(FS) 
            FS = self.flip(FS) 

            # split
            gt = FS[-1,0]
            FS = FS[:-1]

            FS = self.jitter(FS)

        elif self.stack_key == "stack_val":

            gt[ gt < self.min_dist ] = -3.0
            gt[ gt > self.max_dist ] = -3.0
        
        mask=torch.from_numpy(np.where(gt==-3.0,0.,1.).astype(np.bool_))

        return FS, gt , re_focus_dists, mask

class FlyingThings3d(Dataset):
    def __init__(self,mode):
        self.mode = mode
        self.num_imgs=15
        
        self.train_size=(256,256)
        self.input_size=(540,960)
        
        self.rgb_paths = [[] for i in range(self.num_imgs)]
        self.disp_paths = []
        self.focus_dists = np.linspace(10,100,self.num_imgs)
        self.focus_dists = np.expand_dims(self.focus_dists,axis=1)
        self.focus_dists = np.expand_dims(self.focus_dists,axis=2).astype(np.float32)
        with open ("Datasets/FlyingThings3D_FS/"+ mode + "/focal_stack_path.txt",'r') as f:
            for line in f.readlines():
                tmp = line.strip().split()
                for i in range(self.num_imgs):
                    self.rgb_paths[i].append(tmp[i])
                self.disp_paths.append(tmp[-1])

        self.jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.5)
        self.affine = transforms.Compose([
            transforms.RandomCrop(self.train_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=3),
            ])
        
    def __len__(self):
        return len(self.disp_paths)

    def __getitem__(self,idx):#TEST/Train
        os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
        
        # gt
        gt = cv2.imread(self.disp_paths[idx],cv2.IMREAD_UNCHANGED)
        gt[gt< 0.0] = 0.0
        gt= torch.Tensor(gt)

        #image
        FS= np.concatenate([np.expand_dims(cv2.imread(x[idx]),axis=3) for x in self.rgb_paths],3) # h w c n
        FS=torch.Tensor(np.transpose(FS,(3,2,0,1))) # n c h w
        FS = FS/255 # [0,1]

        #augmentation
        if self.mode=="train":
            
            # concat
            gt = gt[None,None].repeat(1,3,1,1) # 1 3 h w
            FS=torch.cat([FS, gt], dim=0) # n+1 3 h w

            # spatial transformation on FS and gt
            FS = self.affine(FS) 

            # split
            gt = FS[-1,0]
            FS = FS[:-1]

            # color jitter on FS
            FS = self.jitter(FS)
 
            Focus_Dists = torch.Tensor(np.tile(self.focus_dists,[1,self.train_size[0],self.train_size[1]]))

            
        elif self.mode == "val":

            _,_,H,W = FS.shape
            pad_h = (32 - H % 32) % 32
            pad_w = (32 - W % 32) % 32
            FS  = F.pad(FS, ( 0,pad_w,  0,pad_h) ) # pad to last 2 channels
            Focus_Dists = torch.Tensor(np.tile(self.focus_dists,[1,self.input_size[0]+pad_h,self.input_size[1]+pad_w]))
        
        mask=torch.from_numpy(np.where(gt==0.0,0.,1.).astype(np.bool_))
        

        return FS, gt, Focus_Dists, mask 
    
class Middlebury(Dataset):
    def __init__(self):
        self.num_imgs=15
        
        self.rgb_paths = [[] for i in range(self.num_imgs)]
        self.disp_paths = []
        
        self.low_bound = 10
        self.high_bound = 60
        
        focus_dists = np.linspace(self.low_bound,self.high_bound,self.num_imgs)
        self.focus_dists = focus_dists.reshape((focus_dists.shape[0],1,1))

        with open ("Datasets/Middlebury_FS/focal_stack/Middlebury_path.txt",'r') as f:
            for line in f.readlines():
                tmp = line.strip().split()
                for i in range(self.num_imgs):
                    self.rgb_paths[i].append(tmp[i])
                self.disp_paths.append(tmp[-1])
        
    def __len__(self):
        return len(self.disp_paths)

    def __getitem__(self,idx):#TEST/Train
        os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

        #depth
        gt = cv2.imread(self.disp_paths[idx],cv2.IMREAD_UNCHANGED) #gt check range, shape
        gt[gt< self.low_bound] = 0.0
        gt[gt > self.high_bound] = 0.0
        gt= torch.Tensor(gt)
        
        #img
        FS= np.concatenate([np.expand_dims(cv2.imread(x[idx]),axis=3) for x in self.rgb_paths],3) #H*W*C*N
        FS=torch.Tensor(np.transpose(FS,(3,2,0,1))) # n c h w
        FS=FS/255  

        # pad
        _,_,H,W = FS.shape
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        FS  = np.pad(FS, ( (0, 0), (0, 0), (0, pad_h), (0, pad_w)) )

        mask=torch.from_numpy(np.where(gt==0.0,0.,1.).astype(np.bool_))
        Focus_Dists = torch.Tensor(np.tile(self.focus_dists,[1,H+pad_h,W+pad_w]))

        return FS, gt, Focus_Dists, mask
    
