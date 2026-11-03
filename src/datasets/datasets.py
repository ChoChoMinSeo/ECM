from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from compressai.registry import register_dataset
import cv2
import torch
import os
from torchvision import transforms
import random
from glob import glob
    
@register_dataset("KittiDrivingDataset")
class KittiDrivingDataset(Dataset):
    def __init__(self, root, transform=None,split = 'train',subsample = None, max_len = None):
        self.base_samples = []
        self.max_len = max_len
        self.split = split
        
        if split == 'train':
            self.base_samples = self.base_samples + glob(root+'data_object_image_2/training/image_2/*.jpg') + glob(root+'data_object_image_2/training/image_2/*.png')
        else:
            self.base_samples = self.base_samples + glob(root+'data_object_image_2/testing/image_2/*.jpg') + glob(root+'data_object_image_2/testing/image_2/*.png')
            self.samples = self.base_samples[:100]
                
        self.transform = transform
        if split=='train':
            if subsample is not None:
                self.base_samples = self.base_samples[::subsample]
            self.set_epoch()
        self.cls_idx = 0
            
    def set_epoch(self):
        random.shuffle(self.base_samples)
        if self.max_len is not None:
            self.samples = self.base_samples * (self.max_len // len(self.base_samples) + 1)
            self.samples = self.samples[:self.max_len]
        else:
            self.samples = self.base_samples
        
    def __getitem__(self, index):
        img = Image.open(self.samples[index]).convert("RGB")
        img = self.transform(img)
        
        if self.split == 'train':
            return img, img , self.cls_idx
        
        _,H,W = img.shape
        if H != 0:
            if H % 256 != 0:
                H = H + 256 - (H % 256)
        if W != 0:
            if W % 256 != 0:
                W = W + 256 - (W % 256)
            
        crop_or_pad = transforms.CenterCrop((H,W))
        img = crop_or_pad(img)
        
        return img,img, self.cls_idx
        
    def __len__(self):
        return len(self.samples)
    
@register_dataset("VimeoTripletResidueDataset")
class VimeoTripletResidueDataset(Dataset):
    def __init__(self, root, transform=None,split = 'train',subsample = None, max_len = None):
        self.base_samples = []
        self.max_len = max_len
        self.split = split
        if split == 'train':
            with open(root+'tri_trainlist.txt' ,'r') as f:
                for line in f:
                    self.base_samples.append(root+f'/sequences/{line.strip()}/im1.png')
                    self.base_samples.append(root+f'/sequences/{line.strip()}/im2.png')
        else:
            with open(root+'tri_testlist.txt' ,'r') as f:
                for line in f:
                    self.base_samples.append(root+f'/sequences/{line.strip()}/im1.png')
                    self.base_samples.append(root+f'/sequences/{line.strip()}/im2.png')
            self.samples = self.base_samples[:100]
            
        self.transform = transform
        if split=='train':
            if subsample is not None:
                self.base_samples = self.base_samples[::subsample]
            self.set_epoch()
        self.cls_idx = 1
    def set_epoch(self):
        random.shuffle(self.base_samples)
        if self.max_len is not None:
            self.samples = self.base_samples * (self.max_len // len(self.base_samples) + 1)
            self.samples = self.samples[:self.max_len]
            
    def __getitem__(self, index):
        img_path = self.samples[index]
        img1 = cv2.imread(img_path)[:,:,::-1]
        if self.samples[index][-5]=='1':
            img_path = img_path[:-7] + 'im2.png'
            img2 = cv2.imread(img_path)[:,:,::-1]
        elif self.samples[index][-5]=='2':
            img_path = img_path[:-7] + 'im3.png'
            img2 = cv2.imread(img_path)[:,:,::-1]
        img = np.concatenate([img1,img2], axis = -1)
        img1, img2 = torch.chunk(self.transform(img), 2)
        img = img1-img2+128/255
        return img,img, self.cls_idx

    def __len__(self):
        return len(self.samples)
    
@register_dataset("VimeoTripletY1Y2Y3Dataset")
class VimeoTripletY1Y2Y3Dataset(Dataset):
    def __init__(self, root, transform=None,split = 'train',subsample = None, max_len = None):
        self.base_samples = []
        self.max_len = max_len
        self.split = split
        
        if split == 'train':
            with open(root+'tri_trainlist.txt' ,'r') as f:
                for line in f:
                    self.base_samples.append(root+f'/sequences/{line.strip()}')
        else:
            with open(root+'tri_testlist.txt' ,'r') as f:
                for line in f:
                    self.base_samples.append(root+f'/sequences/{line.strip()}')
            self.samples = self.base_samples[:100]
            
        self.transform = transform
        if split=='train':
            if subsample is not None:
                self.base_samples = self.base_samples[::subsample]
            self.set_epoch()
        self.cls_idx = 1

    def set_epoch(self):
        random.shuffle(self.base_samples)
        if self.max_len is not None:
            self.samples = self.base_samples * (self.max_len // len(self.base_samples) + 1)
            self.samples = self.samples[:self.max_len]  
            
    def __getitem__(self, index):
        img_path = self.samples[index]
        img1 = cv2.imread(img_path+'/im1.png')# bgr
        img2 = cv2.imread(img_path+'/im2.png')# bgr
        img3 = cv2.imread(img_path+'/im3.png')# bgr
        img = np.concatenate([img1,img2,img3],axis=-1)
        img = self.transform(img)
        img1, img2, img3 = torch.chunk(img, 3, dim= 0)
        img1 = 0.2627*img1[2,:,:]+0.678*img1[1,:,:]+0.0593*img1[0,:,:]
        img2 = 0.2627*img2[2,:,:]+0.678*img2[1,:,:]+0.0593*img2[0,:,:]
        img3 = 0.2627*img3[2,:,:]+0.678*img3[1,:,:]+0.0593*img3[0,:,:]
        img = torch.stack([img1,img2,img3],dim=0)
        return img,img, self.cls_idx

    def __len__(self):
        return len(self.samples)
    
    
@register_dataset("DocumentDataset")
class DocumentDataset(Dataset):
    def __init__(self, root, transform=None,split = 'train',subsample = None, max_len = None):
        self.base_samples = []
        self.max_len = max_len
        self.split = split
        
        if split == 'train':
            with open(root+'train.list' ,'r') as f:
                for line in f:
                    self.base_samples.append(root+f'{line.strip()}')
        else:
            with open(root+'test.list' ,'r') as f:
                for line in f:
                    self.base_samples.append(root+f'{line.strip()}')
            self.samples = self.base_samples[:100]
                
        self.transform = transform
        if split=='train':
            if subsample is not None:
                self.base_samples = self.base_samples[::subsample]
            self.set_epoch()
        self.cls_idx = 3
                
    def set_epoch(self):
        random.shuffle(self.base_samples)
        if self.max_len is not None:
            self.samples = self.base_samples * (self.max_len // len(self.base_samples) + 1)
            self.samples = self.samples[:self.max_len]  
            
    def __getitem__(self, index):
        img = Image.open(self.samples[index]).convert("RGB")
        img = self.transform(img)
        return img, img, self.cls_idx

    def __len__(self):
        return len(self.samples)
    
@register_dataset("KodakNaturalDataset")
class KodakNaturalDataset(Dataset):
    def __init__(self, root, transform=None, subsample = None, max_len = None):
        
        self.samples = [root +f'/{sample}' for sample in os.listdir(root)]
        self.samples.sort()
        self.transform = transform
        self.cls_idx = 0

    def __getitem__(self, index):
        img = Image.open(self.samples[index]).convert("RGB")
        img = self.transform(img)
        return img,img, self.cls_idx

    def __len__(self):
        return len(self.samples)
    
@register_dataset("HQDataset")
class HQDataset(Dataset):
    def __init__(self, roots=[], transform=None, split='train',subsample = None, max_len = None):
        self.split = split
        self.base_samples = []
                
        self.max_len = max_len
        self.split = split
        if isinstance(roots, str):
            roots = [roots]
        for root in roots:
            if root.endswith('.list'):
                tmp_root = ('/').join(root.split('/')[:-1]) + '/'
                with open(root,'r')as f:
                    cur_data_list = f.readlines()
                    cur_data_list = [tmp_root + tmp.strip() for tmp in cur_data_list]
                self.base_samples = self.base_samples + cur_data_list
            else:
                self.base_samples = self.base_samples + glob(root+'/*.jpg') + glob(root+'/*.png') + glob(root+'/*.bmp')
            
        if split == 'train':
            self.transform = transform
            self.set_epoch()
        else:
            self.transform = transforms.ToTensor()
            self.samples = self.base_samples
            self.samples.sort()
        self.cls_idx = 0
        
    def set_epoch(self):
        random.shuffle(self.base_samples)
        if self.max_len is not None:
            self.samples = self.base_samples * (self.max_len // len(self.base_samples) + 1)
            self.samples = self.samples[:self.max_len]  
        else:
            self.samples = self.base_samples
            
    def __getitem__(self, index):
        img = Image.open(self.samples[index]).convert("RGB")
        img = self.transform(img)
        
        if self.split == 'train':
            return img, img, self.cls_idx
        
        _,H,W = img.shape
        if H != 0:
            if H % 256 != 0:
                H = H + 256 - (H % 256)
        if W != 0:
            if W % 256 != 0:
                W = W + 256 - (W % 256)
            
        crop_or_pad = transforms.CenterCrop((H,W))
        img = crop_or_pad(img)
        
        return img,img, self.cls_idx

    def __len__(self):
        return len(self.samples)
    
@register_dataset("LineDataset")
class LineDataset(Dataset):
    def __init__(self, root, transform=None, split='train',subsample = None, max_len = None):
        self.split = split
        self.base_samples = []
                
        self.max_len = max_len
        self.split = split

        if split == 'train':
            with open(root+'train.list','r') as f:
                book_list = f.readlines()
            book_list = [tmp.strip() for tmp in book_list]
            for book_name in book_list:
                tmp_list = glob(root+ '/images/' + book_name +'/*.jpg') + glob(root+ '/images/' + book_name +'/*.png')
                tmp_list.sort()
                tmp_list = tmp_list[1:-1]
                self.base_samples = self.base_samples + tmp_list
        elif split == 'test':
            with open(root+'test.list','r') as f:
                book_list = f.readlines()
            book_list = [tmp.strip() for tmp in book_list]
            for book_name in book_list:
                for i in range(1,21,2):
                    self.base_samples.append(root+ '/images/' + book_name +f'/{i:03d}.jpg')

        if split == 'train':
            self.transform = transform
            self.set_epoch()
        else:
            self.transform = transforms.ToTensor()
            self.samples = self.base_samples
        self.cls_idx = 0
        
    def set_epoch(self):
        random.shuffle(self.base_samples)
        if self.max_len is not None:
            self.samples = self.base_samples * (self.max_len // len(self.base_samples) + 1)
            self.samples = self.samples[:self.max_len]  
        else:
            self.samples = self.base_samples
            
    def __getitem__(self, index):
        img = Image.open(self.samples[index]).convert("RGB")
        img = self.transform(img)
        if self.split == 'train':
            return img,img, self.cls_idx
        
        _,H,W = img.shape
        if H != 0:
            if H % 256 != 0:
                H = H + 256 - (H % 256)
        if W != 0:
            if W % 256 != 0:
                W = W + 256 - (W % 256)
            
        crop_or_pad = transforms.CenterCrop((H,W))
        img = crop_or_pad(img)
        return img,img, self.cls_idx

    def __len__(self):
        return len(self.samples)
    
class HQDataset_cutmix_test(Dataset):
    def __init__(self, root='', add_root='', transform=None, split='train',subsample = None, max_len = None, num_class = 6):
        self.split = split
        self.base_samples = []
        self.add_samples = []
        
        self.base_samples = self.base_samples + glob(root+'/*.jpg') + glob(root+'/*.png')
        self.add_samples = self.add_samples + glob(add_root+'/*.jpg') + glob(add_root+'/*.png')
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop((256,256))
        ])
        self.transform_add = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop((120,120))
        ])
        
        self.base_samples = sorted(self.base_samples)[:5]
        self.add_samples = sorted(self.add_samples)[:5]
        
        self.base_samples_ = []
        self.add_samples_ = []
        for i in range(5):
            self.base_samples_ = self.base_samples_ + [self.base_samples[i]]*5
            self.add_samples_ = self.add_samples_ + self.add_samples
            
        self.cls_idx = onehot_convert([0],num_class,0.0)
        self.cls_idx_add = onehot_convert([0],num_class,0.0)
        self.num_class = num_class
        
    def __getitem__(self, index):
        img = Image.open(self.base_samples_[index]).convert("RGB")
        img = self.transform(img)
        
        add_img = Image.open(self.add_samples_[index]).convert("RGB")
        add_img = self.transform_add(add_img)
        
        img[:,68:188,68:188] = add_img
        modemap = self.cls_idx.view(self.num_class,1,1).expand(self.num_class, 256,256).clone()
        modemap[:,68:188,68:188] = self.cls_idx_add.view(self.num_class,1,1).expand(self.num_class, 120,120).clone()
        
        return img, modemap

    def __len__(self):
        return len(self.base_samples_)
        
def onehot_convert(cls_idx, max_cls = 6, eta = 0.0) :
    output = torch.ones(*torch.tensor(cls_idx).shape[:-1], max_cls, dtype=torch.float32)*eta
    output.scatter_(-1, torch.tensor(cls_idx), 1-(eta*(max_cls-1)))
    return output   

def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
    
class CutMix(Dataset):
    def __init__(self, dataset, num_class, num_mix=1, beta=1., prob=1.0):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index):
        img, cls_idx = self.dataset[index]
        _, h, w = img.shape
        cls_idx = cls_idx.view(self.num_class,1,1).repeat(1,h,w)
        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue
            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            img2, cls_idx2 = self.dataset[rand_index]
            cls_idx2 = cls_idx2.view(self.num_class,1,1).repeat(1,h,w)
            
            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            cls_idx[:, bbx1:bbx2, bby1:bby2] = cls_idx2[:, bbx1:bbx2, bby1:bby2]
            
        return img, cls_idx

    def __len__(self):
        return len(self.dataset)
