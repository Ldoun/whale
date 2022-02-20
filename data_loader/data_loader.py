import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch_xla.core.xla_model as xm
from torchvision import transforms
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ImageDataset(Dataset):
    def __init__(
        self,
        data_dir,
        dataframe,
        image_size,
        mode = 'train',
    ):
        
        super().__init__()
        self.data_dir = data_dir
        self.data = dataframe.copy()
        self.mode = mode
        
        if self.mode == 'train':
            '''self.transforms = transforms.Compose([
                transforms.Resize([384, 512]), #efficent net input size에 맞춰서 교체
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])'''
            self.transforms = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.Normalize(
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225], 
                        max_pixel_value=255.0, 
                        p=1.0
                    ),
                ToTensorV2()], p=1.)
            
        else:
            '''self.transforms = transforms.Compose([
                transforms.Resize([384, 512]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])'''
            self.transforms = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225], 
                        max_pixel_value=255.0, 
                        p=1.0
                    ),
                ToTensorV2()], p=1.)
            
                    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image1 = np.array(Image.open(os.path.join(self.data_dir,self.data.iloc[idx]['image1'])).convert('RGB'))
        image2 = np.array(Image.open(os.path.join(self.data_dir,self.data.iloc[idx]['image2'])).convert('RGB'))
        
        #image = self.transforms(image)
        image1 = self.transforms(image=image1)["image"]
        image2 = self.transforms(image=image2)["image"]
        
        return image1, image2
    
    
def get_data_loaders(train_dataset, valid_dataset, config):
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )
  
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False
    )
  
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['data_loader']['batch_size'],
        sampler=train_sampler,
        num_workers=config['data_loader']['num_workers'],
        pin_memory=config['data_loader']['pin_memory'],
        drop_last=config['data_loader']['drop_last']
    )
    
    valid_dataloader = DataLoader(
        train_dataset,
        batch_size=config['data_loader']['batch_size'],
        sampler=valid_sampler,
        num_workers=config['data_loader']['num_workers'],
        pin_memory=config['data_loader']['pin_memory'],
        drop_last=config['data_loader']['drop_last'],
        shuffle=False
    )
    
    return train_dataloader, valid_dataloader
