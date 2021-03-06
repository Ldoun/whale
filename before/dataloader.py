import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from meta_data import label_encoder
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ImageDataset(Dataset):
    def __init__(
        self,
        image_root_path,
        dataframe,
        image_size,
        mode = 'train',
    ):
        
        super().__init__()
        self.path = image_root_path
        self.data = dataframe.copy()
        self.mode = mode
        self.label_encoder = label_encoder
        
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
        image = np.array(Image.open(os.path.join(self.path,self.data.iloc[idx]['image'])).convert('RGB'))
        label = self.label_encoder[self.data.iloc[idx]['individual_id']]
        
        #image = self.transforms(image)
        image = self.transforms(image=image)["image"]
        label = torch.tensor(label)
        
        return {'image':image, 'label':label}
    