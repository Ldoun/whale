import os
import numpy as np
from PIL import Image
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from base import BaseDataLoader

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = ImageDataset(self.data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class ImageDataset(Dataset):
    def __init__(
        self,
        data_dir,
        mode='valid'
    ):
        super().__init__()
        self.path = data_dir
        self.data = pd.read_csv('./test.csv')
        self.mode = mode
        unique = list(set(self.data['individual_id']))

        label_encoder = {}
        for i, value in enumerate(unique):
            label_encoder[value] = i
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
                A.Resize(448, 448),
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
                A.Resize(448, 448),
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
        
        return image, label