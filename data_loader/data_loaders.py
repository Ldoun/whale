import os
import numpy as np
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets, transforms
from base import BaseDataLoader
from meta_data import label_encoder

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
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class ImageDataset(BaseDataLoader):
    def __init__(
        self,
        data_dir,
        batch_size, 
        dataframe,
        image_size,
        validation_split=0.0,
        mode = 'train',
        shuffle=True,
        num_workers=1,
    ):
        
        self.path = data_dir
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
        super().__init__(self,batch_size, shuffle, validation_split, num_workers)
            
                    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = np.array(Image.open(os.path.join(self.path,self.data.iloc[idx]['image'])).convert('RGB'))
        label = self.label_encoder[self.data.iloc[idx]['individual_id']]
        
        #image = self.transforms(image)
        image = self.transforms(image=image)["image"]
        label = torch.tensor(label)
        
        return {'image':image, 'label':label}
    