import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from meta_data import label_encoder

class ImageDataset(Dataset):
    def __init__(
        self,
        image_root_path,
        dataframe,
        mode = 'train',
    ):
        
        super().__init__()
        self.path = image_root_path
        self.data = dataframe.copy()
        self.mode = mode
        self.label_encoder = label_encoder
        
        if self.mode == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize([384, 512]), #efficent net input size에 맞춰서 교체
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
        else:
            self.transforms = transforms.Compose([
                transforms.Resize([384, 512]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
                    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_root_path,self.data.iloc[idx]['image']))
        label = self.label_encoder[self.data.iloc[idx]['individual_id']]
        
        image = self.transforms(image)
        label = torch.LongTensor([label])
        
        return {'image':image, 'label':label}
    