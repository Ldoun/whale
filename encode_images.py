import os
import torch
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model.model import ClipImageEncoer

try:
    print('using torch xla')
    import torch_xla
    import torch_xla.core.xla_model as xm
    
    device = xm.xla_device()
except:
    device = torch.device("cuda")

args = argparse.ArgumentParser(description='PyTorch Template')
args.add_argument('--image_path', default=None, type=str)
args.add_argument('--csv_file', default=None, type=str)
config = args.parse_args()

train_data = pd.read_csv(config.csv_file)
image_names = []
np_file_names = []
ids = []
data = pd.DataFrame()

save_every = 500

transforms = A.Compose([
    A.Resize(448, 448),
    A.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225], 
            max_pixel_value=255.0, 
            p=1.0
        ),
    ToTensorV2()], p=1.
)

model = ClipImageEncoer(
      embed_dim= 768,
      image_resolution= 448,
      vision_layers= [2,3,4,2],
      vision_width= 64,
      vision_patch_size= None
  ).to(device)

with torch.no_grad():
    for i, row in train_data.iterrows():
        image = np.array(Image.open(os.path.join(config.image_path,row['image'])).convert('RGB'))
        image = transforms(image=image)["image"].to(device)
        
        image_feature, logit_scale = model.encode_image(image)
        np_image_feature = image_feature.numpy()
        
        np.save(f'{i}.npy', np_image_feature)
        image_names.append(image)
        np_file_names.append(f'{i}.npy')
        ids.append(row['individual_id'])
        
        if i % save_every == 0:
            print(f'{i}/{len(train_data)}')
            print(logit_scale)
            data['image'] = pd.Series(image_names)
            data['npy'] = pd.Series(np_file_names)
            data['id'] = pd.Series(ids)