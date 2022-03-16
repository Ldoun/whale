import pandas as pd
import torch
import numpy as np
import torch_xla
import torch_xla.core.xla_model as xm
from model.model import ClipImageEncoer

train_data = pd.read_csv('eda/train.csv')
image_names = []
np_file_names = []
data = pd.DataFrame()

save_every = 500

with torch.no_grad():
    for i, row in train_data.iterrows():
        image = row['image']
        in_id = row['individual_id']
        
        image_feature, logit_scale = ClipImageEncoer(image).encode_image(image)
        np_image_feature = image_feature.numpy()
        
        np.save(f'{i}.npy', np_image_feature)
        image_names.append(image)
        np_file_names.append(f'{i}.npy')
        
        if i % save_every == 0:
            print(f'{i}/{len(train_data)}')
            print(logit_scale)
            data['image'] = pd.Series(image_names)
            data['npy'] = pd.Series(np_file_names)