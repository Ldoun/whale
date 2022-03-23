import enum
import os
import torch
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from model.model import ClipImageEncoer
from data_loader.data_loader import SingleImageDataloader

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp

except Exception as e:
    print(e)
    device = torch.device("cuda")

def encode_images(index, config):
    device = xm.xla_device()
    
    model = ClipImageEncoer(
        embed_dim= 768,
        image_resolution= 448,
        vision_layers= [2,3,4,2],
        vision_width= 64,
        vision_patch_size= None
    ).to(device)
    model.eval()

    data = pd.read_csv(config.csv_file)
    dataset = SingleImageDataloader(
        data_dir = config.image_path,
        dataframe = data,
        image_size = 448
    )

    sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=False
        )

    data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=10,
        pin_memory=False,
        drop_last=False,
        shuffle=False
    )

    data_loader = pl.MpDeviceLoader(data_loader,device)
    n_iter = len(data_loader)
    datafrmae = pd.DataFrame()
    np_file_names = []
    ids = []
    image_names = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            print(f'{batch_idx}/{n_iter}', flush=True)
            image, whale_id, image_name = data
            
            image_feature, logit_scale = model.encode_image(image)
            
            np_image_feature = image_feature.cpu().numpy()
            bs = image.shape[0]
            for i in range(bs):
                np.save(os.path.join(config.save_path, f'{index}_{batch_idx * config.batch_size + i}.npy'), np_image_feature[i,:])
                np_file_names.append(f'{index}_{batch_idx * config.batch_size + i}.npy')
                
            ids.extend(whale_id)
            image_names.extend(image_name)

            if (batch_idx + 1) % config.save_freq == 0:
                datafrmae = pd.DataFrame()
                datafrmae['image'] = pd.Series(image_names)
                datafrmae['npy'] = pd.Series(np_file_names)
                datafrmae['id'] = pd.Series(ids)
                datafrmae.to_csv(f'{index}_npy_image.csv', mode='w')
                
    datafrmae = pd.DataFrame()
    datafrmae['image'] = pd.Series(image_names)
    datafrmae['npy'] = pd.Series(np_file_names)
    datafrmae['id'] = pd.Series(ids)
    datafrmae.to_csv(f'{index}_npy_image.csv', mode='w')

        
    
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('--image_path', default=None, type=str)
    args.add_argument('--csv_file', default=None, type=str)
    args.add_argument('--save_path', default=None, type=str)
    args.add_argument('--save_freq', default=100, type=int)
    args.add_argument('--batch_size', default=32, type=int)
    args.add_argument('--use_xla', default=False, action='store_true')

    config = args.parse_args()

    if config.use_xla:
        print('using xla')
        xmp.spawn(encode_images, args=(config,), nprocs=8)
    
    else:
        print(f'using {device}')
        train_data = pd.read_csv(config.csv_file)
        
        data = pd.read_csv(config.csv_file)
        dataset = SingleImageDataloader(
            data_dir = config.image_path,
            dataframe = data,
            image_size = 448,
            train=False
        )

        data_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=10,
            pin_memory=False,
            drop_last=False,
            shuffle=False
        )

        model = ClipImageEncoer(
            embed_dim= 768,
            image_resolution= 448,
            vision_layers= [2,3,4,2],
            vision_width= 64,
            vision_patch_size= None
        ).to(device)
        model.eval()
        
        n_iter = len(data_loader)
        datafrmae = pd.DataFrame()
        np_file_names = []
        ids = []
        image_names = []

        with torch.no_grad():
            for batch_idx, data in enumerate(data_loader):
                print(f'{batch_idx}/{n_iter}', flush=True)
                image, image_name = data
                image = image.to(device)
            
                image_feature, logit_scale = model.encode_image(image)
                np_image_feature = image_feature.cpu().numpy()
                bs = image.shape[0]
                for i in range(bs):
                    np.save(os.path.join(config.save_path, f'test_{batch_idx * config.batch_size + i}.npy'), np_image_feature[i,:])
                    np_file_names.append(f'test_{batch_idx * config.batch_size + i}.npy')
                
                #ids.extend(whale_id)
                image_names.extend(image_name)
                
                if (batch_idx + 1) % config.save_freq == 0:
                    datafrmae = pd.DataFrame()
                    datafrmae['image'] = pd.Series(image_names)
                    datafrmae['npy'] = pd.Series(np_file_names)
                    #datafrmae['id'] = pd.Series(ids)
                    datafrmae.to_csv(f'test_npy_image.csv', mode='w')
                    
        datafrmae = pd.DataFrame()
        datafrmae['image'] = pd.Series(image_names)
        datafrmae['npy'] = pd.Series(np_file_names)
        #datafrmae['id'] = pd.Series(ids)
        datafrmae.to_csv(f'test_npy_image.csv', mode='w')