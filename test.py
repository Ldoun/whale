import argparse
import pandas as pd
import time
import numpy as np
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch.utils.data import random_split

from models.efficientnet import efficientnet_base
from dataloader import ImageDataset
from utils import *
from loss import FocalLoss

def train_loop(args, writer, model, optimizer, criterion, train_dataloader, epoch):
    model.train()
    itr_start_time = time.time()
    n_iters = len(train_dataloader)
    
    losses = []
    
    for step, batch in enumerate(train_dataloader, start=1):
        optimizer.zero_grad()
        
        image = batch['image']
        label = batch['label'].squeeze()
        
        prediction = model(image)
        
        loss = criterion(prediction, label)
        loss.backward()
        
        xm.optimizer_step(optimizer)
        #optimizer.step()
        
        losses.append(loss.item())
        
        if step % args.log_every == 0:
            elapsed = time.time() - itr_start_time
            xm.add_step_closure(
                train_logging,      
                args=(writer,epoch,args.total_epoch,step,n_iters,elapsed, losses),
                run_async=True
            )
            
            losses = []
            itr_start_time = time.time()

def validation_loop(args, writer, model, criterion,valid_dataloader, epoch):
    model.eval()
    
    losses = []
    correct, total = 0,0
    n_iters = len(valid_dataloader)
    
    with torch.no_grad():
        for step, batch in enumerate(valid_dataloader, start=1):
            image = batch['image']
            label = batch['label'].squeeze()
            
            prediction = model(image)
            
            loss = criterion(prediction, label)
            
            losses.append(loss.item())
            
            _, predicted = torch.max(prediction.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        
            xm.add_step_closure(
                valid_logging,
                args=(writer, epoch, args.total_epoch, step, n_iters, losses, correct, total),
                run_async=True
            )
                
            losses = []
            correct, total = 0,0
            
def save_model(args, model, optimizer, fold, epoch):
    dict_for_infer = {
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        #"scaler": scheduler.state_dict(),
        #"amp": amp.state_dict(),
        "batch_size": args.batch_size,
        "epochs": args.total_epoch,
        "learning_rate": args.lr,
    }
    
    os.makedirs(args.ckt_folder, exist_ok=True)
    save_dir = os.path.join(args.ckt_folder, f"{args.model_name}-checkpoint_{fold}fold_{epoch}epoch")
    
    xm.save(dict_for_infer, save_dir)

    '''with open(os.path.join(args.ckt_folder, "dict_for_infer"), "wb") as f:
        pickle.dump(dict_for_infer, f)

    print("save complete")'''

    
def main(index, args):    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #torch.cuda.manual_seed(args.seed)
    #device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    device = xm.xla_device()
    
    data = pd.read_csv(args.train_csv)
    label = data['species'].values  # try with individual_id
    kfold = StratifiedKFold(n_splits=args.n_fold, shuffle=True)
    
    if args.reload_epoch_from:
        print(f'starting from {args.reload_folder_from} folder, {args.reload_epoch_from} epoch {args.reload_model_from} model')
    
    train, valid = random_split(data, test_size=0.5)
        
    writer = None
    if xm.is_master_ordinal():
        writer = test_utils.get_summary_writer(f'runs/{args.model_name}-5_5')

    train_dataset = ImageDataset(args.img_folder, train, 448, mode = 'train')        
    valid_dataset = ImageDataset(args.img_folder, valid, 448, mode = 'valid')
    
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
    
    train_dataloader = pl.MpDeviceLoader(
        DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=10,
            sampler=train_sampler,
            pin_memory=True,
            drop_last=True,
        ), device
    )
    
    valid_dataloader = pl.MpDeviceLoader(
        DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            num_workers=10,
            sampler=valid_sampler,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
        ), device
    )
    
    model = efficientnet_base(base_model=args.model,ouput_dim=15587)
    model.apply(reset_weights)
    model.to(device)
    
    lr = args.lr * xm.xrt_world_size()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = FocalLoss()
    
    start_epoch = 0
    if fold == args.reload_folder_from and args.reload_epoch_from:
        if args.reload_model_from == "":
            raise('need model file name')
        
        print('loading model')
        start_epoch = args.reload_epoch_from
        save_dir = os.path.join(args.ckt_folder, args.reload_model_from)
        checkpoint = torch.load(save_dir)
        optimizer.load_state_dict(checkpoint['opt'])
        model.load_state_dict(checkpoint['model'])
        

    for epoch in range(start_epoch, args.total_epoch + 1):
        train_loop(args, writer, model, optimizer, criterion, train_dataloader, epoch)
        validation_loop(args, writer, model, criterion, valid_dataloader, epoch)          
        if epoch % args.save_every == 0:
            if xm.is_master_ordinal():
                xm.add_step_closure(
                    save_model,
                    args = (args, model, optimizer, fold, epoch),   
                    run_async= False
                )
                
                print('saved')
                    
    test_utils.close_summary_writer(writer)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--total_epoch", type=int, default=40)
    parser.add_argument("--warmup_step", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_fold", type=int, default=5)
    parser.add_argument("--log_every", type=int, default=1)
    #parser.add_argument("--valid_every", type=int, default=10000)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--ckt_folder", type=str, default="./")
    parser.add_argument("--img_folder", type=str, default="./")
    parser.add_argument("--train_csv", type=str, default="./train.csv")
    parser.add_argument("--reload_epoch_from", type=int, default=0)
    parser.add_argument("--reload_folder_from", type=int, default=0)
    parser.add_argument("--reload_model_from", type=str, default='') 
    parser.add_argument("--model",type=str,default='',required=True)
    parser.add_argument("--model_name",type=str,default='',required=True)

    args = parser.parse_args()

    xmp.spawn(main, args=(args,))