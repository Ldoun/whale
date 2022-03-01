import argparse
import collections
import torch
from torch.utils.data import DataLoader
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import data_loader.data_loader as module_data
import model.loss as module_loss
import model.scheduler as module_scheduler
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(index, config):
    config.init_logger()
    logger = config.get_logger('train')
    
    data = pd.read_csv(config['csv'])
    train_data, valid_data = train_test_split(data, test_size=0.2)
    
    # setup data_loader instances
    train_dataset = config.init_obj('dataset', module_data, mode = 'train', shuffle=True, dataframe = train_data)
    valid_dataset = config.init_obj('dataset', module_data, mode = 'valid', shuffle=False, dataframe = valid_data)
    
    train_dataloader, valid_dataloader = module_data.get_data_loaders(train_dataset, valid_dataset, config)
    
    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) TPU training
    device = xm.xla_device()
    model = model.to(device)
    
    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', module_scheduler, optimizer)

    #loads the training data onto each tpu
    train_dataloader = pl.MpDeviceLoader(train_dataloader,device)
    valid_dataloader = pl.MpDeviceLoader(valid_dataloader,device)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=train_dataloader,
                      valid_data_loader=valid_dataloader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;batch_size'),
        CustomArgs(['--log_step'], type=int, target='trainer;log_step')
    ]
    config = ConfigParser.from_args(args, options)
    xmp.spawn(main, args=(config,), nprocs=config["nprocs"])   