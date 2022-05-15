import numpy as np
import time
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, lr_scheduler, config)
        self.config = config
        self.device = device
        self.n_samples = len(data_loader)
        self.data_loader = data_loader
        
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        if config['trainer']['log_step']:
            self.log_step = config['trainer']['log_step']
        else:
            self.log_step = int(np.sqrt(config['data_loader']['batch_size']))
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        
        
    #single gpu (multi gpu training X)
    def _compute_loss(self, image1, image2, ids):
        image1_feature, image2_feature, logit_scale = self.model(image1,image2)
        logit_scale = logit_scale.mean()
               
        one,zero = torch.ones(len(ids), dtype=torch.float, device=self.device), torch.zeros(len(ids), dtype=torch.float, device=self.device)
        ground_truth = torch.stack([torch.where(ids==ids[i], one, zero) for i in range(len(ids))])
        
        logits_per_image = logit_scale * image1_feature @ image2_feature.t()
        #ground_truth = torch.arange(len(logits_per_image)).long().to(self.device)
        
        loss = self.criterion(logits_per_image, ground_truth)
        return loss, logits_per_image, ground_truth
            
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        start = time.time()
        for batch_idx, (image1, image2, ids) in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            loss, logit, ground_truth = self._compute_loss(image1, image2, ids)
            loss.backward()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(logit, ground_truth))
        
            if batch_idx % self.log_step == 0:
                run_time = time.time() - start
                start = time.time()
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} Run time {:.2f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    run_time))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
            
            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
        
        return log
    
    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        
        with torch.no_grad():
            for batch_idx, (image1, image2, ids) in enumerate(self.valid_data_loader):
                loss, logit, ground_truth = self._compute_loss(image1, image2, ids)
                
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(logit, ground_truth))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
        
        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()
    
    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):#iteration mode
            current = batch_idx * self.data_loader.n_samples
            total = self.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)