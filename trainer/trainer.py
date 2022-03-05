import numpy as np
import time
import torch
import torch_xla.core.xla_model as xm
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
        
    def _compute_loss(self, image1, image2, ids, rank, bs, validation=False):
        image1_feature, image2_feature, logit_scale = self.model(image1,image2)
        logit_scale = logit_scale.mean()
        
        gatered_image1_featuture = xm.all_gather(image1_feature)
        all_image1_featuture = torch.cat(
            [gatered_image1_featuture[:rank * bs]]
            +[image1_feature]
            + [gatered_image1_featuture[(rank + 1) *bs :]]
        )
        gatered_image2_featuture = xm.all_gather(image2_feature)
        all_image2_featuture = torch.cat(
            [gatered_image2_featuture[:rank * bs]]
            +[image2_feature]
            + [gatered_image2_featuture[(rank + 1) *bs :]]
        )
        
        gathered_ids = xm.all_gather(ids)
        one,zero = torch.ones(len(gathered_ids), dtype=torch.float, device=self.device), torch.zeros(len(gathered_ids), dtype=torch.float, device=self.device)
        ground_truth = torch.stack([torch.where(gathered_ids==gathered_ids[i], one, zero) for i in range(len(gathered_ids))])
        
        logits_per_image = logit_scale * all_image1_featuture @ all_image2_featuture.t()
        #ground_truth = torch.arange(len(logits_per_image)).long().to(self.device)
        if validation:
            loss = None
        else:
            loss = self.criterion(logits_per_image, ground_truth)
        return loss, logits_per_image[rank*bs:(rank+1)*bs, rank*bs:(rank+1)*bs], ground_truth[rank*bs:(rank+1)*bs]
        
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
            loss, logit, ground_truth = self._compute_loss(image1, image2, ids, xm.get_ordinal(), self.config['data_loader']['batch_size'])
            loss.backward()
            xm.optimizer_step(self.optimizer)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        
            if batch_idx % self.log_step == 0:
                float_loss = xm.mesh_reduce('loss_tensor_reduce',loss.item(),np.mean)
                run_time = time.time() - start
                start = time.time()
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} Run time {:.2f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    float_loss,
                    run_time))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                if self.writer is not None:
                    self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.train_metrics.update('loss', float_loss) #tpu lazy need to call item per nstep
                for met in self.metric_ftns:
                    met_score = xm.mesh_reduce('met_score', met(logit, ground_truth), np.mean)
                    self.train_metrics.update(met.__name__, met_score)
            
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
                loss, logit, ground_truth = self._compute_loss(image1, image2, ids, xm.get_ordinal(), self.config['data_loader']['batch_size'],validation=True)
                #self.valid_metrics.update('loss', xm.mesh_reduce('valid_loss_reduce',loss.item(),np.mean))
                for met in self.metric_ftns:
                    met_score = xm.mesh_reduce('valid_met_score', met(logit, ground_truth), np.mean)
                    self.valid_metrics.update(met.__name__, met_score)
                if self.writer is not None:
                    self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                    #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
        # add histogram of model parameters to the tensorboard
        if self.writer is not None:
            for name, p in self.model.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()
    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):#iteration mode
            current = batch_idx * self.config['data_loader']['batch_size']
            total = self.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)