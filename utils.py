import numpy as np
import numpy.ma as ma
from math import log
import torch_xla.debug.metrics as met
import torch_xla.debug.metrics_compare_utils as mcu

def reset_weights(m):
	'''
	Try resetting model weights to avoid
	weight leakage.
	'''
	for layer in m.children():
		if hasattr(layer, 'reset_parameters'):
			#print(f'Reset trainable parameters of layer = {layer}')
			layer.reset_parameters()
   
def valid_logging(writer, epoch,total_epoch,step,n_iters, loss,correct,total):
    print(
		"epo:[%d/%d] itr:[%d/%d] Loss=%.5f Acc=%.3f"
		% ( 
			epoch,
			total_epoch,
			step,
			n_iters,
			loss,
			100.0 * (correct / total),
		), flush=True
	)
    
    if writer:
        writer.add_scalar('Loss/valid',
			loss,
            epoch * n_iters + step
		)
        
        writer.add_scalar('Accuracy/valid',
            100.0 * (correct / total), 
            epoch * n_iters + step
		)
   
def train_logging(writer, epoch,total_epoch,step,n_iters,elapsed, loss, write_xla_metrics=False):
    print(
		"train epo:[%d/%d] itr:[%d/%d] step_time:%ds Loss=%.5f"
		% ( 
			epoch,
			total_epoch,
			step,
			n_iters,
			elapsed,
			loss,
		), flush=True
    
    )
    
    if writer:
        writer.add_scalar('Loss/train',
			loss, 
            epoch * n_iters + step
		)
        
        if write_xla_metrics:
            metrics = mcu.parse_metrics_report(met.metrics_report())
            aten_ops_sum = 0
            for metric_name, metric_value in metrics.items():
                if metric_name.find('aten::') == 0:
                    aten_ops_sum += metric_value
                writer.add_scalar(metric_name, metric_value, epoch * n_iters + step)
            writer.add_scalar('aten_ops_sum', aten_ops_sum, epoch * n_iters + step)
