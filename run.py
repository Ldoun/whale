import subprocess
import os


subprocess.call([
    'python3',
    'train.py',
    '--batch_size', '24',
    '--total_epoch', '200',
    '--lr', '1e-3',
    '--n_fold', '5',
    '--log_every', '10',
    #'--valid_every', args.module_name,
    '--save_every', '10',
    '--ckt_folder', '../checkpoint',
    '--img_folder', '../LG_data/train',
    '--train_csv', '../LG_data/train.csv',
    '--reload_epoch_from', '0',
    '--reload_folder_from', '0',
    '--reload_model_from', '',
    '--model', 'efficientnet_b5',
    '--model_name', 'first_trial',
])
