# %load_ext autoreload
# %autoreload 2

import os
import sys

sys.path.append('..')
sys.path.append('../..')
sys.path.append('')
from modules.wgan import Generator, Discriminator
from utils.util import weight_filler
import torch
import numpy as np
import matplotlib
import random
import argparse
from dataset.dataset import ProcessedEEGDataset
from training_loop import training_loop
from eeggan.config.conf import get_run_config


run_config = get_run_config()

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEED = run_config.config.seed
# set seeds
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
rng = np.random.RandomState(SEED)

data = ProcessedEEGDataset(run_config.paths.data_path)


modelname = 'Progressive%s'
if not os.path.exists(run_config.paths.model_path):
    os.makedirs(run_config.paths.model_path)


generator = Generator(run_config.config.n_blocks, 
                        run_config.config.n_chans, 
                        run_config.config.n_z, 
                        run_config.config.in_filters, 
                        run_config.config.out_filters, 
                        run_config.config.factor, 
                        run_config.config.num_map_layer)
discriminator = Discriminator(run_config.config.n_blocks, 
                                run_config.config.n_chans, 
                                run_config.config.in_filters, 
                                run_config.config.out_filters, 
                                run_config.config.factor)

num_steps_discriminator = run_config.config.block_epochs[0] * run_config.config.n_critic
num_steps_generator = run_config.config.block_epochs[0]

generator.train_init(alpha=run_config.config.l_r, 
                        betas=(0., 0.99), 
                        scheduler=run_config.config.scheduler, 
                        warmup_steps = num_steps_generator / 10,
                        num_steps=num_steps_generator)
discriminator.train_init(alpha=run_config.config.l_r, 
                            betas=(0., 0.99), 
                            eps_center=0.001,
                            one_sided_penalty=True, 
                            distance_weighting=True, 
                            scheduler=run_config.config.scheduler,
                            warmup_steps = num_steps_discriminator / 10, 
                            num_steps=num_steps_discriminator)

generator = generator.apply(weight_filler)
discriminator = discriminator.apply(weight_filler)

generator.model.cur_block = run_config.config.i_block_tmp
discriminator.model.cur_block = run_config.config.n_blocks - 1 - run_config.config.i_block_tmp
generator.model.alpha = run_config.config.fade_alpha
discriminator.model.alpha = run_config.config.fade_alpha
# print("Size of the training set:",train.shape)


# move shit to gpu
generator = generator.cuda()
discriminator = discriminator.cuda()
generator.train()
discriminator.train()



training_loop(run_config.config.i_block_tmp, 
                run_config.config.n_blocks,
                run_config.config.n_z, 
                discriminator, 
                generator, 
                data, 
                run_config.config.i_epoch_tmp, 
                run_config.config.block_epochs,
                run_config.config.rampup, 
                run_config.config.fade_alpha, 
                run_config.config.n_critic, 
                run_config.config.n_reg, 
                rng, 
                device, 
                str("Test"), #TODO: 
                model_path = run_config.paths.model_path, 
                model_name = run_config.paths.model_name, 
                batch_list = run_config.config.batch_block_list)
