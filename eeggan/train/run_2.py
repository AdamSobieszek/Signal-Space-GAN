# %load_ext autoreload
# %autoreload 2

import os
import sys
import pickle

from tqdm import tqdm
import sys

sys.path.append('..')
from modules.wgan import Generator, Discriminator

from utils.util import weight_filler
import torch
import numpy as np
import matplotlib
import random
import argparse
from dataset.dataset import ProcessedEEGDataset
from training_loop import training_loop

matplotlib.use('TKAgg')  # for OSX

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Signal EEGAN')

### data and file related arguments
parser.add_argument('--n_critic', type=int, default=1, help='starting number of critics')
parser.add_argument('--rampup', type=int, default=100, help='alpha args.rampup')
parser.add_argument('--seed', type=int, default=0, help='number of epochs')
parser.add_argument('--block_epochs', type=list, default=[150, 100, 200, 200, 400, 800], help='epochs per block')
parser.add_argument('--n_batch', type=int, default=2648 * 8, help='number of batches')

# paths
parser.add_argument('--cuda_path', type=str, default=r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3",
                    help='path to CUDA')
parser.add_argument('--data_path', type=str, default=r"C:\data\eegan\binary", help='path to binary data')
parser.add_argument('--model_path', type=str, default='./test.cnt', help='model path')

# generator and discriminator arguments
parser.add_argument('--n_blocks', type=int, default=6, help='number of documents in a batch for training')
parser.add_argument('--n_chans', type=int, default=1, help='number of epochs')
parser.add_argument('--n_batch', type=int, default=2648 * 8, help='number of epochs')
parser.add_argument('--n_z', type=int, default=16, help='line 153')
parser.add_argument('--in_filters', type=int, default=50, help='number of epochs')
parser.add_argument('--out_filters', type=int, default=50, help='number of epochs')
parser.add_argument('--factor', type=int, default=2, help='number of epochs')
parser.add_argument('--num_map_layer', type=int, default=2, help='number of epochs')

# scheduler
parser.add_argument("--scheduler", type=bool, default=False, help="scheduler")
parser.add_argument("--warmup_steps", type=float, default=600, help="warmup steps")


parser.add_argument("--i_block_temp", type=int, default=0, help="warmup steps")
parser.add_argument("--i_epoch_temp", type=int, default=0, help="warmup steps")
parser.add_argument("--fade_alpha", type=float, default=1.0, help="warmup steps")

# stuff
parser.add_argument("--mode", default='client', help="client or server")
parser.add_argument("--port", default=52162, help="port")
parser.add_argument('--jobid', type=int, default=0, help='current run identifier')

# WANDB
parser.add_argument("--wandb_enabled", type=bool, default=True, help="wandb")
parser.add_argument("--wandb_project", default='eegan', help="wandb project")
parser.add_argument("--entity", default='eegan', help="wandb entity")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()

os.environ["CUDA_PATH"] = args.cuda_path

# set seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
rng = np.random.RandomState(args.seed)

data = ProcessedEEGDataset(args.data_path)

# if not os.path.exists(compiled_data_path):
#     from dataset.dataset import EEGDataClass
#
#     dc = EEGDataClass(data_path)
#
#     train = np.vstack([e[0] for e in dc.events])
#     target = np.ones(train.shape[0]).astype(int)
#     data_tuple = (train, target)
#     pickle.dump(data_tuple, open(compiled_data_path, 'wb'))
#
# train, target = pickle.load(open(compiled_data_path, 'rb'))
#
# train = train[:, None, :, None].astype(np.float32)
#
# train = train - train.mean()
# train = train / train.std()

modelname = 'Progressive%s'
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

generator = Generator(n_blocks, n_chans, z_vars, factor, num_map_layer)
discriminator = Discriminator(n_blocks, n_chans, in_filters, out_filters, factor)

num_steps_discriminator = args.block_epochs[0] * len(data) * args.n_batch
num_steps_generator = args.block_epochs[0] * len(data) * (args.n_batch / args.n_critic)

generator.train_init(alpha=args.l_r, betas=(0., 0.99), scheduler=args.scheduler, warmup_steps=args.warmup_steps,
                     num_steps=num_steps_generator)
discriminator.train_init(alpha=args.l_r, betas=(0., 0.99), eps_center=0.001,
                         one_sided_penalty=True, distance_weighting=True, scheduler=args.scheduler,
                         warmup_steps=args.warmup_steps, num_steps=num_steps_discriminator)

generator = generator.apply(weight_filler)
discriminator = discriminator.apply(weight_filler)

i_block_tmp = 0
i_epoch_tmp = 0
generator.model.cur_block = i_block_tmp
discriminator.model.cur_block = args.n_blocks - 1 - i_block_tmp
generator.model.alpha = fade_alpha
discriminator.model.alpha = fade_alpha
# print("Size of the training set:",train.shape)


# move shit to gpu
generator = generator.cuda()
discriminator = discriminator.cuda()
generator.train()
discriminator.train()



training_loop(args.i_block_tmp, args.n_blocks, discriminator, generator, data, args.i_epoch_tmp, args.block_epochs,
              args.rampup, args.fade_alpha, args.m_critic, args.rng, args.n_batch, device, args.wandb_enabled,
              args.jobid, args.wandb_project, args.entity)
