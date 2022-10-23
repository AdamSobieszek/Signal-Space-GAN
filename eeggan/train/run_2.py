# %load_ext autoreload
# %autoreload 2

import os
import sys

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
import wandb

matplotlib.use('TKAgg')  # for OSX

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Signal EEGAN')

### data and file related arguments
parser.add_argument('--n_critic', type=int, default=2, help='starting number of critics')
parser.add_argument('--rampup', type=int, default=100, help='alpha args.rampup')
parser.add_argument('--seed', type=int, default=0, help='number of epochs')
parser.add_argument('--block_epochs', type=list, default=[150, 200, 300, 400, 600, 800], help='epochs per block')
parser.add_argument('--batch_block_list', type=list, default=[2648*8, 2648*8//3, 2648*8//9, 2648*8//27, 20, 2], help='batch size per block')

# paths
parser.add_argument('--data_path', type=str, default = r"D:\data\workshops\eeg2",                                     help = 'path to binary data')
parser.add_argument('--model_path', type=str, default = r'D:\data\models_brainhack',  help = 'model path')
parser.add_argument('--model_name', type=str, default = 'test',                                               help = 'model path')

# generator and discriminator arguments
parser.add_argument('--l_r', type=int, default=0.06, help='Learning rate')
parser.add_argument('--n_blocks', type=int, default=6, help='number of documents in a batch for training')
parser.add_argument('--n_chans', type=int, default=1, help='number of epochs')
parser.add_argument('--n_z', type=int, default=8, help='line 153')
parser.add_argument('--in_filters', type=int, default=50, help='number of epochs')
parser.add_argument('--out_filters', type=int, default=50, help='number of epochs')
parser.add_argument('--factor', type=int, default=2, help='number of epochs')
parser.add_argument('--num_map_layer', type=int, default=2, help='number of epochs')
parser.add_argument('--n_reg', type=int, default=3, help='number of epochs')

# scheduler
parser.add_argument("--scheduler", type=bool, default=True, help="scheduler")


parser.add_argument("--i_block_tmp", type=int, default=0, help="warmup steps")
parser.add_argument("--i_epoch_tmp", type=int, default=0, help="warmup steps")
parser.add_argument("--fade_alpha", type=float, default=1.0, help="warmup steps")

# stuff
parser.add_argument("--mode", default='client', help="client or server")
parser.add_argument("--port", default=52, help="port")
parser.add_argument('--jobid', type=int, default=0, help='current run identifier')

# WANDB
parser.add_argument("--wandb_enabled", type=bool, default=True, help="wandb")
parser.add_argument("--wandb_project", default='eegan', help="wandb project")
parser.add_argument("--entity", default='hubertp', help="wandb entity")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

if args.wandb_enabled:
    wandb.init(project=args.wandb_project, entity=args.entity, config=args)

# set seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
rng = np.random.RandomState(args.seed)

data = ProcessedEEGDataset(args.data_path)
# grid search
params_list = [[['l_r', 0.05], ['block_epochs', [11,11,11,11,11,11]], ['batch_block_list', [2648*16, 2648*16, 2648*16, 2648*16, 20, 2]], ['n_reg', 2]],
[['l_r', 0.05], ['block_epochs', [11, 11, 11, 11, 11, 11]], ['batch_block_list', [2648*16, 2648*16, 2648*8, 2648*8, 20, 2]], ['n_reg', 3]],
[['l_r', 0.05], ['block_epochs', [500, 700, 1000, 1400, 1600, 1800]], ['batch_block_list', [2648*16, 2648*16, 2648*8, 2648*8, 20, 2]], ['n_reg', 3]],
               [['l_r', 0.03], ['block_epochs', [1000, 1000, 1300, 1400, 1600, 1800]], ['batch_block_list', [2648*16, 2648*16, 2648*8, 2648*8//27, 20, 2]], ['n_reg', 5]]]


modelname = 'Progressive%s'
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

for params in params_list: #params to lista [block_epochs, lr,
    # args.l_r = params[0][1]
    # args.block_epochs = params[1][1]
    # args.batch_block_list = params[2][1]
    # args.n_reg = params[3][1]
    generator = Generator(args.n_blocks, args.n_chans, args.n_z, args.in_filters, args.out_filters, args.factor, args.num_map_layer)
    discriminator = Discriminator(args.n_blocks, args.n_chans, args.in_filters, args.out_filters, args.factor)

    num_steps_discriminator = args.block_epochs[0] * args.n_critic
    num_steps_generator = args.block_epochs[0]

    generator.train_init(alpha=args.l_r, betas=(0., 0.99), scheduler=args.scheduler, warmup_steps = num_steps_generator / 10,
                         num_steps=num_steps_generator)
    discriminator.train_init(alpha=args.l_r, betas=(0., 0.99), eps_center=0.001,
                             one_sided_penalty=True, distance_weighting=True, scheduler=args.scheduler,
                             warmup_steps = num_steps_discriminator / 10, num_steps=num_steps_discriminator)

    generator = generator.apply(weight_filler)
    discriminator = discriminator.apply(weight_filler)

    generator.model.cur_block = args.i_block_tmp
    discriminator.model.cur_block = args.n_blocks - 1 - args.i_block_tmp
    generator.model.alpha = args.fade_alpha
    discriminator.model.alpha = args.fade_alpha
    # print("Size of the training set:",train.shape)


    # move shit to gpu
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    generator.train()
    discriminator.train()



    training_loop(args.i_block_tmp, args.n_blocks,args.n_z, discriminator, generator, data, args.i_epoch_tmp, args.block_epochs,
                  args.rampup, args.fade_alpha, args.n_critic, args.n_reg, rng, device, args.jobid, str(params),args.wandb_enabled,
                  model_path = args.model_path, model_name = args.model_name, batch_list = args.batch_block_list)


