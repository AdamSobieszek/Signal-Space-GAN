# %load_ext autoreload
# %autoreload 2

import os
import joblib
import sys
import pickle

from tqdm import tqdm
import sys
sys.path.append('..')
from braindecode.datautil.iterators import get_balanced_batches
from modules.wgan import Generator, Discriminator


from utils.util import weight_filler
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
import argparse
from dataset.dataset import ProcessedEEGDataset
# check cuda
matplotlib.use('TKAgg') # for OSX

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Signal EEGAN')


### data and file related arguments
parser.add_argument('--n_critic', type=int, default=1,                                                              help = 'starting number of critics')
parser.add_argument('--rampup', type=int, default=100,                                                              help = 'alpha args.rampup')
parser.add_argument('--seed', type=int, default=0,                                                                  help = 'number of epochs')
parser.add_argument('--block_epochs', type=list, default = [150, 100, 200, 200, 400, 800],                          help = 'epochs per block')
parser.add_argument('--n_batch', type=int, default=2648 * 8,                                                        help = 'number of batches')

# paths
parser.add_argument('--cuda_path', type=str, default = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4", help = 'path to CUDA')
parser.add_argument('--data_path', type=str, default = r"D:\data\workshops\eeg",                                     help = 'path to binary data')
parser.add_argument('--model_path', type=str, default = './test.cnt',                                               help = 'model path')

# generator and discriminator arguments
parser.add_argument('--n_blocks', type=int, default=6, help='number of documents in a batch for training')
parser.add_argument('--n_chans', type=int, default=1, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=(2648 * 8), help='number of epochs')
parser.add_argument('--n_z', type=int, default=16, help='line 153')
parser.add_argument('--in_filters', type=int, default=50, help='number of epochs')
parser.add_argument('--out_filters', type=int, default=50, help='number of epochs')
parser.add_argument('--factor', type=int, default=2, help='number of epochs')
parser.add_argument('--num_map_layer', type=int, default=2, help='number of epochs')


# scheduler
parser.add_argument("--scheduler", type = bool, default=False,                                                      help = "scheduler")
parser.add_argument("--warmup_steps", type = float, default=600,                                                    help = "warmup steps")

# stuff
parser.add_argument("--mode", default='client',                                                                     help = "client or server")
parser.add_argument("--port", default=52162,                                                                        help = "port")
parser.add_argument('--jobid', type=int, default=0,                                                                 help = 'current run identifier')

# WANDB
parser.add_argument("--wandb", type=bool, default=True,                                                             help = "wandb")
parser.add_argument("--wandb_project", default='eegan',                                                             help = "wandb project")
parser.add_argument("--entity", default='eegan',                                                                    help = "wandb entity")




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

# dodać assera żeby się nie zgubiło


modelname = 'Progressive%s'
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)
    
generator = Generator(args.n_blocks, args.n_chans, args.n_z, args.in_filters, args.out_filters, args.factor, args.num_map_layer)
discriminator = Discriminator(args.n_blocks, args.n_chans, args.in_filters, args.out_filters, args.factor)

num_steps_discriminator = args.block_epochs[0] * len(data) * args.batch_size
num_steps_generator = args.block_epochs[0] * len(data) * (args.batch_size / args.n_critic)

generator.train_init(alpha=args.l_r, betas=(0., 0.99), scheduler = args.scheduler, warmup_steps = args.warmup_steps, num_steps=num_steps_generator)
discriminator.train_init(alpha=args.l_r, betas=(0., 0.99), eps_center=0.001,
                         one_sided_penalty=True, distance_weighting=True, scheduler = args.scheduler,
                         warmup_steps = args.warmup_steps, num_steps = num_steps_discriminator)

generator = generator.apply(weight_filler)
discriminator = discriminator.apply(weight_filler)

i_block_tmp = 0
i_epoch_tmp = 0
generator.model.cur_block = i_block_tmp
discriminator.model.cur_block = args.n_blocks - 1 - i_block_tmp
fade_alpha = 1.
generator.model.alpha = fade_alpha
discriminator.model.alpha = fade_alpha
# print("Size of the training set:",train.shape)


# move shit to gpu
generator = generator.cuda() 
discriminator = discriminator.cuda()
generator.train()
discriminator.train()

# if args.wandb:
#     import wandb
#     wandb.init(project="args.wandb_project", args.entity=args.entity)
#     wandb.watch(generator, log_freq=5)

for i_block in range(i_block_tmp, args.n_blocks): ################# for blocks
    print("-----------------")

    c = 0

    for i_epoch in tqdm(range(i_epoch_tmp, args.block_epochs[i_block])): ################## for epochs
        i_epoch_tmp = 0
        for idx in range(len(data)):
            collate_loss_d = []
            collate_loss_g = []
            train = data[idx]
            # RESHAPE DATA ############################################################################################

            with torch.no_grad():
                train_tmp = discriminator.model.downsample_to_block(
                    # downsample the training data to the current block
                    torch.from_numpy(train).to(device),
                    discriminator.model.cur_block
                ).data.cpu()

            if fade_alpha < 1:
                fade_alpha += 1. / args.rampup
                generator.model.alpha = fade_alpha
                discriminator.model.alpha = fade_alpha

            batches = get_balanced_batches(train.shape[0], rng, True, batch_size=args.batch_size) # get the batches
            iters = max(int(len(batches) / args.n_critic), 1) # get the number of iterations

            for it in range(iters): ##################for iterations
                #critic training
                for i_critic in range(args.n_critic): # ############################## for critics
                    try:
                        train_batches = train_tmp[batches[it * args.n_critic + i_critic]] # get the batch
                    except IndexError:
                        continue
                    # LOOP
                    batch_real =  train_batches.requires_grad_(True).cuda()

                    z_vars = rng.normal(0, 1, size=(len(batches[it * args.n_critic + i_critic]), args.n_z)).astype(np.float32)
                    with torch.no_grad():
                        z_vars = torch.from_numpy(z_vars).cuda()

                    output = generator(z_vars)

                    batch_fake = output.data.requires_grad_(True).cuda()

                    loss_d = discriminator.train_batch(batch_real, batch_fake)

                #generator training
                z_vars = rng.normal(0, 1, size=(args.batch_size, args.n_z)).astype(np.float32)
                z_vars = z_vars.requires_grad_(True).cuda()
                loss_g = generator.train_batch(z_vars, discriminator)
                collate_loss_d.append(loss_d)
                collate_loss_g.append(loss_g)

        loss_d =  [np.mean([l[0] for l in collate_loss_d]), np.mean([l[1] for l in collate_loss_d]),  np.mean([l[2] for l in collate_loss_d])]
        loss_g = np.mean(collate_loss_g)

        losses_d.append(loss_d)
        losses_g.append(loss_g)

        if args.wandb:
            wandb.log(
                {
                    "Learning_rate": generator.optimizer,
                    "Loss_F": loss_d[0],
                    "Loss_R": loss_d[1],
                    "Penalty": loss_d[2],
                    "Generator Loss": loss_g
                }
            )

        if i_epoch % 10 == 0:

            generator.eval()
            discriminator.eval()

            print('Epoch: %d   Loss_F: %.3f   Loss_R: %.3f   Penalty: %.4f   Loss_G: %.3f' % (
            i_epoch, loss_d[0], loss_d[1], loss_d[2], loss_g))

            freqs_tmp = np.fft.rfftfreq(train_tmp.numpy().shape[2], d=1 / (250. / np.power(2, args.n_blocks - 1 - i_block)))
            train_fft = np.fft.rfft(train_tmp.numpy(), axis=2)
            train_amps = np.abs(train_fft).mean(axis=3).mean(axis=0).squeeze()
            z_vars = Variable(torch.from_numpy(z_vars_im), volatile=True).cuda()
            batch_fake = generator(z_vars)
            fake_fft = np.fft.rfft(batch_fake.data.cpu().numpy(), axis=2)
            batch_fake = batch_fake.data.cpu().numpy()

            plot_stuff(fake_fft, freqs_tmp, i_block, i_epoch, batch_fake, args.model_path, model_name, args.jobid, train_amps)

            discriminator.save_model(os.path.join(modelpath, modelname % args.jobid + '.disc'))
            generator.save_model(os.path.join(modelpath, modelname % args.jobid + '.gen'))

            generator.train()
            discriminator.train()


    fade_alpha = 0.
    generator.model.cur_block += 1
    discriminator.model.cur_block -= 1



    args.n_critic+=1
    if i_block in [0,1,2]:
        args.batch_size //= 3
    if i_block in [3]:
        args.batch_size = 20
    print(args.batch_size)

    # reset learning rate and scheduler for next block
    discriminator.reset_parameters(new_num_steps= args.block_epochs[i_block] * len(data) * args.batch_size)
    generator.reset_parameters(new_num_steps= args.block_epochs[i_block] * len(data) * (args.batch_size / args.n_critic))