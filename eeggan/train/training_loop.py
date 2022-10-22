import torch
from torch.autograd import Variable
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
from utils.util import plot_stuff



def training_loop(i_block_tmp, n_blocks, n_z, discriminator, generator, data, i_epoch_tmp, block_epochs,
                  rampup, fade_alpha, n_critic, rng, n_batch, device, jobid, wandb_enabled = False,
                  model_path = None, model_name = None):
    losses_g = []
    losses_d = []
    for i_block in range(i_block_tmp, n_blocks):  ################# for blocks
        print("-----------------")

        c = 0
        full_epoch = ((len(data)//n_batch)+1)//n_critic
        try:
            _ = 1/full_epoch
        except:
            raise(Exception("Too many critic steps for this batch size and dataset size, decrease batch size or rewrite the code"))

        for i_epoch in tqdm(range(i_epoch_tmp, block_epochs[i_block])):  ################## for epochs
            i_epoch_tmp = 0

            if fade_alpha < 1 and i_epoch%full_epoch == 0:
                fade_alpha += 1. / rampup
                generator.model.alpha = fade_alpha
                discriminator.model.alpha = fade_alpha


            for idx, train_batch in enumerate(DataLoader(data, batch_size=n_batch, shuffle=True)):
                collate_loss_d = []
                collate_loss_g = []
                # RESHAPE DATA ############################################################################################

                with torch.no_grad():
                    train_batches = discriminator.model.downsample_to_block(
                        # downsample the training data to the current block
                        train_batch.to(device),
                        discriminator.model.cur_block
                    ).data

                batch_real = train_batches.requires_grad_(True).cuda()
                batch_real = batch_real.unsqueeze(1)

                z_vars = rng.normal(0, 1, size=(len(train_batches), n_z)).astype(
                    np.float32)
                with torch.no_grad():
                    z_vars = torch.from_numpy(z_vars).cuda()

                output = generator(z_vars)

                batch_fake = output.data.requires_grad_(True).cuda()

                loss_d = discriminator.train_batch(batch_real, batch_fake) ## TUTAJ ERROR!!


                if idx == n_critic:
                    #Break if critic got the given number of training steps
                    break

            # generator training
            z_vars = rng.normal(0, 1, size=(n_batch, n_z)).astype(np.float32)
            z_vars = torch.Tensor(z_vars).cuda().requires_grad_(True)
            loss_g = generator.train_batch(z_vars, discriminator)
            collate_loss_d.append(loss_d)
            collate_loss_g.append(loss_g)

            if i_epoch%full_epoch==0: # Dont save losses too often
                loss_d = [np.mean([l[0].cpu() for l in collate_loss_d]), np.mean([l[1].cpu() for l in collate_loss_d]),
                          np.mean([l[2].cpu() for l in collate_loss_d])]
                loss_g = np.mean([l.cpu() for l in collate_loss_g])

                losses_d.append(loss_d)
                losses_g.append(loss_g)

                if wandb_enabled:
                    wandb.log(
                        {
                            "generator l_r": generator.optimizer.param_groups[0]['lr'],
                            "discriminator l_r": discriminator.optimizer.param_groups[0]['lr'],
                            "Loss_F": loss_d[0],
                            "Loss_R": loss_d[1],
                            "Penalty": loss_d[2],
                            "Generator Loss": loss_g
                        }
                    )

            if i_epoch % (10)*full_epoch == 0:
                generator.eval()
                discriminator.eval()

                print('Epoch: %d   Loss_F: %.10f   Loss_R: %.10f   Penalty: %.10f   Loss_G: %.10f' % (
                    i_epoch, loss_d[0], loss_d[1], loss_d[2], loss_g))

                freqs_tmp = np.fft.rfftfreq(batch_real.cpu().detach().numpy().shape[2],
                                            d=1 / (250. / np.power(2, n_blocks - 1 - i_block)))
                # train_fft = np.fft.rfft(batch_real.numpy(), axis=2)
                train_amps = np.abs(freqs_tmp).mean(axis=3).mean(axis=0).squeeze()
                fake_fft = np.fft.rfft(batch_fake.cpu().detach().data.cpu().numpy(), axis=2)
                batch_fake = batch_fake.cpu().detach().data.cpu().numpy()

                plot_stuff(fake_fft, freqs_tmp, i_block, i_epoch, batch_fake, model_path, model_name, jobid,
                           train_amps)

                discriminator.save_model(os.path.join(model_path, model_name + '%' + str(jobid) + '.disc'))
                generator.save_model(os.path.join(model_path, model_name + '%' + str(jobid) + '.gen'))

                generator.train()
                discriminator.train()

        fade_alpha = 0.
        generator.model.cur_block += 1
        discriminator.model.cur_block -= 1

        n_critic += 1


        # reset learning rate and scheduler for next block
        discriminator.reset_parameters(new_num_steps = block_epochs[i_block], new_warmup_steps = block_epochs[i_block] / 10)
        generator.reset_parameters(new_num_steps = args.block_epochs[i_block] * args.n_critic, new_warmup_steps= (args.block_epochs[i_block] * args.n_critic) / 10)
