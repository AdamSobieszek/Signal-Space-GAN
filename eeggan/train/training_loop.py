import torch
from braindecode.datautil.iterators import get_balanced_batches
from torch.autograd import Variable
import wandb


def training_loop(i_block_tmp, n_blocks, discriminator, generator, data, i_epoch_tmp, block_epochs,
                  rampup, fade_alpha, n_critic, rng, n_batch, device, wandb_enabled = False, jobid,
                  wandb_project, entity):

    if wandb_enabled:
        wandb.init(project = wandb_project, entity = entity)
        wandb.watch(generator, log_freq=5)
    for i_block in range(i_block_tmp, n_blocks):  ################# for blocks
        print("-----------------")

        c = 0

        for i_epoch in tqdm(range(i_epoch_tmp, block_epochs[i_block])):  ################## for epochs
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
                    fade_alpha += 1. / rampup
                    generator.model.alpha = fade_alpha
                    discriminator.model.alpha = fade_alpha

                batches = get_balanced_batches(train.shape[0], rng, True, batch_size=n_batch)  # get the batches
                iters = max(int(len(batches) / n_critic), 1)  # get the number of iterations

                for it in range(iters):  ##################for iterations
                    # critic training
                    for i_critic in range(n_critic):  # ############################## for critics
                        try:
                            train_batches = train_tmp[batches[it * n_critic + i_critic]]  # get the batch
                        except IndexError:
                            continue
                        # LOOP
                        batch_real = train_batches.requires_grad_(True).cuda()

                        z_vars = rng.normal(0, 1, size=(len(batches[it * n_critic + i_critic]), n_z)).astype(
                            np.float32)
                        with torch.no_grad():
                            z_vars = torch.from_numpy(z_vars).cuda()

                        output = generator(z_vars)

                        batch_fake = output.data.requires_grad_(True).cuda()

                        loss_d = discriminator.train_batch(batch_real, batch_fake)

                    # generator training
                    z_vars = rng.normal(0, 1, size=(n_batch, n_z)).astype(np.float32)
                    z_vars = z_vars.requires_grad_(True).cuda()
                    loss_g = generator.train_batch(z_vars, discriminator)
                    collate_loss_d.append(loss_d)
                    collate_loss_g.append(loss_g)

            loss_d = [np.mean([l[0] for l in collate_loss_d]), np.mean([l[1] for l in collate_loss_d]),
                      np.mean([l[2] for l in collate_loss_d])]
            loss_g = np.mean(collate_loss_g)

            losses_d.append(loss_d)
            losses_g.append(loss_g)

            if wandb_enabled:
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

                freqs_tmp = np.fft.rfftfreq(train_tmp.numpy().shape[2],
                                            d=1 / (250. / np.power(2, n_blocks - 1 - i_block)))
                train_fft = np.fft.rfft(train_tmp.numpy(), axis=2)
                train_amps = np.abs(train_fft).mean(axis=3).mean(axis=0).squeeze()
                z_vars = Variable(torch.from_numpy(z_vars_im), volatile=True).cuda()
                batch_fake = generator(z_vars)
                fake_fft = np.fft.rfft(batch_fake.data.cpu().numpy(), axis=2)
                batch_fake = batch_fake.data.cpu().numpy()

                plot_stuff(fake_fft, freqs_tmp, i_block, i_epoch, batch_fake, model_path, model_name, jobid,
                           train_amps)

                discriminator.save_model(os.path.join(modelpath, modelname % jobid + '.disc'))
                generator.save_model(os.path.join(modelpath, modelname % jobid + '.gen'))

                generator.train()
                discriminator.train()

        fade_alpha = 0.
        generator.model.cur_block += 1
        discriminator.model.cur_block -= 1

        args.n_critic += 1
        if i_block in [0, 1, 2]:
            n_batch //= 3
        if i_block in [3]:
            n_batch = 20
        print(n_batch)

        # reset learning rate and scheduler for next block
        discriminator.reset_parameters(new_num_steps=args.block_epochs[i_block] * len(data) * n_batch)
        generator.reset_parameters(new_num_steps=args.block_epochs[i_block] * len(data) * (n_batch / n_critic)