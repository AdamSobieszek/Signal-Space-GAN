


def training_loop(i_block_tmp, n_blocks, discriminator, generator, train, i_epoch_tmp, block_epochs,
                  rampup, fade_alpha, n_critic, rng, n_batch, device):

    losses_d = []
    losses_g = []
    z_vars_im = rng.normal(0, 1, size=(n_batch, n_z)).astype(np.float32)  # initialize the z variables
    for i_block in range(i_block_tmp, n_blocks):  ################# for blocks
        print("-----------------")
        c = 0
        with torch.no_grad():
            train_tmp = discriminator.model.downsample_to_block(  # downsample the training data to the current block
                torch.from_numpy(train).to(device),
                discriminator.model.cur_block
            ).data.cpu()

        for i_epoch in tqdm(range(i_epoch_tmp, block_epochs[i_block])):  ################## for epochs
            i_epoch_tmp = 0

            if fade_alpha < 1:
                fade_alpha += 1. / rampup
                generator.model.alpha = fade_alpha
                discriminator.model.alpha = fade_alpha

            batches = get_balanced_batches(train.shape[0], rng, True, batch_size=n_batch)  # get the batches
            iters = max(int(len(batches) / n_critic), 1)  # get the number of iterations

            for it in range(iters):  ##################for iterations
                # critic training
                for i_critic in range(n_critic):
                    try:
                        train_batches = train_tmp[batches[it * n_critic + i_critic]]
                    except IndexError:
                        continue
                    batch_real = train_batches.requires_grad_(True).cuda()

                    z_vars = rng.normal(0, 1, size=(len(batches[it * n_critic + i_critic]), n_z)).astype(np.float32)
                    with torch.no_grad():
                        z_vars = torch.from_numpy(z_vars).cuda()

                    output = generator(z_vars)

                    batch_fake = output.data.requires_grad_(True).cuda()

                    loss_d = discriminator.train_batch(batch_real, batch_fake)

                # generator training

                z_vars = rng.normal(0, 1, size=(n_batch, n_z)).astype(np.float32)
                z_vars = z_vars.requires_grad_(True).cuda()
                loss_g = generator.train_batch(z_vars, discriminator)

            losses_d.append(loss_d)
            losses_g.append(loss_g)

            if wandb:
                wandb.log(
                    {
                        "Learning_rate": lr,
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

                plot_stuff(fake_fft, freqs_tmp, i_block, i_epoch, batch_fake, model_path, model_name, jobid, train_amps)

                discriminator.save_model(os.path.join(modelpath, modelname % jobid + '.disc'))
                generator.save_model(os.path.join(modelpath, modelname % jobid + '.gen'))

                generator.train()
                discriminator.train()

            lr /= 1.05
            lr = max(lr, 0.001)

        fade_alpha = 0.
        generator.model.cur_block += 1
        discriminator.model.cur_block -= 1
        lr = 0.01

        n_critic += 1
        if i_block in [0, 1, 2]:
            n_batch //= 3
        if i_block in [3]:
            n_batch = 20
        print(n_batch)