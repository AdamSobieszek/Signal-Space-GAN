# coding=utf-8
from torch.autograd import Variable
from torch.nn import Module
import matplotlib.pyplot as plt
import numpy as np
import os


def cuda_check(module_list):
    """
    Checks if any module or variable in a list has cuda() true and if so
    moves complete list to cuda

    Parameters
    ----------
    module_list : list
        List of modules/variables

    Returns
    -------
    module_list_new : list
        Modules from module_list all moved to the same device
    """
    cuda = False
    for mod in module_list:
        if isinstance(mod,Variable): cuda = mod.is_cuda
        elif isinstance(mod,Module): cuda = next(mod.parameters()).is_cuda

        if cuda:
            break
    if not cuda:
        return module_list

    module_list_new = []
    for mod in module_list:
        module_list_new.append(mod.cuda())
    return module_list_new


def change_learning_rate(optimizer,lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def weight_filler(m):
    classname = m.__class__.__name__
    if classname.find('MultiConv') != -1:
        for conv in m.convs:
            conv.weight.data.normal_(0.0, 1.)
            if conv.bias is not None:
                conv.bias.data.fill_(0.)
    elif classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 1.) # From progressive GAN paper
        if m.bias is not None:
            m.bias.data.fill_(0.)
    elif classname.find('BatchNorm') != -1 or classname.find('LayerNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.)

def plot_stuff(fake_fft, freqs_tmp, i_block, i_epoch, batch_fake, model_path, model_name, jobid, train_amps):
    # debuger pdb
    # import pdb; pdb.set_trace()
    # fake_amps = np.abs(fake_fft).mean(axis=3).mean(axis = 0).squeeze()
    #
    # plt.figure()
    # plt.plot(freqs_tmp.squeeze().mean(axis = 0), np.log(fake_amps), label='Fake')
    # plt.plot(freqs_tmp.squeeze().mean(axis = 0), np.log(train_amps), label='Real')
    # plt.title(f'Frequency Spektrum, block {i_block}')
    # plt.xlabel('Hz')
    # plt.legend()
    # plt.savefig(os.path.join(model_path, model_name + '%' + str(jobid) + '_fft_%d_%d.png' + '%' + str(i_block) + str(i_epoch) + '.jpg'))
    # plt.close()
    # pdn debuger

    plt.figure(figsize=(10, 10))
    plt.title(f'Fake samples, block {i_block}')
    for i in range(10):
        plt.subplot(10, 1, i + 1)
        plt.plot(batch_fake[i].squeeze())
        plt.xticks((), ())
        plt.yticks((), ())
    plt.subplots_adjust(hspace=0)
    # check if file exists
    if os.path.isfile(os.path.join(model_path, model_name + '%' + str(jobid) + '_fake_%d_%d.png' + '%' + str(i_block) + str(i_epoch) + '.jpg')):
        os.remove(os.path.join(model_path, model_name + '%' + str(jobid) + '%' + str(i_block) + str(i_epoch) + '_fake_%d_%d.png'))
    plt.savefig(os.path.join(model_path, model_name + '%' + str(jobid) + '%' + str(i_block) + str(i_epoch) + '_fakes_%d_%d.png'))
    plt.close()

    plt.savefig(os.path.join(model_path, 'aaaa.jpg'))
