# coding=utf-8
import torch
import torch.nn as nn
import torch.autograd as autograd
import eeggan.utils.util as utils
from torch.autograd import Variable
from torch import optim
from eeggan.modules.progressive import(
    ProgressiveGenerator,
    GeneratorBlocks,
    ProgressiveDiscriminator,
    DiscriminatorBlocks
)
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from eeggan.modules.layers.mapping_network import MappingNetwork
import os

class GAN_Module(nn.Module):
    """
    Parent module for different GANs

    Attributes
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer for training the model parameters
    loss : torch.nn.Loss
        Loss function
    """
    def __init__(self):
        super(GAN_Module, self).__init__()

        self.did_init_train = False

    def save_model(self,fname, folder):
        """
        Saves `state_dict` of model and optimizer

        Parameters
        ----------
        fname : str
            Filename to save
        """
        cuda = False
        if next(self.parameters()).is_cuda: cuda = True
        cpu_model = self.cpu()
        model_state = cpu_model.state_dict()
        opt_state = cpu_model.optimizer.state_dict()
        # if a file exists, delete it
        if not os.path.exists(folder):
            os.makedirs(folder)

        torch.save((model_state,opt_state,self.did_init_train),fname)
        if cuda:
            self.cuda()


    def load_model(self,fname):
        """
        Loads `state_dict` of model and optimizer

        Parameters
        ----------
        fname : str
            Filename to load from
        """
        model_state,opt_state,self.did_init_train = torch.load(fname)

        self.load_state_dict(model_state)
        self.optimizer.load_state_dict(opt_state)


class WGAN_I_Discriminator(GAN_Module):
    """
    Improved Wasserstein GAN discriminator
    References
    ----------
    Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017).
    Improved Training of Wasserstein GANs.
    Retrieved from http://arxiv.org/abs/1704.00028
    """
    def __init__(self):
        super(WGAN_I_Discriminator, self).__init__()

    def train_init(self,alpha=1e-4,betas=(0.5,0.9),
                   lambd=10,one_sided_penalty=False,distance_weighting=False,
                   eps_drift=0.,eps_center=0.,lambd_consistency_term=0.,
                   scheduler = False,
                   warmup_steps = 600,
                   num_steps = 1000):
        """
        Initialize Adam optimizer for discriminator
        Parameters
        ----------
        alpha : float, optional
            Learning rate for Adam
        betas : (float,float), optional
            Betas for Adam
        lambda : float, optional
            Weight for gradient penalty (default: 10)
        one_sided_penalty : bool, optional
            Use one- or two-sided penalty
            See Hartmann et al., 2018 (default: False)
        distance_weighting : bool
            Use distance-weighting
            See Hartmann et al., 2018 (default: False)
        eps_drift : float, optional
            Weigth to keep discriminator output from drifting away from 0
            See Karras et al., 2017 (default: 0.)
        eps_center : float, optional
            Weight to keep discriminator centered at 0
            See Hartmann et al., 2018 (default: 0.)
        References
        ----------
        Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017).
        Progressive Growing of GANs for Improved Quality, Stability,
        and Variation. Retrieved from http://arxiv.org/abs/1710.10196
        Hartmann, K. G., Schirrmeister, R. T., & Ball, T. (2018).
        EEG-GAN: Generative adversarial networks for electroencephalograhic
        (EEG) brain signals. Retrieved from https://arxiv.org/abs/1806.01875
        """
        # super(WGAN_I_Discriminator,self).train_init(alpha,betas)
        self.alpha = alpha
        self.betas = betas
        self.warmup_steps = warmup_steps
        self.num_steps = num_steps
        self.scheduler = scheduler

        self.optimizer = optim.Adam(self.parameters(),lr=self.alpha,betas=self.betas)
        if scheduler:
            self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup (self.optimizer,
                                                num_warmup_steps = self.warmup_steps,
                                                num_training_steps = self.num_steps,
                                                num_cycles = 3)
        self.loss = torch.nn.BCELoss()
        self.did_init_train = True

        self.loss = None
        self.lambd = lambd
        self.one_sided_penalty = one_sided_penalty
        self.distance_weighting = distance_weighting
        self.eps_drift = eps_drift
        self.eps_center = eps_center
        self.lambd_consistency_term = lambd_consistency_term



    def pre_train(self):
        if not self.did_init_train:
            self.train_init()

        self.zero_grad()
        self.optimizer.zero_grad()
        for p in self.parameters():
            p.requires_grad = True

    def update_parameters(self):
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

    def reset_parameters(self, new_num_steps, new_warmup_steps):
        self.optimizer = optim.Adam(self.parameters(),lr = self.alpha, betas = self.betas)
        if self.scheduler:
            self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(self.optimizer,
                                                num_warmup_steps = new_warmup_steps,
                                                num_training_steps = new_num_steps,
                                                                    num_cycles=3)



    def train_batch(self, batch_real, batch_fake):
        """
        Train discriminator for one batch of real and fake data
        Parameters
        ----------
        batch_real : autograd.Variable
            Batch of real data
        batch_fake : autograd.Variable
            Batch of fake data
        Returns
        -------
        loss_real : float
            WGAN loss for real data
        loss_fake : float
            WGAN loss for fake data
        loss_penalty : float
            Improved WGAN penalty term
        loss_drift : float
            Drifting penalty
        loss_center : float
            Center penalty
        """
        self.pre_train()

        one = torch.FloatTensor([1])
        mone = one * -1

        batch_real,one,mone = utils.cuda_check([batch_real,one,mone])

        fx_real = self(batch_real)
        loss_real = fx_real.mean().reshape(-1)
        loss_real.backward(mone,
                           retain_graph=(self.eps_drift>0 or self.eps_center>0))

        fx_fake = self(batch_fake)


        loss_fake = fx_fake.mean().reshape(-1)


        loss_fake.backward(one,
                           retain_graph=(self.eps_drift>0 or self.eps_center>0))

        # concat fx_real and fx_fake
        fx = torch.cat([fx_real, fx_fake], dim=0)
        zeros = torch.zeros_like(fx_fake)
        ones = torch.zeros_like(fx_real) + 1
        actual_values = torch.cat([ones, zeros], dim=0)

        truth_values = (fx > 0).float() == actual_values.float()
        accuracy = truth_values.sum() / len(truth_values)

        loss_drift = 0
        loss_center = 0
        if self.eps_drift>0:
            tmp_drift = self.eps_drift*loss_real**2
            tmp_drift.backward(retain_graph=self.eps_center>0)
            loss_drift = tmp_drift.data[0]
        if self.eps_center>0:
            tmp_center = (loss_real+loss_fake)
            tmp_center = self.eps_center*tmp_center**2
            tmp_center.backward()
            loss_center = tmp_center.data[0]

        #loss_consistency_term
        #if self.lambd_consistency_term>0:
        #	batch_real_1

        dist = 1
        if self.distance_weighting:
            dist = (loss_real-loss_fake).detach()
            dist = dist.clamp(min=0)
        loss_penalty = self.calc_gradient_penalty(batch_real, batch_fake)
        loss_penalty = self.lambd*dist*loss_penalty
        loss_penalty.backward()

        # Update parameters
        self.update_parameters()

        loss_real = -loss_real.data[0]
        loss_fake = loss_fake.data[0]
        loss_penalty = loss_penalty.data[0]
        return loss_real,loss_fake,loss_penalty,loss_drift,loss_center, accuracy # return loss


    def calc_gradient_penalty(self, batch_real, batch_fake):
        """
        Improved WGAN gradient penalty
        Parameters
        ----------
        batch_real : autograd.Variable
            Batch of real data
        batch_fake : autograd.Variable
            Batch of fake data
        Returns
        -------
        gradient_penalty : autograd.Variable
            Gradient penalties
        """
        alpha = torch.rand(batch_real.data.size(0),*((len(batch_real.data.size())-1)*[1]))
        alpha = alpha.expand(batch_real.data.size())
        batch_real,alpha = utils.cuda_check([batch_real,alpha])

        interpolates = alpha * batch_real.data + ((1 - alpha) * batch_fake.data)
        interpolates = Variable(interpolates, requires_grad=True)
        alpha,interpolates = utils.cuda_check([alpha,interpolates])

        disc_interpolates = self(interpolates)

        ones = torch.ones(disc_interpolates.size())
        interpolates,ones = utils.cuda_check([interpolates,ones])

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=ones,
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        tmp = (gradients.norm(2, dim=1) - 1)
        if self.one_sided_penalty:
            tmp = tmp.clamp(min=0)
        gradient_penalty = ((tmp) ** 2).mean()

        return gradient_penalty


class WGAN_I_Generator(GAN_Module):
    """
    Improved Wasserstein GAN generator
    References
    ----------
    Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017).
    Improved Training of Wasserstein GANs.
    Retrieved from http://arxiv.org/abs/1704.00028
    """
    def __init__(self):
        super(WGAN_I_Generator, self).__init__()

    def train_init(self, alpha=1e-4, betas=(0.5,0.9),
                   scheduler = None,
                   warmup_steps = 600,
                   num_steps = 400):
        """
        Initialize Adam optimizer for generator
        Parameters
        ----------
        alpha : float, optional
            Learning rate for Adam
        betas : (float,float), optional
            Betas for Adam
        """
        self.alpha = alpha
        self.betas = betas
        self.warmup_steps = warmup_steps
        self.num_steps = num_steps
        self.scheduler = scheduler

        self.optimizer = optim.Adam(self.parameters(),lr=alpha,betas=betas)

        self.optimizer = optim.Adam(self.parameters(),lr=self.alpha,betas=self.betas)
        if scheduler:
            self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(self.optimizer,
                                                num_warmup_steps = self.warmup_steps,
                                                num_training_steps = self.num_steps,
                                                num_cycles=3)

        self.loss = None

        self.did_init_train = True

    def pre_train(self,discriminator):
        if not self.did_init_train:
            self.train_init()

        self.zero_grad()
        self.optimizer.zero_grad()
        for p in discriminator.parameters():
            p.requires_grad = False  # to avoid computation

    def update_parameters(self):
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

    def reset_parameters(self, new_num_steps, new_warmup_steps):
        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha, betas=self.betas)
        if self.scheduler:
            self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(self.optimizer,
                                                             num_warmup_steps=new_warmup_steps,
                                                             num_training_steps=new_num_steps,
                                                                                num_cycles=3)

    def train_batch(self, batch_noise, discriminator):
        """
        Train generator for one batch of latent noise
        Parameters
        ----------
        batch_noise : autograd.Variable
            Batch of latent noise
        discriminator : nn.Module
            Discriminator to evaluate realness of generated data
        Returns
        -------
        loss : float
            WGAN loss against evaluation of discriminator of generated samples
            to be real
        """

        self.pre_train(discriminator)

        mone = torch.FloatTensor([1]) * -1
        batch_noise, mone = utils.cuda_check([batch_noise,mone])

        # Generate and discriminate
        gen = self(batch_noise)
        disc = discriminator(gen)
        loss = disc.mean().reshape(-1)
        # Backprop gradient
        loss.backward(mone)

        # Update parameters
        self.update_parameters()

        loss = loss.data[0]
        return loss # return loss

class Generator(WGAN_I_Generator):
    def __init__(
        self, n_blocks:int, n_chans:int,
        n_z:int, in_filters:int,
        out_filters:int, factor:int,
        num_map_layers:int
    ):

        super(Generator,self).__init__()
        prog = GeneratorBlocks(
            n_blocks=n_blocks,
            n_chans=n_chans,
            z_vars=n_z,
            in_filters=in_filters,
            out_filters=out_filters,
            factor=factor
        )
        blocks = prog.get_blocks()

        self.model = ProgressiveGenerator(blocks)
        self.mapping = MappingNetwork(
            n_z, n_z,
            num_layers=num_map_layers
        )
        self.num_map_layers = num_map_layers
        self.pl_mean = 0

    def forward(self,input, truncation_psi = 1):
        if self.num_map_layers ==0:
            out = input
        else:
            out = self.mapping(input, truncation_psi)
        return self.model(out)

class Discriminator(WGAN_I_Discriminator):
    def __init__(
    self, n_blocks:int, n_chans:int,
    in_filters:int, out_filters:int,
    factor:int
    ):
        super(Discriminator,self).__init__()
        prog = DiscriminatorBlocks(
            n_blocks=n_blocks,
            n_chans=n_chans,
            in_filters=50,
            out_filters=50,
            factor=factor
        )
        blocks = prog.get_blocks()
        self.model = ProgressiveDiscriminator(blocks)

    def forward(self,input):
        return self.model(input)
