import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from itertools import chain
import os
from util.image_pool import ImagePool
from models.bnaf import Adam

class BaseGANModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN lossï¼šGAN(G(X))')
        parser.add_argument('--lambda_idt', type=float, default=10.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--tag', type=str, default='debug', help='weight for GAN lossï¼šGAN(G(X))')
        parser.set_defaults(pool_size=0)  # no image pooling
        opt, _ = parser.parse_known_args()
        dataset = os.path.basename(opt.dataroot.strip('/'))
        model_id = '%s_%s' % (dataset, opt.direction)
        parser.set_defaults(name='%s/%s' % (opt.tag, model_id))

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'idt']
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'idt_B']


        if self.isTrain:
            self.model_names = ['G', 'D']
            self.opt_names = []
        else:  # during test time, only load G
            self.model_names = ['G']
            self.opt_names = []

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        #print(self.netF_A)
        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netG, True)
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)
        self.idt_B = self.netG(self.real_B)

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        self.loss_G_GAN = self.criterionGAN(self.netD(self.fake_B), True).mean() * self.opt.lambda_GAN
        self.loss_idt = self.criterionIdt(self.idt_B, self.real_B) * self.opt.lambda_idt
        self.loss_G = self.loss_G_GAN + self.loss_idt
        return self.loss_G

    @torch.no_grad()
    def sample(self, x_A, x_B):
        if self.opt.direction != 'AtoB':
            x_A, x_B = x_B, x_A
        input_A, fake_B, input_B, idt_B = [], [], [], []
        for x_a, x_b in zip(x_A, x_B):
            x_a, x_b = x_a.unsqueeze(0), x_b.unsqueeze(0)
            fake_b = self.netG(x_a)
            idt_b = self.netG(x_b)
            input_A.append(x_a)
            input_B.append(x_b)
            fake_B.append(fake_b)
            idt_B.append(idt_b)
        return input_A, fake_B, input_B, idt_B
