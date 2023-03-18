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

class DecentGANModel(BaseModel):
    """ This class implements DECENT model, described in the paper
    Unpaired Image-to-Image Translation with Density Changing Regularization
    Shaoan Xie, Qirong Ho, Kun Zhang
    NeurIPS 2022

    The code borrows heavily from the PyTorch implementation of CycleGAN and CUT
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss')
        parser.add_argument('--lambda_var', type=float, default=0.01, help='weight for variance loss')
        parser.add_argument('--lambda_idt', type=float, default=10.0, help='weight for identity loss')
        parser.add_argument('--var_layers', type=str, default='0,4,8,12,16', help='compute density loss on which layers')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flow_type', type=str, default='bnaf', help='flow type to estimate density')
        parser.add_argument('--maf_dim', type=int, default=1024, help='dimension of MAF')
        parser.add_argument('--maf_layers', type=int, default=2, help='number of layers in MAF')
        parser.add_argument('--maf_comps', type=int, default=10, help='number of components in MAF')
        parser.add_argument('--flow_blocks', type=int, default=1, help='number of blocks in flow model')
        parser.add_argument('--bnaf_layers', type=int, default=0, help='number of layers in BNAF')
        parser.add_argument('--bnaf_dim', type=int, default=10, help='dimension of BNAF')
        parser.add_argument('--flow_lr', type=float, default=1e-3, help='learning rate for flow')
        parser.add_argument('--flow_ema', type=float, default=0.998, help='exponential moving average rate for flow')
        parser.add_argument('--var_all', action='store_true', help='compute var on all images or single image')
        parser.add_argument('--tag', type=str, default='debug', help='tag to recognize the checkpoints')
        parser.set_defaults(pool_size=0)  # no image pooling
        opt, _ = parser.parse_known_args()
        dataset = os.path.basename(opt.dataroot.strip('/'))
        model_id = '%s_%s/var%s_np%s_nb%s_nl%s_nd%s_lr%s_ema%s' % (dataset, opt.direction,
                    opt.lambda_var, opt.num_patches, opt.flow_blocks, opt.bnaf_layers,
                    opt.bnaf_dim, opt.flow_lr, opt.flow_ema)
        if opt.var_all:
            model_id += '_var_all'
        else:
            model_id += '_var_single'
        parser.set_defaults(name='%s/%s' % (opt.tag, model_id))

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'G_GAN', 'D_real', 'D_fake', 'idt', 'var', 'nll_A', 'nll_B', 'exp_A', 'exp_B']
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'idt_B']
        self.var_layers = [int(i) for i in self.opt.var_layers.split(',')]


        if self.isTrain:
            self.model_names = ['G', 'F_A', 'F_B', 'D']
            self.opt_names = ['G', 'D', 'F']
        else:  # during test time, only load G
            self.model_names = ['G', 'F_A', 'F_B']
            self.opt_names = ['F']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF_A = networks.PatchDensityEstimator(opt, self.gpu_ids)
        self.netF_B = networks.PatchDensityEstimator(opt, self.gpu_ids)
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
        if True:
            self.compute_F_loss().backward()                   # calculate graidents for F
            self.optimizer_F = Adam(chain(self.netF_A.parameters(), self.netF_B.parameters()),
                                            lr=self.opt.flow_lr, amsgrad=True, polyak=self.opt.flow_ema)
            self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update FA, FB
        self.set_requires_grad(self.netG, False)
        self.set_requires_grad(self.netF_A, True)
        self.set_requires_grad(self.netF_B, True)
        self.optimizer_F.zero_grad()
        self.loss_F = self.compute_F_loss()
        self.loss_F.backward()
        torch.nn.utils.clip_grad_norm_(self.netF_A.parameters(), max_norm=0.1)
        torch.nn.utils.clip_grad_norm_(self.netF_B.parameters(), max_norm=0.1)
        self.optimizer_F.step()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.optimizer_F.swap() # using exponential for estimating
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netG, True)
        self.set_requires_grad(self.netF_A, False)
        self.set_requires_grad(self.netF_B, False)
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        self.optimizer_F.swap() # swap back for normal training

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
        self.fake_B, self.patches_real_A = self.netG(self.real_A, layers=self.var_layers)
        self.idt_B, self.patches_real_B = self.netG(self.real_B, layers=self.var_layers)

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
        self.loss_var = self.calculate_var_loss()
        if self.opt.lambda_var > 0:
            self.loss_G += self.opt.lambda_var * (self.loss_var)
        return self.loss_G

    def compute_F_loss(self):
        log_probs_A, _, _ = self.netF_A(self.patches_real_A, self.opt.num_patches, None, detach=True)
        log_probs_B, _, _ = self.netF_B(self.patches_real_B, self.opt.num_patches, None, detach=True)
        self.loss_nll_A = 0.
        self.loss_nll_B = 0.
        for log_prob_a, log_prob_b, var_layer in zip(log_probs_A, log_probs_B, self.var_layers):
            #assert len(log_prob_a) == self.opt.num_patches
            self.loss_nll_A += (-log_prob_a.mean())
            self.loss_nll_B += (-log_prob_b.mean())
        self.loss_F = (self.loss_nll_A + self.loss_nll_B)/len(self.var_layers)
        return self.loss_F

    def calculate_var_loss(self):
        n_layers = len(self.var_layers)
        patches_fake_B = self.netG(self.fake_B, self.var_layers, encode_only=True)
        with torch.no_grad():
            log_probs_A, feat_lens, sample_ids = self.netF_A(self.patches_real_A, self.opt.num_patches, None, detach=True)
        log_probs_fake_B, _, _ = self.netF_B(patches_fake_B, self.opt.num_patches, sample_ids)
        total_var_loss = 0.0
        nll_A = 0 # only used to check exponential average density estimator works as expected
        nll_B = 0
        for log_prob_a, log_prob_b, feat_len, var_layer in zip(log_probs_A, log_probs_fake_B, feat_lens, self.var_layers):
            nll_A += -log_prob_a.mean()
            nll_B += -log_prob_b.mean()
            density_changes = (log_prob_a.detach() - log_prob_b).squeeze()
            density_changes_per_dim = density_changes/(feat_len.mean().item()*np.log(2))
            assert len(density_changes.size()) == 1
            assert len(density_changes) == self.opt.batch_size * self.opt.num_patches
            if self.opt.var_all:
                # compute density changing loss on all patches across different input images
                loss_layer = torch.var(density_changes_per_dim).mean()
            else:
                # compute density changing loss on patches on single input images, then average it
                density_changes_per_dim = density_changes_per_dim.view(self.opt.batch_size, self.opt.num_patches)
                loss_layer = torch.var(density_changes_per_dim, dim=-1).mean()
            total_var_loss += loss_layer
        self.loss_exp_A = nll_A
        self.loss_exp_B = nll_B
        return total_var_loss/n_layers

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
