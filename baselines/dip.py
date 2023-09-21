from __future__ import print_function

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import ehtplot

# !git clone https://github.com/DmitryUlyanov/deep-image-prior
# !mv deep-image-prior/* ./

import ehtplot.color
import matplotlib.pyplot as plt
import PIL
import os

import numpy as np
from models import *
from torch import Tensor
import torch
import torch.optim

# from numpy.lib.arraypad import _validate_lengths
# from skimage.measure import compare_psnr
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.denoising_utils import *
from utils.data_utils import get_true_and_noisy_data
from utils.eht_utils import loss_angle_diff
from utils.training_utils import loss_center

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

imsize =-1
PLOT = True
sigma = 25
sigma_ = sigma/255.

print(torch.__version__)

parser = argparse.ArgumentParser(description='Deep Image Prior baseline')
parser.add_argument('--task', type=str, default='closure-phase',
                    help='inverse problem to solve (default: closure-phase)')
parser.add_argument('--dataset', type=str, default='m87', help='dataset to use (default: m87)')
args = parser.parse_args()

fname = args.dataset
task = args.task
# task = 'compressed-sensing'
num_imgs = 60
curr_cmap = 'afmhot_10us'
image_size = 64
cphase_weight = 10
centroid_weight = 10

dip_baseline_dir = os.path.join('baseline_results', fname, f'dip-centroid{centroid_weight}')
if not os.path.exists(dip_baseline_dir):
    os.makedirs(dip_baseline_dir)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_centroid_fit = loss_center(device, center=image_size/2-0.5, dim=image_size)


def forward_model(A, x, task, idx=0, dataset=None):
    if task == 'compressed-sensing' and dataset=="sagA_video":
        y = torch.einsum('ab,bcd->acd', x.reshape(x.shape[0], x.shape[2]*x.shape[2]), A[idx])
    elif task == 'compressed-sensing' and dataset != "sagA_video":
        y = torch.einsum('ab,bcd->acd', x.reshape(x.shape[0], x.shape[2]*x.shape[2]), A)
    elif task == 'closure-phase':
        if dataset == 'm87' or dataset == 'sagA':
            y = A(x)
        elif dataset == 'sagA_video':
            y = A(x, idx=idx)
        else:
            raise ValueError('invalid dataset for task closure-phase')
    else:
        raise ValueError('invalid task')
    return y


def optimize(optimizer_type, parameters, closure, LR, num_iter):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')

        def closure2():
            optimizer.zero_grad()
            return closure()

        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)

        for j in range(num_iter + 1):
            optimizer.zero_grad()
            closure()
            optimizer.step()

            # if (j % (num_iter // 3) == 0):
            #     out_np = torch_to_np(net(net_input))
            #     plt.figure(figsize=(5, 5))
            #     plt.imshow(np.clip(out_np, 0, 1)[0], cmap=curr_cmap, vmin=0, vmax=1)
            #     plt.axis("off")
            #     plt.savefig(os.path.join(baselines_dir, f'{idx}_{j}.svg'), format='svg')
            #     plt.close()
            #     np.save(os.path.join(baselines_dir, f'{idx}_{j}.npy'), out_np)
    else:
        assert False


true_imgs, noisy_imgs, A, sigma, kernels =\
    get_true_and_noisy_data(image_size, (None, None), num_imgs, fname, 8,
                            'closure-phase', 'learning', True, cphase_count='min',
                            envelope_params=None)
if task != 'closure-phase':
    sigma = [x[np.newaxis,:,np.newaxis].type(dtype) for x in sigma[0]]
    true_imgs, noisy_imgs, A, _, kernels =\
        get_true_and_noisy_data(image_size, sigma, num_imgs, fname, 8, task,
                                'learning', True, cphase_count='min',
                                envelope_params=None)
    A = A.type(dtype)

for idx in range(num_imgs):
    img_noisy_np = noisy_imgs[idx]
    img_np = true_imgs[idx]
    img_noisy_np_image = img_noisy_np

    INPUT = 'noise'  # 'meshgrid'
    pad = 'reflection'
    OPT_OVER = 'net'  # 'net,input'

    reg_noise_std = 1. / 30.  # set to 1./20. for sigma=50
    LR = 0.001

    OPTIMIZER = 'adam'  # 'LBFGS'
    # OPTIMIZER = 'LBFGS'

    exp_weight = 0.99

    if task == "denoising":
        num_iter = 300  # 100
        show_every = 50
    else:
        num_iter = 3000
        show_every = 500
    input_depth = 32
    figsize = 4

    net = get_net(input_depth, 'skip', pad,
                  skip_n33d=128,
                  skip_n33u=128,
                  skip_n11=4,
                  num_scales=5,
                  n_channels=1,
                  upsample_mode='bilinear').type(dtype)

    # else:
    #     assert False
    net_input = get_noise(input_depth, INPUT, (image_size, image_size)).type(dtype).detach()

    # Compute number of parameters
    s = sum([np.prod(list(p.size())) for p in net.parameters()]);
    print('Number of params: %d' % s)

    # Loss
    mse = torch.nn.MSELoss().type(dtype)

    # img_noisy_torch = np_to_torch(img_noisy_np[0]).type(dtype)
    img_noisy_torch = img_noisy_np

    out = net(net_input)
    print(out.shape)

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    out_avg = None
    last_net = None
    psrn_noisy_last = 0

    i = 0


    def closure():

        global i, out_avg, psrn_noisy_last, last_net, net_input

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input)

        # Smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

        if task != 'closure-phase':
            total_loss = mse(forward_model(A, out, task, dataset=fname),
                             img_noisy_torch)
        else:
            recon = forward_model(A, out, task, dataset=fname)
            total_loss = mse(recon[0], img_noisy_torch[0]) \
                         + cphase_weight * loss_angle_diff(img_noisy_torch[1], recon[1], sigma[1][idx]) \
                         + centroid_weight * loss_centroid_fit(out)
        total_loss.backward()

        # psrn_noisy = compare_psnr(img_noisy_np_image, out.detach().cpu().numpy()[0], data_range=2)
        psrn_gt = compare_psnr(img_np.detach().cpu().numpy(),
                               out.detach().cpu().numpy()[0], data_range=2)
        psrn_gt_sm = compare_psnr(img_np.detach().cpu().numpy(),
                                  out_avg.detach().cpu().numpy()[0], data_range=2)

        # Note that we do not have GT for the "snail" example
        # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
        # print('Idx %05d Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (
        #     idx, i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
        print('Idx %05d Iteration %05d    Loss %f     PSRN_gt: %f PSNR_gt_sm: %f' % (
            idx, i, total_loss.item(), psrn_gt, psrn_gt_sm), '\r', end='')

        if PLOT and i % show_every == 0:
            out_np = torch_to_np(out)
            plot_image_grid([np.clip(out_np, 0, 1),
                             np.clip(torch_to_np(out_avg), 0, 1)], factor=figsize, nrow=1)

            plt.figure(figsize=(5, 5))
            plt.imshow(np.clip(out_np, 0, 1)[0], cmap=curr_cmap, vmin=0, vmax=1)
            plt.axis("off")
            plt.savefig(os.path.join(dip_baseline_dir, f'{idx}_{i}.svg'), format='svg')
            plt.close()
            np.save(os.path.join(dip_baseline_dir, f'{idx}_{i}.npy'), out_np)

        # Backtracking
        # if i % show_every:
        #     if psrn_noisy - psrn_noisy_last < -5:
        #         print('Falling back to previous checkpoint.')
        #
        #         for new_param, net_param in zip(last_net, net.parameters()):
        #             net_param.data.copy_(new_param.cuda())
        #
        #         return total_loss * 0
        #     else:
        #         last_net = [x.detach().cpu() for x in net.parameters()]
        #         psrn_noisy_last = psrn_noisy

        i += 1

        return total_loss


    p = get_params(OPT_OVER, net, net_input)
    optimize(OPTIMIZER, p, closure, LR, num_iter)

    out_np = torch_to_np(net(net_input))
    q = plot_image_grid([np.clip(out_np, 0, 1), img_np.detach().cpu().numpy()],
                        factor=13);

    plt.figure(figsize=(5, 5))
    plt.imshow(np.clip(out_np, 0, 1)[0], cmap=curr_cmap, vmin=0, vmax=1)
    plt.axis("off")
    # plt.savefig(os.path.join(baselines_dir, f'{idx}_{num_iter}.svg'), format='svg')
    plt.savefig(os.path.join(dip_baseline_dir, f'{str(idx).zfill(3)}.svg'), format='svg')
    plt.close()

    np.save(os.path.join(dip_baseline_dir, f'{str(idx).zfill(3)}.npy'), out_np)
