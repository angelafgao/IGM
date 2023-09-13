import argparse
import os
import torch
torch.manual_seed(0)
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec

import numpy as np
import copy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.autograd import Variable
from . import flow_def
from . import training_utils
from . import data_utils

torch.set_default_dtype(torch.float32)
import torch.optim as optim
import pickle
import math
from torch import Tensor

# from sys import exit
import matplotlib.pyplot as plt
from torch.nn import functional as F

GPU = torch.cuda.is_available()
if GPU == True:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
    print("num GPUs",torch.cuda.device_count())
else:
    dtype = torch.FloatTensor

#def torch_forward_model(x, F, task):
#    if task == 'denoising':
#        return x
#    elif task == 'phase-retrieval':
#        y = torch.fft.fft2(x)
#        return y.abs()
    
#def loss_data_fit(A, x, y, task, sigma):
#    if task == 'denoising':
#        return 0.5 * torch.sum((x-y)**2 / sigma**2, (-1, -2))
#    elif task == 'phase-retrieval':
#        absFx = torch_forward_model(x, A, task)
#        return 0.5 * torch.sum((y - absFx)**2  / (sigma**2), 1)

def forward_model(A, x, task):
    if task == 'phase-retrieval':
        y_complex = torch.fft.fft2(A[1]*x)
        y_mag = y_complex.abs()
        y = y_mag
    elif task == 'compressed-sensing':
        y = torch.einsum('ab,bcd->acd', x.reshape(x.shape[0], x.shape[2]*x.shape[2]), A)
    elif task == 'denoising':
        y = x
    return y

def loss_data_fit(x, y, sigma, A, task):
    if task == 'denoising':
        loss = 0.5 * torch.sum((x-y)**2/sigma**2, (-1, -2))
    elif task == 'phase-retrieval':
        meas = forward_model(A, x, task)
        loss = 0.5 * torch.sum((meas- y)**2 / (sigma*A[0])**2, (-1, -2))
    else:
        loss = 0.5 * torch.sum((forward_model(A, x, task)- y)**2 / sigma**2, (-1, -2)) 
    return loss

def calc_ELBO(generator, flow, latent_model, target, latent_dim, num_samples, A, task, sigma):
    if latent_model == 'flow':
        params_sample = torch.randn((num_samples, latent_dim)).to(device)
        z_sample, logdet = flow.reverse(params_sample)
        logprob = -logdet
    if latent_model == 'gaussian':
        mu = flow[0]
        L = flow[1]
        spread_cov = (L@(L.t())).to(device) + torch.diag(torch.ones(latent_dim)).to(device)*(1e-3)
        prior = torch.distributions.MultivariateNormal(mu, spread_cov)
        z_sample = prior.sample((num_samples,)).to(device)
        logprob = -prior.log_prob(z_sample)
    img = generator.decode(z_sample)
    loss_data = loss_data_fit(img, target, sigma, A, task)#loss_data_fit(A, img, target, task, sigma)
    loss_prior = 0.5 * torch.sum(z_sample**2, 1)
    data_fit = torch.mean(loss_data)
    z_norm = torch.mean(loss_prior)
    neg_logdet = torch.mean(logprob)
    return data_fit, z_norm, neg_logdet

def train_flow(generator, 
                flow, 
                latent_model, 
                latent_dim, 
                lr, 
                A, 
                task, 
                sigma, 
                num_samples,
                target, 
                epochs):
    params = []
    if latent_model == 'flow':
        params += flow.parameters()
    elif latent_model == 'gaussian':
        mu = Variable(flow[0])
        mu.requires_grad = True
        L = Variable(flow[1])
        L.requires_grad = True
        flow[0] = mu
        flow[1] = L
        params += [mu]
        params += [L]
    optimizer = optim.Adam(params, lr=lr)
    BEST_ELBO = 1000000.
    BEST_FLOW = copy.deepcopy(flow)
    for k in range(epochs):
        data_fit, z_norm, neg_logdet = calc_ELBO(generator, flow, latent_model, target, latent_dim, num_samples, A, task, sigma)
        ELBO = data_fit + z_norm + neg_logdet
        ELBO.backward(retain_graph = True)
        cur_loss = ELBO.item()
        if cur_loss < BEST_ELBO:
            BEST_FLOW = copy.deepcopy(flow)
            BEST_ELBO = cur_loss
        optimizer.step()
        #if k % 1000 == 0:
        #    print("Epoch {}".format(k))
        #    print("ELBO: {}".format(cur_loss))
        #    print("Data fit: {}".format(data_fit.item()))
        #    print("Prior: {}".format(z_norm.item()))
        #    print("Entropy: {}".format(neg_logdet.item()))
        #    print("Best ELBO: {}".format(BEST_ELBO))
    return BEST_FLOW

def create_confusion_matrix(num_classes, full_ELBO_comp_mat, task, folder_path):
    matplotlib.rcParams.update({'font.size': 24})
    ELBO_comp_mat = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        ELBO_comp_mat[i,:] = full_ELBO_comp_mat[i,:] / np.amin(full_ELBO_comp_mat[i,:])
    ELBO_comp_mat = np.round(ELBO_comp_mat, 3)#np.rint(ELBO_comp_mat)#np.rint(ELBO_comp_mat, 3)
    fig, ax = plt.subplots(1,1, figsize=(18,16))
    im = ax.imshow(ELBO_comp_mat, cmap='Blues', interpolation='nearest')
    ax.set_title('$-\mathrm{ELBOProxy}$ confusion rows for each IGM ($\downarrow$)')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis('off')
    jump_x = (10 - 0) / (2.0 * 10)
    jump_y = (10 - 0) / (2.0 * 10)
    x_positions = np.linspace(start=0, stop=10, num=10, endpoint=False)
    y_positions = np.linspace(start=0, stop=10, num=10, endpoint=False)
    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            label = int(full_ELBO_comp_mat[y_index, x_index])
            text_x = x
            text_y = y
            ax.text(text_x, text_y, label, color='black', ha='center', va='center')
    fig.colorbar(im)
    image_format = 'svg' # e.g .png, .svg, etc.
    image_name = 'model-sel-MNIST-r2-' + task + '-conf-mat.svg'
    fig.savefig(os.path.join(folder_path, image_name), format=image_format, dpi=800)

def perform_model_selection(num_classes, 
                            test_imgs, 
                            noisy_imgs,
                            image_size,
                            generators,
                            latent_dim,
                            num_samples,
                            lr,
                            A,
                            task,
                            sigma,
                            epochs,
                            num_runs,
                            latent_model):
    nrow = 10
    ncol = 12

    fig = plt.figure(figsize=(ncol+1, nrow+1)) 

    gs = gridspec.GridSpec(nrow, ncol,
            wspace=0.0, hspace=0.0, 
            top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
            left=0.5/(ncol+1), right=1-0.5/(ncol+1)) 
    plt.show()

    if task == 'denoising' or task == 'phase-retrieval':
        nrow = 10
        ncol = 12

        fig = plt.figure(figsize=(ncol+1, nrow+1)) 
        matplotlib.rcParams.update({'font.size': 20})
    elif task == 'CS':
        nrow = 10
        ncol = 11

        fig = plt.figure(figsize=(ncol+1, nrow+1)) 
        matplotlib.rcParams.update({'font.size': 20})
    n_flow = 16
    affine = True
    seqfrac = 2
    permute = 'random'
    batch_norm = True
    use_dropout = False

    array_of_imgs = np.zeros((num_classes, num_classes+2, image_size, image_size))
    full_ELBO_comp_mat = np.zeros((num_classes, num_classes)) ## (i,j)-th entry = avg ELBO of model trained on class i denoising imgs from class j

    for i in range(num_classes):
        true_imgs = test_imgs[i].to(device)
        noisy_meas = noisy_imgs[i].to(device)
        ax= plt.subplot(gs[i,0])
        ax.imshow(true_imgs.detach().cpu().numpy().reshape([image_size, image_size]), cmap='gray')
        array_of_imgs[i,0,:,:] = true_imgs.detach().cpu().numpy().reshape([1, 1, image_size, image_size])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis("off")
        if i == 0:
            ax.set_title("Truth")
        ax= plt.subplot(gs[i,1])
        if task == 'denoising':
            ax.imshow(noisy_meas.detach().cpu().numpy().reshape([image_size, image_size]), vmin=0, vmax=1, cmap='gray')
            array_of_imgs[i,1,:,:] = noisy_meas.detach().cpu().numpy().reshape([1, 1,image_size, image_size])
        else:
            fftshift = np.fft.fftshift(noisy_meas.detach().cpu().numpy())
            ax.imshow(fftshift.reshape([image_size, image_size]), cmap='gray')
            array_of_imgs[i,1,:,:] = fftshift.reshape([1, 1, image_size, image_size])
            array_of_imgs[i,1,:,:] = noisy_meas.detach().cpu().numpy().reshape([1, 1,image_size, image_size])

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis("off")
        if i == 0:
            ax.set_title("Meas")
        print("Reconstruction on image from class {0}".format(i))
        print("------------------------------------")
        for j in range(num_classes):
            generator = generators[j]
            print("Generator from class {0}".format(j))
            avg_ELBO = 0.
            best_ELBO = 100000000000.
            for kk in range(num_runs):
                print("Run # {0}".format(kk))
                flow = flow_def.get_flow_model(latent_dim, n_flow, affine, seqfrac, permute, batch_norm, use_dropout)
                flow = train_flow(generator, flow, latent_model, latent_dim, lr, A, task, sigma, num_samples, noisy_meas, epochs)
                data_fit, z_norm, neg_logdet = calc_ELBO(generator, 
                                                         flow, 
                                                         latent_model, 
                                                         noisy_meas, 
                                                         latent_dim, 
                                                         num_samples, 
                                                         A, 
                                                         task, 
                                                         sigma)
                ELBO = data_fit + z_norm + neg_logdet
                avg_ELBO += ELBO.item()
                if ELBO.item() < best_ELBO:
                    best_ELBO = ELBO.item()
                    best_flow = copy.deepcopy(flow)
            avg_ELBO = avg_ELBO / num_runs
            full_ELBO_comp_mat[i,j] = avg_ELBO
            G = lambda z: generator.decode(z)
            avg_img = training_utils.get_avg_std_img(best_flow, G, latent_model, latent_dim, 0)[0] 
            ax = plt.subplot(gs[i,j+2])
            ax.imshow(avg_img.detach().cpu().numpy().reshape([image_size, image_size]), cmap='gray')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.axis("off")
            array_of_imgs[i,j+2,:,:] = avg_img.detach().cpu().numpy().reshape([1, 1, image_size, image_size])
            if i == 0:
                ax.set_title(str(j))
            print("Final avg ELBO: {0}".format(avg_ELBO))

        
            folder_path = './results/model_selection/'
            image_format = 'svg' # e.g .png, .svg, etc.
            image_name = 'model-sel-MNIST-r2-' + task + '-samples.' + image_format
            array_name = 'model-sel-MNIST-r2-' + task + '-samples.npy'
            conf_mat_name = 'model-sel-MNIST-r2-' + task + '-conf-mat.npy'
            np.save(os.path.join(folder_path, array_name), array_of_imgs)
            np.save(os.path.join(folder_path, conf_mat_name), full_ELBO_comp_mat)
            fig.savefig(os.path.join(folder_path, image_name), format=image_format, dpi=800)
            create_confusion_matrix(num_classes, full_ELBO_comp_mat, task, folder_path)

        print("------------------------------------")
        print("------------------------------------")
        print("------------------------------------")

    folder_path = './results/model_selection/'
    image_format = 'svg' # e.g .png, .svg, etc.
    image_name = 'model-sel-MNIST-r2-' + task + '-samples.' + image_format
    array_name = 'model-sel-MNIST-r2-' + task + '-samples.npy'
    conf_mat_name = 'model-sel-MNIST-r2-' + task + '-conf-mat.npy'
    np.save(os.path.join(folder_path, array_name), array_of_imgs)
    np.save(os.path.join(folder_path, conf_mat_name), full_ELBO_comp_mat)
    fig.savefig(os.path.join(folder_path, image_name), format=image_format, dpi=800)
    create_confusion_matrix(num_classes, full_ELBO_comp_mat, task, folder_path)
