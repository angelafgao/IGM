import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch.autograd as autograd
import random
import copy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.autograd import Variable

torch.set_default_dtype(torch.float32)
import torch.optim as optim
import pickle
import math
from torch import Tensor

# from sys import exit
import matplotlib.pyplot as plt
from torch.nn import functional as F

from utils.data_utils import get_true_and_noisy_data
from utils.eht_utils import loss_angle_diff

GPU = torch.cuda.is_available()
if GPU == True:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
    print("num GPUs",torch.cuda.device_count())
else:
    dtype = torch.FloatTensor
    
seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Parse command line arguments
parser = argparse.ArgumentParser(description='AmbientGAN baseline')
parser.add_argument('--task', type=str, default='closure-phase',
                    help='inverse problem to solve (default: closure-phase)')
parser.add_argument('--dataset', type=str, default='m87', help='dataset to use (default: m87)')
parser.add_argument('--lr', type=float, default=5e-6, help='learning rate (default: 5e-6)')
parser.add_argument('--epochs', type=int, default=20000, help='number of epochs (default: 20k)')
args = parser.parse_args()
    
# Define constants
n_cpu = 8
latent_dim = 100
sample_interval = 400
img_size = 64
channels = 1
img_shape = (channels, img_size, img_size)
n_critic = 1
n_gen = 5
clip_value = 0.01
lambda_gp = 10
LAMBDA = lambda_gp
device = 'cuda'
use_cuda = True
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

num_imgs_total_bh = 60

# lr = 0.000005#0.00000005#0.0000005
lr = args.lr
out_dir = os.path.join('baseline_results', args.dataset, f'ambientgan-lr{lr:.0e}')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


# Original WGAN-GP (from Pytorch-GAN: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py)
DIM = latent_dim
class ConvGenerator(nn.Module):
    def __init__(self):
        super(ConvGenerator, self).__init__()
        preprocess = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 4 * DIM),
            nn.BatchNorm1d(4 * 4 * 4 * DIM),
            nn.ReLU(True),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * DIM, 2 * DIM, 2, stride=2),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * DIM, DIM, 2, stride=2),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, channels, 2, stride=2)
        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * DIM, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        return output.view(-1, channels, img_size, img_size)


class ConvDiscriminator(nn.Module):
    def __init__(self):
        super(ConvDiscriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(channels, DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
        )
        preprocess = nn.Sequential(
            nn.Linear(2267 * 2, 2 * DIM),
            nn.ReLU(True),
        )
        self.preprocess = preprocess
        self.main = main
        self.linear = nn.Linear(4*4*4*DIM, 1)

    def forward(self, input):
        preinput = self.preprocess(input)
        preinput = preinput.view(-1, 1, 2, DIM)
        output = self.main(preinput)
        output = output.view(-1, 4*4*4*DIM)
        output = self.linear(output)
        return output

# Networks
def conv_block(c_in, c_out, k_size=4, stride=2, pad=1, use_bn=True, transpose=False):
    module = []
    if transpose:
        module.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=not use_bn))
    else:
        module.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=not use_bn))
    if use_bn:
        module.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*module)


class Generator(nn.Module):
    def __init__(self, z_dim=latent_dim, channels=1, conv_dim=64):
        super(Generator, self).__init__()
        self.tconv1 = conv_block(z_dim, conv_dim * 8, pad=0, transpose=True)
        self.tconv2 = conv_block(conv_dim * 8, conv_dim * 4, transpose=True)
        self.tconv3 = conv_block(conv_dim * 4, conv_dim * 2, transpose=True)
        self.tconv4 = conv_block(conv_dim * 2, conv_dim, transpose=True)
        self.tconv5 = conv_block(conv_dim, channels, transpose=True, use_bn=False)

		# Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = x.reshape([x.shape[0], -1, 1, 1])
        x = F.relu(self.tconv1(x))
        x = F.relu(self.tconv2(x))
        x = F.relu(self.tconv3(x))
        x = F.relu(self.tconv4(x)) 
        x = torch.sigmoid(self.tconv5(x))
        return x


class Critic(nn.Module):
    def __init__(self, channels=1, conv_dim=64, task='denoising', preprocess_dim=-1):
        super(Critic, self).__init__()
        self.conv1 = conv_block(channels, conv_dim, use_bn=False)
        self.conv2 = conv_block(conv_dim, conv_dim * 2)
        self.conv3 = conv_block(conv_dim * 2, conv_dim * 4)
        self.conv4 = conv_block(conv_dim * 4, conv_dim * 8)
        self.conv5 = conv_block(conv_dim * 8, 1, k_size=4, stride=1, pad=0, use_bn=False)
        self.task = task
        self.conv_dim = conv_dim 
        if self.task == 'denoising':
            preprocess = nn.Identity()
        elif self.task == 'cs' or self.task == 'cs_real_noise':
            preprocess = nn.Sequential(
            #conv_block(60, conv_dim, use_bn=False),
            nn.Linear(2267 * 2, conv_dim * 4),
            nn.LeakyReLU(0.1),
            nn.Linear(conv_dim * 4, conv_dim * 8),
            nn.LeakyReLU(0.1),
            nn.Linear(conv_dim * 8, conv_dim ** 2)
        )
        elif self.task == 'closure-phase':
            preprocess = nn.Sequential(
                nn.Linear(preprocess_dim, conv_dim * 4),
                nn.LeakyReLU(0.1),
                nn.Linear(conv_dim * 4, conv_dim * 8),
                nn.LeakyReLU(0.1),
                nn.Linear(conv_dim * 8, conv_dim ** 2)
            )
        else:
            raise ValueError
        self.preprocess = preprocess
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        alpha = 0.1
        x = self.preprocess(x)
        #if self.task == 'cs':
        x = x.view(-1, 1, self.conv_dim, self.conv_dim)
        x = F.leaky_relu(self.conv1(x), alpha)
        x = F.leaky_relu(self.conv2(x), alpha)
        x = F.leaky_relu(self.conv3(x), alpha)
        x = F.leaky_relu(self.conv4(x), alpha)
        x = self.conv5(x)
        return x.squeeze()

# from same source stated above
def compute_gradient_penalty(netD, real_data, fake_data, task):
    # print "real_data: ", real_data.size(), fake_data.size()
    BATCH_SIZE = real_data.shape[0]
    alpha = torch.rand(BATCH_SIZE, 1)
    #print(real_data.nelement()/BATCH_SIZE)
    if task == 'denoising':
        alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous().view(BATCH_SIZE, 1, img_size, img_size)
    elif task == 'cs':
        alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous().view(BATCH_SIZE, 2267 * 2)
    alpha = alpha.cuda(device) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(device) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def torch_forward_model(x, A):
    #print(x.shape)
    return torch.einsum('ab,bcd->acd', x.reshape(x.shape[0], x.shape[2]*x.shape[2]), A)


def VLBI_corrupt(imgs, A, sigma):
    corr_imgs = torch.zeros((imgs.shape[0], 2267 * 2))
    #print(imgs.shape)
    for i in range(imgs.shape[0]):
        x = imgs[i,:,:,:]
        Ax = torch_forward_model(x, A)
        y = Ax + sigma*torch.randn_like(Ax)
        #print(torch.sum(y**2))
        corr_imgs[i,:] = y.reshape([1, 2267 * 2]) #/ (274 * 2)
    return corr_imgs.to(device)

def VLBI_corrupt_real_noise(imgs, A, sigma):
    corr_imgs = torch.zeros((imgs.shape[0], 2267 * 2))
    #print(imgs.shape)
    for i in range(imgs.shape[0]):
        x = imgs[i,:,:,:]
        Ax = torch_forward_model(x, A)
        y = Ax + sigma[i]*torch.randn_like(Ax)
        #print(torch.sum(y**2))
        corr_imgs[i,:] = y.reshape([1, 2267 * 2]) #/ (274 * 2)
    return corr_imgs.to(device)


def cphase_corrupt(imgs, A, sigma):
    sigma_amp, sigma_ph = sigma
    n_amp = sigma_amp[0].shape[0]
    n_ph = sigma_ph[0].shape[0]

    corr_imgs = torch.zeros((imgs.shape[0], n_amp + n_ph))
    for i in range(imgs.shape[0]):
        x = imgs[i,:,:,:]
        Ax = A(x)
        y_amp, y_ph = Ax
        y_amp = y_amp + sigma_amp[i] * torch.randn_like(y_amp) * x.abs().sum()
        y_ph = y_ph + sigma_ph[i] * torch.randn_like(y_ph)
        corr_imgs[i, :n_amp] = y_amp
        corr_imgs[i, n_amp:] = y_ph
    return corr_imgs.to(device)


def noise_corrupt(imgs, sigma):
    corr_imgs = imgs.clone().float()
    corr_imgs = corr_imgs.to(device) + sigma*torch.randn_like(corr_imgs).to(device)
    return corr_imgs
    
def get_data(task):
    if task == 'cs':
        mat_path = './utils/dft_m87_eht2025.npy'
        A = Tensor(np.load(mat_path)).type(dtype).to(device)
        sigma = 2
        corr_data = np.load('./utils/amb_gan_measurements/noisy_imgs.npy')
        print(corr_data.shape)
        tot_num_imgs = corr_data.shape[0]
        corr_data = torch.tensor(corr_data).to(device).reshape([tot_num_imgs, 2267* 2])
        batch_size = min(150, tot_num_imgs)
    elif task == 'cs_real_noise':
        mat_path = './utils/dft_m87_eht2025.npy'
        A = Tensor(np.load(mat_path)).type(dtype).to(device)
        print(A.shape)
        corr_data_mat = np.load('./utils/amb_gan_measurements/noisy_imgs_m87_real_noise.npz', allow_pickle=True)
        lst = corr_data_mat.files
        tot_num_imgs = 60#len(lst)
        corr_data = np.zeros([60, 2267, 2])
        ii = 0
        for item in lst:
            #print(item)
            #print(corr_data[item].shape)
            corr_data[ii, :, :] = corr_data_mat[item] 
            ii += 1
        #print(corr_data)
        #print(corr_data.shape)
        corr_data = torch.tensor(corr_data).to(device).reshape([tot_num_imgs, 2267 * 2])
        batch_size = min(150, tot_num_imgs)
        sigma_mat = np.load("./utils/sigma_EHT2025_m87_frame.npz", allow_pickle=True)
        sigma = []
        for i in range(tot_num_imgs):
            sigma.append(torch.tensor(sigma_mat['arr_0'][i][np.newaxis, :, np.newaxis]).to(device))

    elif task == 'closure-phase':
        tot_num_imgs = num_imgs_total_bh
        batch_size = min(150, tot_num_imgs)

        true_imgs, noisy_imgs, A, sigma, kernels = \
            get_true_and_noisy_data(img_size, (None, None), tot_num_imgs, args.dataset, 8,
                                    'closure-phase', 'learning', True, cphase_count='min',
                                    envelope_params=None)

        amp0, ph0 = noisy_imgs[0]
        n_amp = amp0.shape[1]
        n_ph = ph0.shape[1]
        corr_data = np.zeros((tot_num_imgs, n_amp + n_ph))
        corr_data = torch.tensor(corr_data).to(device)
        for i in range(tot_num_imgs):
            amp, ph = A(true_imgs[i])
            corr_data[i, :n_amp] = amp
            corr_data[i, n_amp:] = ph

    elif task == 'denoising':
        A = None
        sigma = 0.1
        corr_data = np.load('./utils/amb_gan_measurements/bond_noisy_imgs_final.npy')
        corr_data = torch.tensor(corr_data).to(device)
        corr_data = corr_data.float().reshape([corr_data.shape[0], channels, img_size, img_size])
        tot_num_imgs = corr_data.shape[0]
        batch_size = min(150, tot_num_imgs)

    else:
        raise ValueError

    return A, sigma, corr_data, tot_num_imgs, batch_size

def solve_denoising_recon_ambGAN(denoise_G, corr_data, sigma, epoch):
    # Solving inverse problem
    num_iter = 25000
    z = Variable(torch.randn(corr_data.shape[0], latent_dim)).to(device)
    z.requires_grad = True
    optimizer = torch.optim.Adam([z], lr=1e-3)
    denoise_G.eval()
    for tt in range(num_iter):
        optimizer.zero_grad()
        estimates = denoise_G(z)
        loss = torch.mean(torch.sum((estimates - corr_data)**2, (1,2,3)) / sigma**2) + 0.5 * torch.mean(torch.sum(z**2, 1))
        loss.backward()
        optimizer.step()
        #if tt % 2500 == 0:
        #    print("Num iters: {0}".format(tt))
        #    print("error: {0}".format(loss.item()))

    plt.figure(figsize = (10,10))
    amb_gen_images = denoise_G(z)
    for i in range(64):
        plt.subplot(8,8,i+1)
        plt.imshow(amb_gen_images[i].cpu().data.numpy()[0], cmap = plt.get_cmap('gray'))
        plt.axis('off')
        plt.subplots_adjust(wspace =0, hspace =0)
    plt.savefig(f'{out_dir}/final_amb_gan_denoise_solution_' + str(epoch) + '_epoch.png')


    array_name = f'{out_dir}/final_ambGAN-denoising-reconstruction_' + str(epoch) + '_epoch.npy'
    np.save(array_name, amb_gen_images.cpu().data.numpy())

def solve_cs_recon_ambGAN(amb_G, A, corr_data, sigma, epoch):
    # Solving inverse problem
    num_iter = 10000
    z = Variable(torch.randn(corr_data.shape[0], latent_dim)).to(device)
    z.requires_grad = True
    optimizer = torch.optim.Adam([z], lr=1e-3)
    corr_data = corr_data.reshape([corr_data.shape[0], 2267, 2])
    amb_G.eval()
    #noise_ratio = torch.tensor(0.).to(device)
    #for i in range(60):
    #    noise_ratio += torch.sum(torch.tensor(sigma[i]).to(device)**2)
    #noise_ratio = noise_ratio / 60
    #data_div_noise = corr_data / sigma**2
    #print(data_div_noise.shape)
    #print(torch.sum(data_div_noise, (-1,-2)).shape)
    for tt in range(num_iter):
        optimizer.zero_grad()
        estimates = amb_G(z)
        yhat = torch_forward_model(estimates, A)
        loss = 0.1 * 0.5 * torch.mean(torch.sum(z**2, 1))
        for ii in range(60):
            loss += 0.5*torch.mean(torch.sum((yhat[ii,:,:] - corr_data[ii,:,:])**2 / sigma[ii] ** 2, (-1,-2))) / 60
        #loss = torch.mean(torch.sum((yhat[ii,:,:] - corr_data[ii,:,:])**2 / sigma ** 2, (1,2))
        #torch.mean(torch.sum((yhat - corr_data)**2, (1,2)) / noise_ratio) #+ 0.5 * torch.mean(torch.sum(z**2, 1))

        loss.backward()
        optimizer.step()
        #if tt % 5000 == 0:
        #    print("Num iters: {0}".format(tt))
        #    print("error: {0}".format(loss.item()))

    plt.figure(figsize = (10,10))
    amb_gen_images = amb_G(z)
    for i in range(60):
        plt.subplot(12,5,i+1)
        plt.imshow(amb_gen_images[i].cpu().data.numpy()[0], cmap = plt.get_cmap('gray'))
        plt.axis('off')
        plt.subplots_adjust(wspace =0, hspace =0)
    plt.savefig(f'{out_dir}/final_long_amb_gan_bh2025_solution_' + str(epoch) + '_epoch_real_noise.png')

    np.save(f"{out_dir}/final_long_bh2025_recon_ambGAN_" + str(epoch) + "_epoch_real_noise.npy", amb_gen_images.cpu().data.numpy())


def solve_cphase_recon_ambGAN(amb_G, A, corr_data, sigma, epoch):
    sigma_amp, sigma_ph = sigma
    n_amp = len(sigma_amp[0])
    n_ph = len(sigma_ph[0])

    # Solving inverse problem
    num_iter = 10000
    z = Variable(torch.randn(corr_data.shape[0], latent_dim)).to(device)
    z.requires_grad = True
    optimizer = torch.optim.Adam([z], lr=1e-3)
    # corr_data = corr_data.reshape([corr_data.shape[0], 2267, 2])
    corr_amp = corr_data[:,:n_amp]
    corr_ph = corr_data[:,n_amp:]
    amb_G.eval()
    for tt in range(num_iter):
        optimizer.zero_grad()
        estimates = amb_G(z)
        # yhat = torch_forward_model(estimates, A)
        y_amp_hat, y_ph_hat = A(estimates)
        loss = 0.1 * 0.5 * torch.mean(torch.sum(z ** 2, 1))

        for ii in range(60):
            # loss += 0.5 * torch.mean(
            #     torch.sum((yhat[ii, :, :] - corr_data[ii, :, :]) ** 2 / sigma[ii] ** 2, (-1, -2))) / 60
            loss += 0.5 * torch.mean((y_amp_hat[ii,:] - corr_amp[ii,:]) ** 2 / sigma_amp[ii] ** 2) / 60
            loss += 10 * loss_angle_diff(corr_ph[None,ii,:], y_ph_hat[None,ii,:], sigma_ph[ii]).squeeze() / 60

        loss.backward()
        optimizer.step()
        # if tt % 5000 == 0:
        #    print("Num iters: {0}".format(tt))
        #    print("error: {0}".format(loss.item()))

    plt.figure(figsize=(10, 10))
    amb_gen_images = amb_G(z)
    for i in range(60):
        plt.subplot(12, 5, i + 1)
        plt.imshow(amb_gen_images[i].cpu().data.numpy()[0], cmap=plt.get_cmap('gray'))
        plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f'{out_dir}/final_long_amb_gan_bh2025_solution_' + str(epoch) + '_epoch_real_noise.png')

    np.save(f"{out_dir}/final_long_bh2025_recon_ambGAN_" + str(epoch) + "_epoch_real_noise.npy",
            amb_gen_images.cpu().data.numpy())


# training function for AmbientGAN
def amb_train(generator, discriminator, n_epochs, optimizer_G, optimizer_D, dataloader, measurement, A, corr_data, task, sigma):
    d_loss_lst = []
    g_loss_lst = []
    
    generator.train()
    discriminator.train()
    for epoch in range(n_epochs):
        for i, imgs in enumerate(dataloader):
            real_imgs = Variable(imgs[0].type(Tensor))
            for d_its in range(n_critic):
                #  Train Discriminator
                # ---------------------
                optimizer_D.zero_grad()
                # Sample noise as generator input
                z = Variable(torch.randn((imgs[0].size(0), latent_dim))).to(device)
                # Generate a batch of images
                fake_imgs = generator(z)
                # Apply lossy measurement on generated images
                if task == 'cs' or task == 'cs_real_noise' or task == 'closure-phase':
                    fake_measurements = measurement(fake_imgs, A, sigma)
                elif task == 'denoising':
                    fake_measurements = measurement(fake_imgs, sigma)
                else:
                    raise ValueError('invalid task')
                # Real images
                real_validity = discriminator(real_imgs)
                # Fake images
                fake_validity = discriminator(fake_measurements)
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_measurements.data, task)
                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

                d_loss.backward()
                optimizer_D.step()
            for g_its in range(n_gen):
                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()
                # Generate a batch of images
                z = Variable(torch.randn((imgs[0].size(0), latent_dim))).to(device)
                fake_imgs = generator(z)
                # Apply lossy measurement on generated images
                if task == 'cs' or task == 'cs_real_noise' or task == 'closure-phase':
                    fake_measurements = measurement(fake_imgs, A, sigma)
                elif task == 'denoising':
                    fake_measurements = measurement(fake_imgs, sigma)                # Loss measures generator's ability to fool the discriminator
                else:
                    raise ValueError
                # Train on fake images
                fake_validity = discriminator(fake_measurements)
                g_loss = -torch.mean(fake_validity)
                g_loss.backward()
                optimizer_G.step()           
        if epoch % 100 == 0:
            print("[Epoch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, n_epochs, d_loss.item(), g_loss.item()))
        d_loss_lst.append(d_loss.item())
        g_loss_lst.append(g_loss.item())
        if epoch > 0 and epoch % 250 == 0:
            # plot discriminator loss vs epochs
            plt.figure(figsize = (10,10))
            plt.grid()
            plt.plot(d_loss_lst, label = 'AmbientGAN')
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.legend()
            plt.show()
            plt.savefig(f'{out_dir}/final_long_amb_gan_bh2025_disc_loss_' + str(epoch) + '_epoch.png')

            # plot generator loss vs epochs
            plt.figure(figsize = (10,10))
            plt.grid()
            plt.plot(g_loss_lst, label = 'AmbientGAN')
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.legend()
            plt.show()
            plt.savefig(f'{out_dir}/final_long_amb_gan_bh2025_gen_loss_' + str(epoch) + '_epoch.png')
            
            
            generator.eval()
            inputs = torch.randn(corr_data.shape[0], latent_dim).cuda()
            amb_gen_images = generator(inputs)
            plt.figure(figsize = (10,10))
            for i in range(60):
                plt.subplot(10,6,i+1)
                plt.imshow(amb_gen_images[i].cpu().data.numpy()[0], cmap = plt.get_cmap('gray'))
                plt.axis('off')
                plt.subplots_adjust(wspace =0, hspace =0)
            plt.savefig(f'{out_dir}/final_long_amb_gan_' + task + '_samples_' + str(epoch) + '_epoch_real_noise.png')
        if epoch > 0 and epoch % 249 == 0:
            print("solving inverse problem.......")
            if task == 'cs' or task == 'cs_real_noise':
                solve_cs_recon_ambGAN(generator, A, corr_data, sigma, epoch)
            elif task == 'denoising':
                solve_denoising_recon_ambGAN(generator, corr_data, sigma, epoch)
            elif task == 'closure-phase':
                solve_cphase_recon_ambGAN(generator, A, corr_data, sigma, epoch)
            else:
                raise ValueError('invalid task')
    return({'d_loss': d_loss_lst, 'g_loss': g_loss_lst})

    
################################ BLACK HOLE RECONSTRUCTION

# task = 'cs_real_noise'
task = args.task
b1 = 0.5
b2 = 0.999

# train AmbientGAN for MNIST with noise corruption
if task == 'cs':
    measurement_operation = VLBI_corrupt
if task == 'cs_real_noise':
    measurement_operation = VLBI_corrupt_real_noise
elif task == 'denoising':
    measurement_operation = noise_corrupt
elif task == 'closure-phase':
    measurement_operation = cphase_corrupt
else:
    raise ValueError

A, sigma, corr_data, tot_num_imgs, batch_size = get_data(task)
preprocess_dim = -1
if task == 'closure-phase':
    sigma_amp, sigma_ph = sigma
    n_amp, n_ph = sigma_amp[0].shape[0], sigma_ph[0].shape[0]
    preprocess_dim = n_amp + n_ph

# Dataloaders for lossy datasets
from torch.utils.data import TensorDataset
corr_dataset = TensorDataset(corr_data)
corr_dataloader = torch.utils.data.DataLoader(corr_dataset, batch_size = 30, shuffle=True)



# Model and optimizer
## IMPORTS
import generative_model
import utils
import generative_model.model_utils as model_utils
import utils.training_utils as training_utils
import utils.data_utils as data_utils

#generator, G = model_utils.get_generator(latent_dim, img_size, 'deepdecoder')

amb_mnist_G = Generator(latent_dim)
amb_mnist_D = Critic(task=task, preprocess_dim=preprocess_dim)
print(amb_mnist_G)
print(amb_mnist_D)
amb_mnist_optim_G = torch.optim.Adam(amb_mnist_G.parameters(), lr=lr, betas=(b1, b2))
amb_mnist_optim_D = torch.optim.Adam(amb_mnist_D.parameters(), lr=lr, betas=(b1, b2))
if cuda:
    amb_mnist_G.cuda()
    amb_mnist_D.cuda()
    
n_epochs = args.epochs
# n_epochs = 1
amb_mnist_loss = amb_train(amb_mnist_G, amb_mnist_D, n_epochs,
                           amb_mnist_optim_G, amb_mnist_optim_D, 
                           corr_dataloader, measurement_operation, A, corr_data, task, sigma)

solve_cphase_recon_ambGAN(amb_mnist_G, A, corr_data, sigma, n_epochs)

# Examples of AmbientGAN trained
amb_mnist_G.eval()
inputs = torch.randn(batch_size, latent_dim).cuda()
amb_gen_images = amb_mnist_G(inputs)
plt.figure(figsize = (10,10))
import matplotlib.pyplot as plt
for i in range(60):
    plt.subplot(12,5,i+1)
    plt.imshow(amb_gen_images[i].cpu().data.numpy()[0], cmap = plt.get_cmap('gray'))
    plt.axis('off')
    plt.subplots_adjust(wspace =0, hspace =0)
plt.savefig(os.path.join(out_dir, 'final_long_amb_gan_bh2025_samples_final_real_noise.png'))
plt.close()
    
# plot discriminator loss vs epochs
plt.figure(figsize = (10,10))
plt.grid()
plt.plot(amb_mnist_loss['d_loss'], label = 'AmbientGAN')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.savefig(os.path.join(out_dir, 'final_long_amb_gan_bh2025_disc_loss.png'))
plt.close()

# plot generator loss vs epochs
plt.figure(figsize = (10,10))
plt.grid()
plt.plot(amb_mnist_loss['g_loss'], label = 'AmbientGAN')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.savefig(os.path.join(out_dir, 'final_long_amb_gan_bh2025_gen_loss.png'))
plt.close()
