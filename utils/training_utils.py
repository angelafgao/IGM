import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as functional
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.autograd import Variable
import os

kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}


torch.set_default_dtype(torch.float32)
import torch.optim as optim
import pickle
import math
from torch import Tensor
from torch.utils.data import WeightedRandomSampler

from .data_utils import get_envelope
from .eht_utils import loss_angle_diff
from .vis_utils import latest_epoch_path

class GMM_Custom(torch.distributions.MultivariateNormal):
    def __init__(self, mu, L, eps, device, latent_dim, **kwargs):
        C = (L@(L.t())).to(device) + torch.diag(torch.ones(latent_dim)).to(device)*(eps)
        self.eps = eps
        self.L = L
        torch.distributions.MultivariateNormal.__init__(self, mu, C, **kwargs)
        
#         self.latent_dim = C.shape[0]
#         self.mu = mu
#         self.C = C
      
    def sample(self, n):
        latent_dim = self.covariance_matrix.shape[0]
#         x = torch.rand(n[0], latent_dim).to(self.loc.device)
#         x = torch.matmul(x, self.L.t())
#         x = x + self.loc.unsqueeze(0)

        x = torch.rand(latent_dim, n[0]).to(self.loc.device)
        x = torch.matmul(self.L, x)
        x = x.t() + self.loc.unsqueeze(0)
        return x


def get_latent_model(latent_dim, num_imgs):
    list_of_models = [[torch.randn((latent_dim,)).to(device),
                   torch.tril(torch.ones((latent_dim, latent_dim))).to(device)] for i in range(num_imgs)]
    return list_of_models

def get_gmm_gen_params(models, generator, num_imgs, model_type, eps_fixed):
    params = []
    for kk in range(num_imgs):
        if model_type != "gmm_low" and model_type != "gmm_low_eye":
            mu, L = models[kk][0], models[kk][1]
            mu = Variable(mu, requires_grad = True)
            L = Variable(L, requires_grad = True)
            models[kk][0] = mu
            models[kk][1] = L
            params += [models[kk][0]]
            params += [models[kk][1]]
        elif model_type == "gmm_low" or model_type == "gmm_low_eye":
            if eps_fixed == True:
                mu, L, eps = models[kk][0], models[kk][1], models[kk][2]
                mu = Variable(mu, requires_grad = True)
                L = Variable(L, requires_grad = True)
                models[kk][0] = mu
                models[kk][1] = L
                models[kk][2] = eps
                params += [models[kk][0]]
                params += [models[kk][1]]
            else:
                mu, L, eps = models[kk][0], models[kk][1], models[kk][2]
                mu = Variable(mu, requires_grad = True)
                L = Variable(L, requires_grad = True)
                eps = Variable(eps, requires_grad = True)
                models[kk][0] = mu
                models[kk][1] = L
                models[kk][2] = eps
                params += [models[kk][0]]
                params += [models[kk][1]]
                params += [models[kk][2]]
    params += generator.parameters()
    return params

def get_avg_std_img(model, generator, latent_model, latent_dim, GMM_EPS, image_size, generator_type):
    nimg = 40
#     if latent_model == 'flow':

    if (latent_model == 'gmm') or (latent_model == "gmm_eye"):
        prior = torch.distributions.MultivariateNormal(model[0],
                        (GMM_EPS)*torch.eye(latent_dim).to(device)+model[1]@(model[1].t()))
        z_sample = prior.sample((nimg,)).to(device)
    elif (latent_model == 'gmm_low') or (latent_model == "gmm_low_eye"):
        prior = torch.distributions.LowRankMultivariateNormal(model[0], model[1], model[2]*model[2]+1e-6)
        z_sample = prior.sample((nimg,)).to(device)
    elif (latent_model == "gmm_custom"):
        prior = GMM_Custom(model[0], model[1], GMM_EPS, device, latent_dim)
                           #(GMM_EPS)*torch.eye(latent_dim).to(device)+model[1]@(model[1].t()))
        z_sample = prior.sample((nimg,)).to(device)
        
    if generator_type == "norm_flow":
        img,_ = generator(z_sample)
        img = img.reshape([nimg, image_size, image_size])
    else:
        img = generator(z_sample)
    avg = torch.mean(img, dim=0)
    std = torch.std(img, dim=0)
    return avg, std

def forward_model(A, x, task, idx=0, dataset=None, use_envelope=False):
    if task == 'phase-retrieval':
        y_complex = torch.fft.fft2(A[1]*x)
        #xhat = torch.zeros((x.shape[0], x.shape[2]*x.shape[2], 2)).to(device)
        #xhat[:,:,0] = x.reshape(x.shape[0], x.shape[2]*x.shape[2])
        #x_comp_vec = torch.view_as_complex(xhat)
        #y_complex = torch.mm(A, x_comp_vec.T)
        y_mag = y_complex.abs()
        y = y_mag
    elif task == 'gauss-phase-retrieval':
        xhat = torch.zeros((x.shape[0], x.shape[2]*x.shape[2], 2)).to(device)
        xhat[:,:,0] = x.reshape(x.shape[0], x.shape[2]*x.shape[2])
        x_comp_vec = torch.view_as_complex(xhat)
        y_complex = torch.mm(A, x_comp_vec.T)
        y_mag = y_complex.abs()
        y = y_mag
    elif task == 'phase-problem':
        y_complex = torch.fft.fft2(A[1]*x)
        y = y_complex.angle()
    elif task == 'compressed-sensing' and dataset=="sagA_video":
        y = torch.einsum('ab,bcd->acd', x.reshape(x.shape[0], x.shape[2]*x.shape[2]), A[idx])
    elif task == 'compressed-sensing' and dataset != "sagA_video":
        y = torch.einsum('ab,bcd->acd', x.reshape(x.shape[0], x.shape[2]*x.shape[2]), A)
    elif task == 'super-res':
        y = A(x)
    elif task == 'inpainting':
        y = A * x
    elif task == 'closure-phase':
        if dataset == 'm87' or dataset == 'sagA':
            y = A(x, use_envelope=use_envelope)
        elif dataset == 'sagA_video':
            y = A(x, idx=idx, use_envelope=use_envelope)
        else:
            raise ValueError('invalid dataset for task closure-phase')
    return y

def loss_data_fit(x, y, sigma, A, task, dataset, idx, gamma=None, cp_scale=1, use_envelope=False):
    mse = torch.nn.MSELoss()
    
    if task == 'denoising':
        loss = 0.5 * torch.sum((x-y)**2/sigma**2, (-1, -2))
    elif task == 'phase-retrieval':
        meas = forward_model(A, x, task)
        loss = 0.5 * torch.sum((meas - y)**2 / (sigma*A[0])**2, (-1, -2))
    elif task == 'gauss-phase-retrieval':
        meas = forward_model(A, x, task)
        loss = 0.5 * torch.sum((meas - y)**2 / (sigma)**2, (-1, -2))
    elif task == 'closure-phase':
        if gamma is None:
            raise ValueError('missing gamma for task closure-phase')

        meas_mag, meas_phase = forward_model(A, x, task, idx=idx, dataset=dataset,
                                             use_envelope=use_envelope)
        y_mag, y_phase = y

        sigma_v, sigma_cp = sigma
        if hasattr(sigma_v, "__len__"):
            loss_mag = 0.5 * torch.sum((meas_mag - y_mag) ** 2 / sigma_v[idx] ** 2, -1)
        else:
            loss_mag = 0.5 * torch.sum((meas_mag - y_mag) ** 2 / sigma_v ** 2, -1)
        if hasattr(sigma_cp, "__len__"):
            loss_phase = loss_angle_diff(y_phase, meas_phase, sigma_cp[idx])
        else:
            loss_phase = loss_angle_diff(y_phase, meas_phase, sigma_cp)

        loss_mag_scaled = gamma * loss_mag
        loss_phase_scaled = cp_scale * meas_mag.shape[-1] / meas_phase.shape[-1] * loss_phase
        loss = loss_mag_scaled + loss_phase_scaled

        return loss, loss_mag_scaled.detach(), loss_phase_scaled.detach()
    else:
        meas = forward_model(A, x, task, idx=idx, dataset=dataset)
        if (dataset == "sagA_video" or dataset == "m87") and hasattr(sigma, "__len__"):
#             print("new loss")
            loss = 0.5 * torch.sum((meas - y)**2 / sigma[idx]**2, (-1, -2)) 
        else:
            loss = 0.5 * torch.sum((meas - y)**2 / sigma**2, (-1, -2)) 
    return loss


def loss_center(device, center=15.5, dim=32):
    # image prior - centering loss
    X = np.concatenate([np.arange(dim).reshape((1, dim))] * dim, 0)
    Y = np.concatenate([np.arange(dim).reshape((dim, 1))] * dim, 1)
    X = torch.Tensor(X).type(torch.float32).to(device = device)
    Y = torch.Tensor(Y).type(torch.float32).to(device = device)
    def func(y_pred):
        y_pred_flux = torch.mean(y_pred, (-1, -2))
        xc_pred_norm = torch.mean(y_pred * X, (-1, -2)) / y_pred_flux
        yc_pred_norm = torch.mean(y_pred * Y, (-1, -2)) / y_pred_flux

        loss = 0.5 * ((xc_pred_norm-center)**2 + (yc_pred_norm-center)**2)
        return loss[:,0]
    return func


# NOTE: the values are hardcoded rn for 12 filters (assuming batch size 12)
def get_loc_shift_mats(s, d, etype='sq', r=3, all_locs=False):
    if not all_locs:
        p = s // 2 - d
        X = [p, p + d // 3, p + 2 * d // 3, s - p]
        Y = [p, s // 2, s - p]
        centers = [(x, y) for x in X for y in Y]
    else:
        centers = [(x, y) for x in range(s) for y in range(s)]
    num_filters = len(centers)
    filters = np.zeros((num_filters, 1, s, s))
    for i in range(num_filters):
        x, y = centers[i]
        filters[i, :, x, y] = 1
    filters = Tensor(filters).to(device)

    envelope = get_envelope(image_size=s, etype=etype, ds1=d, ds2=d+r)
    return filters, envelope


def train_latent_gmm_and_generator(models,
                                   generator,
                                   generator_func,
                                   generator_type,
                                   lr,
                                   sigma,
                                   targets,
                                   true_imgs,
                                   num_samples,
                                   num_imgs_show,
                                   num_imgs,
                                   num_epochs,
                                   As,
                                   task,
                                   save_img,
                                   dropout_val,
                                   layer_size,
                                   num_layer_decoder,
                                   batchGD,
                                   dataset,
                                   class_idx,
                                   seed,
                                   GMM_EPS,
                                   sup_folder,
                                   folder,
                                   latent_model,
                                   image_size,
                                   num_channels,
                                   no_entropy,
                                   eps_fixed,
                                   gamma=None,
                                   cphase_anneal=None,
                                   cphase_scale=1,
                                   envelope_params=None,
                                   centroid_params=None,
                                   locshift_params=False,
                                   from_checkpoint=False,
                                   validate_args=False):
    # Initialize training params (+ load from checkpoint if specified)

    start_epoch = 0
    if from_checkpoint:
        pattern = 'cotrain-generator_*.pt'
        checkpoints_path = f'./{sup_folder}/{folder}/model_checkpoints/'
        last_generator_path = latest_epoch_path(checkpoints_path, pattern=pattern)

        pattern_begin, pattern_end = pattern.split('*')
        start_epoch = int(os.path.split(last_generator_path)[-1][len(pattern_begin):-len(pattern_end)])

        last_generator_sd = torch.load(last_generator_path)
        last_optimizer_sd = torch.load(last_generator_path.replace('cotrain-generator', 'optimizer'))
        generator.load_state_dict(last_generator_sd)

        models = []
        for i in range(num_imgs):
            latent_model_path = os.path.join(checkpoints_path, f'gmm-{i}.npz')
            latent_model_ = np.load(latent_model_path)
            models.append([Tensor(latent_model_[f]).to(device) for f in latent_model_.files])

    params = get_gmm_gen_params(models, generator, num_imgs, latent_model, eps_fixed)
    optimizer = optim.Adam(params, lr=lr)
    if from_checkpoint:
        optimizer.load_state_dict(last_optimizer_sd)

    loss_list = []
    loss_data_list = []
    loss_prior_list = []
    loss_ent_list = []
    loss_mag_list = []
    loss_phase_list = []
    loss_centroid_list = []
#     loss_div_list = []
    latent_dim = models[0][0].shape[0]
    if from_checkpoint:
        for (loss_checkpt_file, loss_checkpt_list) in [
            ('loss.npy', loss_list),
            ('loss_data.npy', loss_data_list),
            ('loss_prior.npy', loss_prior_list),
            ('loss_ent.npy', loss_ent_list),
            ('loss_mag.npy', loss_mag_list),
            ('loss_phase.npy', loss_phase_list),
            ('loss_centroid.npy', loss_centroid_list)
        ]:
            loss_checkpt_path = f'{sup_folder}/{folder}/{loss_checkpt_file}'
            if os.path.exists(loss_checkpt_path):
                loss_checkpt_list[:] = np.load(loss_checkpt_path).tolist()

    # Set up optional params for closure phase problem

    # cphase anneal
    phase_anneal = np.ones(num_epochs)
    if cphase_anneal is not None and cphase_anneal> 0:
        n_anneal = 10000
        phase_anneal[cphase_anneal: cphase_anneal + n_anneal] = np.linspace(1, 0, num=n_anneal)
        phase_anneal[cphase_anneal + n_anneal:] = 0

    # envelope
    use_envelope = True if envelope_params is not None else False

    # centroid
    loss_centroid_fit = loss_center(device, center=image_size/2-0.5, dim=image_size)
    if centroid_params:
        centroid_loss_wt, anneal_epoch = centroid_params
        centroid_anneal = np.ones(num_epochs)
        if anneal_epoch is not None:
            n_anneal = 10000
            e1, e2 = anneal_epoch, anneal_epoch + n_anneal
            centroid_anneal[e1:e2] = np.linspace(1, 0, num=e2-e1)
            centroid_anneal[e2:] = 0

    # location shift w/ convolutions
    if locshift_params:
        etype, ds1, ds2, learn_locshift = locshift_params
        loc_shift = get_loc_shift_mats(image_size, d=ds1, etype=etype, r=ds2 - ds1,
                                       all_locs=learn_locshift)
    else:
        loc_shift = None
        learn_locshift = False

    if learn_locshift:
        prob_locations = (1 / image_size) ** 2 * torch.ones(image_size * image_size)
        prob_locations = Variable(prob_locations, requires_grad=True)
        params.append(prob_locations)
    else:
        prob_locations = None

    # Training loop
    for k in range(start_epoch, num_epochs):
        loss_sum = 0
        loss_data_sum = 0
        loss_prior_sum = 0
        loss_ent_sum = 0
        loss_mag_sum = 0
        loss_phase_sum = 0
        loss_centroid_sum = 0
        optimizer.zero_grad()
        if 'multi' in task:
            for i in range(num_imgs//2):
                if batchGD == False:
                    optimizer.zero_grad()
                target = targets[i]
                
                if (latent_model == 'gmm') or (latent_model == "gmm_eye"):
                    mu, L = models[i]
                    spread_cov = (L@(L.t())).to(device) + torch.diag(torch.ones(latent_dim)).to(device)*(GMM_EPS)
                    prior = torch.distributions.MultivariateNormal(mu, spread_cov)
                elif (latent_model == 'gmm_low') or (latent_model == "gmm_low_eye"):
                    mu, L, eps = models[i]
                    prior = torch.distributions.LowRankMultivariateNormal(mu, L, eps*eps+1e-6)
                elif (latent_model == "gmm_custom"):
                    mu, L = models[i]
#                     spread_cov = (L@(L.t())).to(device) + torch.diag(torch.ones(latent_dim)).to(device)*(GMM_EPS)
                    prior = GMM_Custom(mu, L, GMM_EPS, device, latent_dim)
                    
                z_sample = prior.sample((num_samples,)).to(device)
            
                if generator_type == "norm_flow":
                    img, logdet = generator_func(z_sample)
                    img = img.reshape([num_samples, image_size, image_size])
                else:
                    img = generator_func(z_sample)
                    
                if no_entropy == True:
                    log_ent = 0
                else:
                    if generator_type == "norm_flow":
                        log_ent = prior.log_prob(z_sample) #+ logdet
                    else:
                        log_ent = prior.log_prob(z_sample)
                
                loss_prior = 0.5 * torch.sum(z_sample**2, 1)
                if 'multi-denoising' in task:
                    curr_task = 'denoising'
                if 'multi-compressed-sensing' in task:
                    curr_task = 'compressed-sensing'
                loss_data = loss_data_fit(img, target, sigma[0], As[0], curr_task, idx=i, dataset=dataset,
                                          gamma=gamma)

                loss_denoise = torch.mean(loss_data + log_ent + loss_prior)

                loss_sum = loss_sum + loss_denoise
                loss_data_sum = loss_data_sum + torch.mean(loss_data)
                loss_prior_sum = loss_prior_sum + torch.mean(loss_prior)
                if no_entropy == False:
                    loss_ent_sum = loss_ent_sum + torch.mean(log_ent)

                if batchGD == False:
                    loss_denoise.backward(retain_graph = True)
                    optimizer.step()
            for j in range(num_imgs//2+1, num_imgs):
                if batchGD == False:
                    optimizer.zero_grad()
                target = targets[j]
                
                if (latent_model == 'gmm') or (latent_model == "gmm_eye"):
                    mu, L = models[j]
                    spread_cov = (L@(L.t())).to(device) + torch.diag(torch.ones(latent_dim)).to(device)*(GMM_EPS)
                    prior = torch.distributions.MultivariateNormal(mu, spread_cov)
                elif (latent_model == 'gmm_low') or (latent_model == "gmm_low_eye"):
                    mu, L, eps = models[j]
                    prior = torch.distributions.LowRankMultivariateNormal(mu, L, eps*eps+1e-6)
                elif (latent_model == "gmm_custom"):
                    mu, L = models[j]
                    spread_cov = (L@(L.t())).to(device) + torch.diag(torch.ones(latent_dim)).to(device)*(GMM_EPS)
                    prior = GMM_Custom(mu, L, GMM_EPS, device, latent_dim)
                z_sample = prior.sample((num_samples,)).to(device)
                
                if generator_type == "norm_flow":
                    img, logdet = generator_func(z_sample)
                    img = img.reshape([num_samples, image_size, image_size])
                else:
                    img = generator_func(z_sample)
                    
                if no_entropy == True:
                    log_ent = 0
                else:
                    if generator_type == "norm_flow":
                        log_ent = prior.log_prob(z_sample) #+ logdet
                    else:
                        log_ent = prior.log_prob(z_sample)
                        
                loss_prior = 0.5 * torch.sum(z_sample**2, 1)
                if 'denoising-compressed-sensing'in task:
                    curr_task = 'compressed-sensing'
                elif 'compressed-sensing-phase-retrieval' or 'denoising-phase-retrieval' in task:
                    if 'gauss' in task:
                        curr_task = 'gauss-phase-retrieval'
                    else:
                        curr_task = 'phase-retrieval'
                elif 'phase-problem' in task:
                    curr_task = 'phase-problem'
                loss_data = loss_data_fit(img, target, sigma[1], As[1], curr_task, idx=j, dataset=dataset,
                                          gamma=gamma)

                loss_new_task = torch.mean(loss_data + log_ent + loss_prior)

                loss_sum = loss_sum + loss_new_task
                loss_data_sum = loss_data_sum + torch.mean(loss_data)
                loss_prior_sum = loss_prior_sum + torch.mean(loss_prior)
                if no_entropy == False:
                    loss_ent_sum = loss_ent_sum + torch.mean(log_ent)

                if batchGD == False:
                    loss_new_task.backward(retain_graph = True)
                    optimizer.step()
            if batchGD == True:
                loss_sum.backward(retain_graph = True)
                optimizer.step()
        else:    
            for i in range(num_imgs):
                if batchGD == False:
                    optimizer.zero_grad()
                target = targets[i]
                
#                 mu, L = models[i]
#                 spread_cov = (L@(L.t())).to(device) + torch.diag(torch.ones(latent_dim)).to(device)*(GMM_EPS)
#                 prior = torch.distributions.MultivariateNormal(mu, spread_cov)
                
                if (latent_model == 'gmm') or (latent_model == "gmm_eye"):
                    mu, L = models[i]
                    spread_cov = (L@(L.t())).to(device) + torch.diag(torch.ones(latent_dim)).to(device)*(GMM_EPS)
                    prior = torch.distributions.MultivariateNormal(mu, spread_cov, validate_args=validate_args)
                elif (latent_model == 'gmm_low') or (latent_model == "gmm_low_eye"):
                    mu, L, eps = models[i]
#                     print(eps*eps)
                    prior = torch.distributions.LowRankMultivariateNormal(mu, L, eps*eps+1e-6,
                                                                          validate_args=validate_args)
                elif (latent_model == "gmm_custom"):
                    mu, L = models[i]
                    spread_cov = (L@(L.t())).to(device) + torch.diag(torch.ones(latent_dim)).to(device)*(GMM_EPS)
                    prior = GMM_Custom(mu, L, GMM_EPS, device, latent_dim, validate_args=validate_args)
                
                z_sample = prior.sample((num_samples,)).to(device)
                
                if generator_type == "norm_flow":
                    img, logdet = generator_func(z_sample)
                    img = img.reshape([num_samples, image_size, image_size])
                else:
                    img = generator_func(z_sample)
#                 print(img.shape, z_sample.shape, img.shape)
                    
                if no_entropy == True:
                    log_ent = 0
                else:
                    if generator_type == "norm_flow":
                        log_ent = prior.log_prob(z_sample) #+ logdet
                    else:
                        log_ent = prior.log_prob(z_sample)

                loss_prior = 0.5 * torch.sum(z_sample**2, 1)

                if loc_shift is not None:
                    filters, envelope = loc_shift
                    if learn_locshift:
                        samples = WeightedRandomSampler(prob_locations, img.shape[0], replacement=True)
                        samples = list(samples)
                        filters = filters[samples]

                    img = envelope * img
                    img = functional.conv2d(filters.transpose(0, 1), img.flip((3)).flip((2)),
                                            padding='same', groups=filters.shape[0]).transpose(0, 1)

                loss_data = loss_data_fit(img, target, sigma, As, task, idx=i, dataset=dataset,
                                          gamma=gamma, cp_scale=cphase_scale * phase_anneal[k],
                                          use_envelope=use_envelope)
                if 'closure-phase' in task:
                    loss_data, loss_mag, loss_phase = loss_data

                if centroid_params:
                    loss_centroid = centroid_anneal[k] * centroid_loss_wt * loss_centroid_fit(img)
                    loss = torch.mean(loss_data + log_ent + loss_prior + loss_centroid)
                else:
                    loss = torch.mean(loss_data + log_ent + loss_prior)

                loss_sum = loss_sum + loss
                loss_data_sum = loss_data_sum + torch.mean(loss_data)
                loss_prior_sum = loss_prior_sum + torch.mean(loss_prior)
                if 'closure-phase' in task:
                    loss_mag_sum += torch.mean(loss_mag)
                    loss_phase_sum += torch.mean(loss_phase)
                if centroid_params:
                    loss_centroid_sum += torch.mean(loss_centroid)
                if no_entropy == False:
                    loss_ent_sum = loss_ent_sum + torch.mean(log_ent)

                if batchGD == False:    
                    loss.backward(retain_graph = True)
                    optimizer.step()
            if batchGD == True:
                loss_sum.backward(retain_graph = True)
                optimizer.step()   
            
            #cur_loss = loss.item()
        
        loss_list.append(loss_sum.item())
        loss_data_list.append(loss_data_sum.item())
        loss_prior_list.append(loss_prior_sum.item())
        if 'closure-phase' in task:
            loss_mag_list.append(loss_mag_sum.item())
            loss_phase_list.append(loss_phase_sum.item())
        if centroid_params:
            loss_centroid_list.append(loss_centroid_sum.item())
        if no_entropy == False:
            loss_ent_list.append(loss_ent_sum.item())
        else:
            loss_ent_list.append(loss_ent_sum)
        if k % 50 == 0:
            print("-----------------------------")
            print("Epoch {}".format(k))
            #print("Curr ELBO: {}".format(cur_loss))
            print("Loss all: {:e}".format(loss_sum.item()/num_imgs))
            print("Loss data fit: {:e}".format(loss_data_sum.item()/num_imgs))
            if 'closure-phase' in task:
                print(f"Scaled loss magnitude: {loss_mag_sum.item()/num_imgs:e} / "
                      f"phase: {loss_phase_sum.item()/num_imgs:e}")
            if centroid_params:
                print(f"Loss centroid: {loss_centroid_sum.item()/num_imgs:e}")
            print("Loss prior: {}".format(loss_prior_sum.item()/num_imgs))
            if no_entropy == False:
                print("Loss entropy: {}".format(loss_ent_sum.item()/num_imgs))
            else:
                print("Loss entropy: {}".format(loss_ent_sum/num_imgs))
            print("-----------------------------")
            if ((k<500) and (k % 100 == 0)) or ((k>=500) and (k % 1000 == 0)):
                if 'multi' in task:
                    img_indices = [i for i in range(num_imgs_show // 2)]
                    img_indices += [num_imgs // 2 + i for i in range(num_imgs_show)]
                else:
                    img_indices = range(num_imgs_show)
                avg_img_list = [get_avg_std_img(models[i], generator_func, latent_model, 
                                                latent_dim, GMM_EPS, image_size, generator_type)[0] for i in img_indices]
                std_img_list = [get_avg_std_img(models[i], generator_func, 
                                                latent_model, latent_dim, GMM_EPS, 
                                                image_size, generator_type)[1] for i in img_indices]
                
                plt.figure()
                plt.plot(loss_list)
                plt.savefig(f'./{sup_folder}/{folder}/loss.png')
                plt.close()
                
                plt.figure()
                plt.plot(loss_list, label="all")
                plt.plot(loss_data_list, label="data")
                plt.plot(loss_prior_list, label="prior")
                plt.plot(loss_ent_list, label="ent")
                if 'closure-phase' in task:
                    plt.plot(loss_mag_list, label='mag')
                    plt.plot(loss_phase_list, label='phase')
                if centroid_params:
                    plt.plot(loss_centroid_list, label='centroid')
                plt.legend()
                plt.savefig(f'./{sup_folder}/{folder}/loss_all.png')
                plt.yscale('log')
                plt.savefig(f'./{sup_folder}/{folder}/loss_all_log.png')
                plt.close()
                
                np.save(f'./{sup_folder}/{folder}/loss.npy', loss_list)
                np.save(f'./{sup_folder}/{folder}/loss_data.npy', loss_data_list)
                np.save(f'./{sup_folder}/{folder}/loss_prior.npy', loss_prior_list)
                np.save(f'./{sup_folder}/{folder}/loss_ent.npy', loss_ent_list)
                if 'closure-phase' in task:
                    np.save(f'./{sup_folder}/{folder}/loss_mag.npy', loss_mag_list)
                    np.save(f'./{sup_folder}/{folder}/loss_phase.npy', loss_phase_list)
                if centroid_params:
                    np.save(f'./{sup_folder}/{folder}/loss_centroid.npy', loss_centroid_list)

                plot_results(models, generator_func, true_imgs, targets, num_channels, 
                             image_size, num_imgs_show, num_imgs,
                             sigma, avg_img_list, std_img_list, save_img, str(k), 
                             dropout_val, layer_size, num_layer_decoder, 
                             batchGD, dataset, folder, sup_folder, GMM_EPS, 
                             task, latent_model, generator_type, envelope_params=envelope_params,
                             loc_shift=loc_shift, prob_locations=prob_locations)

                checkpoint_results(latent_dim, generator_func, models, str(k), num_imgs, 
                                   GMM_EPS, folder, sup_folder, latent_model, image_size, generator_type)   

                save_model_gen_params(generator, models, optimizer, str(k), num_imgs, folder,
                                      sup_folder, latent_model)
                
    return models, generator

def plot_results(models, generator, true_imgs, noisy_imgs, num_channels, 
                 image_size, num_imgs_show, num_imgs, 
                 sigma, avg_img_list, std_img_list, save_img, epoch, 
                 dropout_val, layer_size, num_layer_decoder, 
                 batchGD, dataset, folder, sup_folder, GMM_EPS, 
                 task, latent_model, generator_type, envelope_params=None,
                 loc_shift=None, prob_locations=None):

    num_samples = 7
#     num_channels = true_imgs[0].shape[1]
    latent_dim = models[0][0].shape[0]
#     image_size = true_imgs[0].shape[2]

    if envelope_params is not None:
        etype, ds1, ds2 = envelope_params
        envelope = get_envelope(image_size, etype=etype, ds1=ds1, ds2=ds2)
        envelope = envelope.detach().cpu().numpy().reshape((image_size, image_size))
    elif loc_shift is not None:
        _, envelope = loc_shift
        envelope = envelope.detach().cpu().numpy().reshape((image_size, image_size))
    else:
        envelope = np.ones((image_size, image_size))

    if 'multi' in task:
        img_indices = [i for i in range(num_imgs_show // 2)]
        img_indices += [num_imgs // 2 + i for i in range(num_imgs_show)]
    else:
        img_indices = range(num_imgs_show)
    fig,ax = plt.subplots(len(img_indices), num_samples + 7, figsize = (22,12))
    if not hasattr(sigma, '__len__'):
        fig.suptitle(task + ' with ' + str(num_imgs)+' images ' + str(sigma) + ' noise std')
    else:
        fig.suptitle(task + ' with ' + str(num_imgs) + ' images realistic noise std')
    kk = 0
    for ii in img_indices:
        true = true_imgs[ii].detach().cpu().numpy().reshape([num_channels, image_size, image_size])
        true = true[0,:,:].reshape([image_size, image_size])
        if 'denoising' in task and ii < num_imgs_show:
            noisy = noisy_imgs[ii].detach().cpu().numpy().reshape([num_channels, image_size, image_size])
            noisy = noisy[0,:,:].reshape([image_size, image_size])
        else:
            noisy = np.zeros([image_size, image_size])
        ax[kk][0].imshow(true, cmap = "gray", vmin=0, vmax=1)
        ax[kk][0].axis("off") 
        ax[kk][0].set_title("true (x)")
        ax[kk][1].imshow(noisy, cmap = "gray", vmin=0, vmax=1)
        ax[kk][1].axis("off") 
        ax[kk][1].set_title("noisy (y)")
        
        
#         mu, L = models[ii]
#         spread_cov = L@(L.t()) + torch.diag(torch.ones(latent_dim)).to(device)*(GMM_EPS)
#         prior = torch.distributions.MultivariateNormal(mu, spread_cov)  
        
        if (latent_model == 'gmm') or (latent_model == "gmm_eye"):
            mu, L = models[ii]
            spread_cov = (L@(L.t())).to(device) + torch.diag(torch.ones(latent_dim)).to(device)*(GMM_EPS)
            prior = torch.distributions.MultivariateNormal(mu, spread_cov)
        elif (latent_model == 'gmm_low') or (latent_model == "gmm_low_eye"):
            mu, L, eps = models[ii]
            prior = torch.distributions.LowRankMultivariateNormal(mu, L, eps*eps+1e-6)
        elif (latent_model == "gmm_custom"):
            mu, L = models[ii]
            spread_cov = (L@(L.t())).to(device) + torch.diag(torch.ones(latent_dim)).to(device)*(GMM_EPS)
            prior = GMM_Custom(mu, L, GMM_EPS, device, latent_dim)


        z_sample = prior.sample((num_samples,)).to(device)
        if generator_type == "norm_flow":
            samples_to_show,_ = generator(z_sample)
            samples_to_show = samples_to_show.reshape([num_samples, 1, image_size, image_size])
        else:
            samples_to_show = generator(z_sample)

        mean_i = avg_img_list[kk].detach().cpu().numpy().reshape([num_channels, image_size, image_size])   
        mean_i = mean_i[0,:,:].reshape([image_size, image_size])
        mean_i = envelope * mean_i
        ax[kk][2].imshow(mean_i, cmap = "gray", vmin=0, vmax=1)
        ax[kk][2].axis("off") 
        ax[kk][2].set_title("mean (mu)")
        std_i = std_img_list[kk].detach().cpu().numpy().reshape([num_channels, image_size, image_size]) 
        std_i = std_i[0,:,:].reshape([image_size, image_size])
        std_i = envelope * std_i
        ax[kk][3].imshow(std_i, cmap = "gray")
        ax[kk][3].axis("off") 
        ax[kk][3].set_title("std")
        norm_err = (np.abs(mean_i-true)/np.linalg.norm(std_i))
        ax[kk][4].imshow(norm_err, cmap = "hot", vmin=0, vmax = 5)
        ax[kk][4].axis("off") 
        ax[kk][4].set_title(f"|mu-x|/std \n {np.mean(norm_err):.3f}")
        if 'multi' in task:
            sig_val = sigma[0]
        elif (dataset in ("m87", "sagA_video", "sagA")) and hasattr(sigma, "__len__"):
            sig_val = 2#sigma[ii].detach().cpu().numpy()
        else:
            sig_val = sigma
        norm_err = (np.abs(mean_i-true)/sig_val)
        ax[kk][5].imshow(norm_err, cmap = "hot", vmin=0, vmax = 5)
        ax[kk][5].axis("off") 
        ax[kk][5].set_title(f"|mu-x|/sigma \n {np.mean(norm_err):.3f} ")
        
        norm_err = (np.abs(mean_i-noisy)/sig_val)
        ax[kk][6].imshow(norm_err, cmap = "hot", vmin=0, vmax = 5)
        ax[kk][6].axis("off") 
        ax[kk][6].set_title(f"|mu-y|/sigma \n {np.mean(norm_err):.3f}")
        
        for jj in range(num_samples):
            sample = samples_to_show[jj,:,:,:].detach().cpu().numpy().reshape([num_channels,image_size, image_size])
            sample = sample[0,:,:].reshape([image_size, image_size])
            sample = envelope * sample
            ax[kk][jj+7].imshow(sample, cmap='gray', vmin=0, vmax=1)
            ax[kk][jj+7].axis("off")  
            std = np.around(np.sqrt(np.mean((mean_i - sample)**2)), 3)
            ax[kk][jj+7].set_title(str(std))
        kk += 1

    if save_img == True:
        folder_path = sup_folder + '/' + folder
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.savefig(f'./{sup_folder}/{folder}/{str(epoch).zfill(7)}epochs.png')
        plt.close()

        if 'multi' in task:
            for i in range(len(noisy_imgs)//2):
                np.save(f"./{sup_folder}/{folder}/noisy_imgs_1.npy", 
                        [noisy_imgs[i].detach().cpu().numpy() for i in range(len(noisy_imgs)//2)])
            for i in range(len(noisy_imgs)//2,):
                np.save(f"./{sup_folder}/{folder}/noisy_imgs_2.npy", 
                        [noisy_imgs[i].detach().cpu().numpy() for i in range(len(noisy_imgs)//2,)])
        else:
            if dataset not in ("sagA_video", "m87", "sagA"):
                np.save(f"./{sup_folder}/{folder}/noisy_imgs.npy", 
                        [noisy_imgs[i].detach().cpu().numpy() for i in range(len(noisy_imgs))])
            elif task == 'closure-phase':
                noisy_imgs_ = [(x[0].detach().cpu().numpy(), x[1].detach().cpu().numpy())
                               for x in noisy_imgs]
                noisy_imgs_ = [item for sublist in noisy_imgs_ for item in sublist]
                np.savez(f"./{sup_folder}/{folder}/noisy_imgs.npz",
                         *[noisy_imgs_[i] for i in range(len(noisy_imgs_))])
            else:
                np.savez(f"./{sup_folder}/{folder}/noisy_imgs.npz", 
                        *[noisy_imgs[i].detach().cpu().numpy() for i in range(len(noisy_imgs))])
        np.save(f"./{sup_folder}/{folder}/true_imgs.npy", [true_imgs[i].detach().cpu().numpy() for i in range(len(true_imgs))])

        if prob_locations is not None:
            prob_locs = prob_locations.detach().cpu().numpy().reshape((image_size, image_size))
            plt.imshow(prob_locs, cmap='gray')
            plt.savefig(f'./{sup_folder}/{folder}/{str(epoch).zfill(7)}prob_locs.png')
            plt.close()
            np.save(f'./{sup_folder}/{folder}/prob_locs.npy', prob_locs)

def checkpoint_results(latent_dim, generator, models, epoch, num_imgs, 
                       GMM_EPS, folder, sup_folder, latent_model, image_size, generator_type):
    
    if not os.path.exists(f"./{sup_folder}/{folder}/files"):
        os.makedirs(f"./{sup_folder}/{folder}/files")
        
    if not os.path.exists(f"./{sup_folder}/{folder}/files/{epoch.zfill(7)}"):
        os.makedirs(f"./{sup_folder}/{folder}/files/{epoch.zfill(7)}")

    ''' generate samples from generator'''
    s = 40
    temp = 1
    rand_samples = torch.randn([s, latent_dim]).to(device)*temp
    if generator_type == "norm_flow":
        samples,_ = generator(rand_samples)
        samples = samples.reshape([s, image_size, image_size])
    else:
        samples = generator(rand_samples)
    
    np.save(f"./{sup_folder}/{folder}/files/{epoch.zfill(7)}/generator_samples.npy", samples.detach().cpu().numpy())
     
    s = 40
    
    ''' plot each models samples'''
    for ii in range(num_imgs):
#         mu, L = models[ii]
#         spread_cov = L@(L.t()) + torch.diag(torch.ones(latent_dim)).to(device)*(GMM_EPS)
#         prior = torch.distributions.MultivariateNormal(mu, spread_cov)#model[1]@(model[1].t()))    
        
        if (latent_model == 'gmm') or (latent_model == "gmm_eye"):
            mu, L = models[ii]
            spread_cov = (L@(L.t())).to(device) + torch.diag(torch.ones(latent_dim)).to(device)*(GMM_EPS)
            prior = torch.distributions.MultivariateNormal(mu, spread_cov)
        elif (latent_model == 'gmm_low') or (latent_model == "gmm_low_eye"):
            mu, L, eps = models[ii]
            prior = torch.distributions.LowRankMultivariateNormal(mu, L, eps*eps+1e-6)
        elif (latent_model == "gmm_custom"):
            mu, L = models[ii]
            spread_cov = (L@(L.t())).to(device) + torch.diag(torch.ones(latent_dim)).to(device)*(GMM_EPS)
            prior = GMM_Custom(mu, L, GMM_EPS, device, latent_dim)
        
        z_sample = prior.sample((s,)).to(device)
        if generator_type == "norm_flow":
            samples,_ = generator(z_sample)
            samples = samples.reshape([s, image_size, image_size])
        else:
            samples = generator(z_sample)
        
        np.save(f"./{sup_folder}/{folder}/files/{epoch.zfill(7)}/xsamples_{str(ii).zfill(3)}.npy", samples.detach().cpu().numpy())
        np.save(f"./{sup_folder}/{folder}/files/{epoch.zfill(7)}/zsamples_{str(ii).zfill(3)}.npy", z_sample.detach().cpu().numpy())
    
def save_model_gen_params(generator, models, optimizer, epoch, num_imgs, folder, sup_folder, model_type):
    torch.save(generator.state_dict(), f"./{sup_folder}/{folder}/model_checkpoints/cotrain-generator_{epoch.zfill(7)}.pt")
    torch.save(optimizer.state_dict(), f"./{sup_folder}/{folder}/model_checkpoints/optimizer_{epoch.zfill(7)}.pt")
    for mm in range(num_imgs):
        if model_type == "gmm" or model_type == "gmm_eye":
            mu, L = models[mm]
            np.savez(f'./{sup_folder}/{folder}/model_checkpoints/gmm-{mm}', 
                          mu=mu.detach().cpu().numpy(), L=L.detach().cpu().numpy())
        elif model_type == "gmm_low" or (model_type == "gmm_low_eye"):
            mu, L, ep = models[mm]
            np.savez(f'./{sup_folder}/{folder}/model_checkpoints/gmm-{mm}', 
                          mu=mu.detach().cpu().numpy(), L=L.detach().cpu().numpy(), ep=ep.detach().cpu().numpy())
        elif model_type == "gmm_custom":
            mu, L = models[mm]
            np.savez(f'./{sup_folder}/{folder}/model_checkpoints/gmm-{mm}', 
                          mu=mu.detach().cpu().numpy(), L=L.detach().cpu().numpy())

    
        
