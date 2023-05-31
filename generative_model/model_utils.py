import torch
import numpy as np
from . import stylegan_networks 
from . import deep_decoder
from . import realnvpfc_model_batch    
from . import vae

import math

GPU = torch.cuda.is_available()
if GPU == True:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
    print("num GPUs",torch.cuda.device_count())
else:
    dtype = torch.FloatTensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_params(net):
    s = sum([np.prod(list(p.size())) for p in net.parameters()]);
    return s

def get_stylegan(latent_dim, depth):
    generator = stylegan_networks.StyledGenerator(latent_dim, depth, 5).to(device)
    return generator

def get_deep_decoder(image_size, latent_dim, dropout_val, layer_size, num_layer_decoder=6):
    in_size = (4,4)
    out_size = (image_size, image_size)
    output_depth = 1
    num_channels = [layer_size]*num_layer_decoder
    num_layers = 5
    generator = deep_decoder.deepdecoder(latent_dim,
                            in_size,
                            out_size,
                            output_depth,
                            num_channels=num_channels,
                            need_sigmoid=True,
                            last_noup=False,
                            dropout_val = dropout_val).type(dtype).to(device)
    return generator

def get_generator(latent_dim, image_size, generator_type):
    if generator_type == 'stylegan':
        style_depth = 3
        generator = get_stylegan(latent_dim, style_depth)
        step = int(math.log(image_size, 2) - 2)
        G = lambda z: generator(z, step=step)
        return generator, G
    elif generator_type == 'deepdecoder':
        dropout_val = 1e-4
        layer_size = 150
        num_layer_decoder = 6
        generator = get_deep_decoder(image_size, latent_dim, dropout_val, layer_size, num_layer_decoder)
        G = lambda z: generator(z)
        return generator, G
    elif generator_type == 'vae':   
        generator = vae.ConvVAE(latent_dim).to(device)
        G = lambda z: generator.decode(z)
        return generator, G
    elif generator_type == "norm_flow":
        n_flow = 16
        affine = True
        seqfrac = 8
        permute = 'random'
        batch_norm = True
        use_dropout = False
        generator = get_flow_model(latent_dim, n_flow, affine, seqfrac, permute, batch_norm, use_dropout)
        G = lambda z: flow_results_with_sigmoid(generator, z)
        return generator, G
    
    
def flow_results_with_sigmoid(generator, z):
    x, logdet = generator.reverse(z)
    x_out = torch.sigmoid(x)
    
    det_sigmoid = torch.sum(-x -  2*torch.nn.Softplus()(-x), -1)
    logdet_out = logdet + det_sigmoid
    return x_out, logdet_out

def get_flow_model(latent_dim, n_flow, affine, seqfrac, permute, batch_norm, use_dropout):
    model = realnvpfc_model_batch.RealNVP(latent_dim, n_flow, affine=affine, seqfrac=seqfrac, permute=permute, batch_norm=batch_norm, use_dropout=use_dropout).to(device)
    return model

def get_latent_model(latent_dim, num_imgs, model_type):
    if model_type == 'gmm' or model_type == 'gmm_custom':
        list_of_models = [[torch.randn((latent_dim,)).to(device),
                   torch.tril(torch.ones((latent_dim, latent_dim))).to(device)] for i in range(num_imgs)]
    elif model_type == "gmm_eye":
        list_of_models = [[torch.randn((latent_dim,)).to(device),
                   torch.eye(latent_dim).to(device)] for i in range(num_imgs)]
    elif model_type == "gmm_low":
        list_of_models = [[torch.randn((latent_dim,)).to(device),
                   torch.eye(latent_dim).to(device), 1e-3*torch.ones([latent_dim]).to(device)] for i in range(num_imgs)]
    elif model_type == "gmm_low_eye":
        list_of_models = [[torch.randn((latent_dim,)).to(device),
                           torch.tril(torch.ones((latent_dim, latent_dim))).to(device),
                           1e-3*torch.ones([latent_dim]).to(device)] for i in range(num_imgs)]
    elif model_type == 'flow':
        n_flow = 16
        affine = True
        seqfrac = 2
        permute = 'random'
        batch_norm = True
        use_dropout = False
        list_of_models = [get_flow_model(latent_dim, n_flow, affine, seqfrac, permute, batch_norm, use_dropout) for i in range(num_imgs)]
    return list_of_models

def load_params(model, PATH):
    model.load_state_dict(torch.load(PATH))







