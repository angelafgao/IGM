import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as functional
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib
import os,sys,inspect

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from torch.autograd import Variable
import random
from torch.utils.data import TensorDataset
torch.set_default_dtype(torch.float32)
import torch.optim as optim
import pickle
import math
from torch import Tensor
plt.rcParams['savefig.dpi'] = 200

# from sys import exit
import matplotlib.pyplot as plt
from torch.nn import functional as F


"""## Deep Decoder Def"""
def add_module(self, module):
    self.add_module(str(len(self) + 1), module)
torch.nn.Module.add = add_module

class conv_model(nn.Module):
    def __init__(self, latent_dim, num_layers, num_channels, num_output_channels, out_size, in_size):
        super(conv_model, self).__init__()

        ### parameter setup
        kernel_size = 3
        strides = [1]*(num_layers-1)
        
        ### compute up-sampling factor from one layer to another
        scale_x,scale_y = (out_size[0]/in_size[0])**(1./(num_layers-1)), (out_size[1]/in_size[1])**(1./(num_layers-1))
        hidden_size = [(int(np.ceil(scale_x**n * in_size[0])),
                        int(np.ceil(scale_y**n * in_size[1]))) for n in range(1, (num_layers-1))] + [out_size]
        
        ### hidden layers
        self.net = nn.Sequential()
        self.net.add(nn.Linear(latent_dim, 5*in_size[0]*in_size[1]))
        self.net.add(nn.ReLU())
        self.net.add(nn.Dropout(0.1))
        self.net.add(nn.Unflatten(1, (5, in_size[0], in_size[1])))
        for i in range(num_layers-1):
            
            self.net.add(nn.Upsample(size=hidden_size[i], mode='nearest'))
            conv = nn.Conv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size-1)//2, bias=True)
            self.net.add(conv)
            self.net.add(nn.ReLU())
            self.net.add(nn.Dropout2d(0.1))
            self.net.add(nn.BatchNorm2d( num_channels, affine=True))
        ### final layer
        self.net.add(nn.Conv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size-1)//2, bias=True) )
        self.net.add(nn.ReLU())
        self.net.add(nn.BatchNorm2d( num_channels, affine=True))
        self.net.add(nn.Conv2d(num_channels, num_output_channels, 1, 1, padding=0, bias=True))
        
    def forward(self, x, scale_out=1):
        return self.net(x)*scale_out


def get_net_input(num_channels,w=128,h=128):
    totalupsample = 2**len(num_channels)
    width = int(128/totalupsample)
    height = int(128/totalupsample)
    shape = [1,num_channels[0], width, height]
    net_input = Variable(torch.zeros(shape)).type(dtype)
    net_input.data.uniform_()
    net_input.data *= 1./10
    return net_input

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)

torch.nn.Module.add = add_module


def conv(in_f, out_f, kernel_size, stride=1, pad='zero',bias=False):
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)        

    layers = filter(lambda x: x is not None, [padder, convolver])
    return nn.Sequential(*layers)


def deepdecoder(
        latent_dim,
        in_size,
        out_size,
        num_output_channels=3, 
        num_channels=[128]*5, 
        filter_size=1,
        need_sigmoid=True,
        pad ='reflection', 
        upsample_mode='bilinear', 
        act_fun=nn.ReLU(), # nn.LeakyReLU(0.2, inplace=True) 
        bn_before_act = False,
        bn_affine = True,
        bias=False,
        last_noup=False, # if true have a last extra conv-relu-bn layer without the upsampling before linearly combining them
        dropout_val = 0.01
        ):
    
    depth = len(num_channels)
    scale_x,scale_y = (out_size[0]/in_size[0])**(1./depth), (out_size[1]/in_size[1])**(1./depth)
    hidden_size = [(int(np.ceil(scale_x**n * in_size[0])),
                    int(np.ceil(scale_y**n * in_size[1]))) for n in range(1, depth)] + [out_size]
        
    num_channels = num_channels + [num_channels[-1],num_channels[-1]]
    
    n_scales = len(num_channels) 
    
    if not (isinstance(filter_size, list) or isinstance(filter_size, tuple)) :
        filter_size   = [filter_size]*n_scales
    
    model = nn.Sequential()

    model.add(nn.Linear(latent_dim, num_channels[0]*in_size[0]*in_size[1]))
    model.add(nn.ReLU())
    model.add(nn.Dropout(dropout_val))
    model.add(nn.Unflatten(1, (num_channels[0], in_size[0], in_size[1])))

    for i in range(len(num_channels)-2):
        model.add(conv( num_channels[i], num_channels[i+1],  filter_size[i], 1, pad=pad, bias=bias))
        if upsample_mode!='none' and i != len(num_channels)-2:
            if upsample_mode == 'nearest':
                model.add(nn.Upsample(size=hidden_size[i], mode=upsample_mode))
            else:
                # align_corners: from pytorch.org: if True, the corner pixels of the input and output tensors are aligned, and thus preserving the values at those pixels. Default: False
                # default seems to work slightly better
                model.add(nn.Upsample(size=hidden_size[i], mode=upsample_mode,
                                      align_corners=False))

        if(bn_before_act): 
            model.add(nn.BatchNorm2d( num_channels[i+1] ,affine=bn_affine, track_running_stats=False))
        if act_fun is not None:    
            model.add(act_fun)
        model.add(nn.Dropout2d(dropout_val))
        if not bn_before_act:
            model.add(nn.BatchNorm2d( num_channels[i+1], affine=bn_affine, track_running_stats=False))
    
    if last_noup:
        model.add(conv( num_channels[-2], num_channels[-1],  filter_size[-2], 1, pad=pad, bias=bias))
        model.add(act_fun)
        model.add(nn.Dropout2d(dropout_val))
        model.add(nn.BatchNorm2d( num_channels[-1], affine=bn_affine, track_running_stats=False))
    
    model.add(conv( num_channels[-1], num_output_channels, 1, pad=pad,bias=bias))
    #model.add(nn.Dropout(0.2))
    if need_sigmoid:
        model.add(nn.Sigmoid())
   
    return model