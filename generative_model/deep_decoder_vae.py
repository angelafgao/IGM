import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# TODO: should probably be class attributes or have param file
#KERNEL_SIZE = 4  # (4, 4) kernel
#INIT_CHANNELS = 8  # initial number of filters
#IMAGE_CHANNELS = 1  # MNIST images are grayscale
#IMAGE_SIZE = 32
#LATENT_DIM = 32  # latent dimension for sampling
#NC = 1


def add_module(self, module):
    self.add_module(str(len(self) + 1), module)


torch.nn.Module.add = add_module


def conv(in_f, out_f, kernel_size, stride=1, pad='zero', bias=False):
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)

    layers = filter(lambda x: x is not None, [padder, convolver])
    return nn.Sequential(*layers)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


def DeepEncoderDecoder(latent_dim=32, num_channels=[100] * 5, image_size=32,
                       num_output_channels=1):
    k = int(np.sqrt(latent_dim))
    in_size = (k, k)
    out_size = (image_size, image_size)
    depth = len(num_channels)
    bn_affine = True
    bn_before_act = False
    scale_x, scale_y = (out_size[0] / in_size[0]) ** (1. / depth), (out_size[1] / in_size[1]) ** (1. / depth)

    dec_hidden_size = [(int(np.ceil(scale_x ** n * in_size[0])),
                        int(np.ceil(scale_y ** n * in_size[1]))) for n in range(1, depth)] + [out_size]
    enc_hidden_size = dec_hidden_size[::-1]

    num_channels = num_channels + [num_channels[-1], num_channels[-1]]

    n_scales = len(num_channels)
    act_fun = nn.ReLU()
    pad = 'reflection'
    bias = False
    filter_size = 1

    if not (isinstance(filter_size, list) or isinstance(filter_size, tuple)):
        filter_size = [filter_size] * n_scales

    Encoder = nn.Sequential()
    Encoder.add(conv(1, num_channels[0], filter_size[0], 1, pad=pad, bias=bias))
    for i in range(depth - 1):
        Encoder.add(conv(num_channels[i], num_channels[i + 1], filter_size[i], 1, pad=pad, bias=bias))
        if i != depth - 1:
            # align_corners: from pytorch.org: if True, the corner pixels of the input and output tensors are aligned, and thus preserving the values at those pixels. Default: False
            # default seems to work slightly better
            Encoder.add(nn.Upsample(size=enc_hidden_size[i + 1], mode='bilinear', align_corners=False))
        if (bn_before_act):
            Encoder.add(nn.BatchNorm2d(num_channels[0], affine=bn_affine))
        if act_fun is not None:
            Encoder.add(act_fun)
        if not bn_before_act:
            Encoder.add(nn.BatchNorm2d(num_channels[0], affine=bn_affine))

    Encoder.add(conv(num_channels[-1], num_channels[-1], 1, pad=pad, bias=bias))
    Encoder.add(nn.Sigmoid())

    in_size = [4, 4]

    Decoder = nn.Sequential()
    Decoder.add(nn.Linear(latent_dim, 100))  # num_channels[0]*in_size[0]*in_size[1]))
    # z = self.fc2(z)
    # print(z.shape)
    # print(z)
    Decoder.add(nn.Unflatten(1, (1, 10, 10)))  # (num_channels[0], in_size[0], in_size[1])))
    # print(z.shape)
    # print(z)
    Decoder.add(conv(1, num_channels[0], filter_size[0], 1, pad=pad, bias=bias))
    for i in range(depth - 1):
        Decoder.add(conv(num_channels[i], num_channels[i + 1], filter_size[i], 1, pad=pad, bias=bias))
        if i != depth - 1:
            # align_corners: from pytorch.org: if True, the corner pixels of the input and output tensors are aligned, and thus preserving the values at those pixels. Default: False
            # default seems to work slightly better
            Decoder.add(nn.Upsample(size=dec_hidden_size[i + 1], mode='bilinear', align_corners=False))
        if (bn_before_act):
            Decoder.add(nn.BatchNorm2d(num_channels[i + 1], affine=bn_affine))
        if act_fun is not None:
            Decoder.add(act_fun)
        if not bn_before_act:
            Decoder.add(nn.BatchNorm2d(num_channels[i + 1], affine=bn_affine))

    Decoder.add(conv(num_channels[-1], num_output_channels, 1, pad=pad, bias=bias))
    Decoder.add(nn.Sigmoid())
    return Encoder, Decoder


class DeepDecoderVAE(nn.Module):
    def __init__(self, latent_dim=32, num_channels=[100] * 5, image_size=32,
                 num_output_channels=1):
        super(DeepDecoderVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.image_size = image_size
        self.num_output_channels = num_output_channels
        Encoder, Decoder = DeepEncoderDecoder(latent_dim=self.latent_dim,
                                              num_channels=self.num_channels,
                                              image_size=self.image_size,
                                              num_output_channels=self.num_output_channels)
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.fc1 = nn.Linear(self.num_channels[0], 128)
        # self.fc1 = nn.Sequential(nn.Linear(num_channels[0], 100),
        #                          nn.ReLU(),
        #                          nn.Linear(100, 128))
        self.fc_mu = nn.Linear(128, self.latent_dim)
        # self.fc_mu = nn.Sequential(nn.Linear(128, 100),
        #                         nn.ReLU(),
        #                         nn.Linear(100, latent_dim))
        self.fc_log_var = nn.Linear(128, self.latent_dim)
        # self.fc_log_var = nn.Sequential(nn.Linear(128, 100),
        #                         nn.ReLU(),
        #                         nn.Linear(100, latent_dim))

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample

    def encode(self, x):
        # encoding
        x = self.Encoder(x)
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_log_var(hidden)
        return mu, logvar

    def decode(self, z):
        return self.Decoder(z)

    def forward(self, x):
        # x: [batch size, 1, 32,32] -> x: [batch size, 32*32]
        # x = x.view(-1, 32*32)
        mu, logvar = self.encode(x)
        # print(mu, logvar)
        z = self.reparameterize(mu, logvar)
        # print(z)
        return self.decode(z), mu, logvar
