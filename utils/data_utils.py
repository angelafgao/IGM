import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as functional
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib
import sys,inspect
from torch.utils.data import TensorDataset
torch.set_default_dtype(torch.float32)
import os
import zipfile 
import torch
import natsort
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
seed = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.autograd import Variable

GPU = torch.cuda.is_available()
if GPU == True:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
    print("num GPUs",torch.cuda.device_count())
else:
    dtype = torch.FloatTensor


torch.set_default_dtype(torch.float32)
import torch.optim as optim
import pickle
import math
from torch import Tensor

kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}


## Create a custom Dataset class
class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
          root_dir (string): Directory with all the images
          transform (callable, optional): transform to be applied to each image sample
        """
        # Read names of images in the root directory
        image_names = os.listdir(root_dir)

        self.root_dir = root_dir
        self.transform = transform 
        self.image_names = natsorted(image_names)

    def __len__(self): 
        return len(self.image_names)

    def __getitem__(self, idx):
        # Get the path to the image 
        img_path = os.path.join(self.root_dir, self.image_names[idx])
        # Load image and convert it to RGB
        img = Image.open(img_path).convert('RGB')
        # Apply transformations to the image
        if self.transform:
            img = self.transform(img)

        return img
    


def forward_model(A, x, task, idx=0, dataset=None):
    if task == 'compressed-sensing' and dataset=="sagA_video":
        y = torch.einsum('ab,bcd->acd', x.reshape(x.shape[0], x.shape[2]*x.shape[2]), A[idx])
    elif task == 'compressed-sensing' and dataset != "sagA_video":
        y = torch.einsum('ab,bcd->acd', x.reshape(x.shape[0], x.shape[2]*x.shape[2]), A)
    elif task == 'phase-retrieval':
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
    elif task == 'super-res':
        y = A(x)
    elif task == 'inpainting':
        y = A * x
    elif task == 'denoising':
        y = x
    return y

def get_measurement_operator(task, dataset, image_size):
    if 'compressed-sensing' in task:
        if dataset=="sagA":
            mat_path = './utils/dft_mat64.npy'
            A = Tensor(np.load(mat_path)).type(dtype).to(device)
            kernels = None
        elif dataset=="sagA_video":
            mat_path = "./utils/sgra_mjd57850_frames64_4.0_to_15.5utc_ngEHT.npz"
            mat = np.load(mat_path)
            A = []
            for i in range(60):
                x = mat[f'arr_{i}']
                x_new = np.zeros([4096, x.shape[0], 2])
                x_new[:,:,0] = np.real(x.transpose())
                x_new[:,:,1] = np.imag(x.transpose())
                A.append(Tensor(x_new).type(dtype).to(device))
            kernels = None
        elif dataset=="m87":
            mat_path = './utils/dft_m87_eht2025.npy'
            A = Tensor(np.load(mat_path)).type(dtype).to(device)
            kernels = None
        elif dataset=="m87-32":
            mat_path = './utils/dft_m87_eht2025_32.npy'
            np_mat = np.load(mat_path)
            A_np = np.zeros((32*32, 2267, 2))
            A_np[:,:,0] = np_mat[:,:,0].T
            A_np[:,:,1] = np_mat[:,:,1].T
            A = Tensor(A_np).type(dtype).to(device)
            kernels = None
        elif dataset=="MNIST":
            mat_path = './utils/dft_m87_eht2025.npy'
            A = Tensor(np.load(mat_path)).type(dtype).to(device)
            kernels = None
    elif 'phase-retrieval' in task:
        if 'gauss' in task:
            n = image_size*image_size
            m = int(0.1 * n)
            A = torch.randn(m,n).to(device) *(1./math.sqrt(2.)) + 1j*torch.randn(m,n).to(device) *(1./math.sqrt(2.)) 
            kernels = None
        else:
            xx, yy = np.meshgrid(np.arange(image_size), np.arange(image_size))
            xx, yy = np.meshgrid(np.arange(image_size), np.arange(image_size))
            ds1 = 3
            ds2 = 8
            envelope = np.zeros([1,1,image_size, image_size])
            ramp =  (image_size//2-(np.max(np.concatenate([np.abs(xx[None,:]-image_size//2+0.5)/2+(ds2*2),np.abs(yy[None,:]-image_size//2+0.5)], axis=0), axis=(0)))[None,None,:]-ds1)/(ds2-ds1)
            envelope = np.zeros([1,1,image_size, image_size]) + ramp[:,:,ds1-1, 2*ds1-1]
            envelope[:,:,ds1:-ds1, ds1*2:-ds1*2] = ramp[:,:,ds1:-ds1, ds1*2:-ds1*2]
            envelope[:,:,ds2:-ds2, ds2*2:-ds2*2] = ramp[:,:,ds2, 2*ds2]
            envelope -= ramp[:,:,ds1-1, 2*ds1-1]
            envelope /= (ramp[:,:,ds2, 2*ds2] - ramp[:,:,ds1-1, 2*ds1-1] )
            Finv = (np.abs(xx-(image_size//2-0.5)) + np.abs(yy - (image_size//2-0.5)))/(image_size-1)
            A = torch.zeros(2, 1, image_size, image_size).to(device)
            A[0] = Tensor(Finv[None, None, :]).to(device) #1/f
            A[1] = Tensor(envelope).to(device)# envelope
            kernels = None
    elif 'phase-problem' in task:
        xx, yy = np.meshgrid(np.arange(image_size), np.arange(image_size))
        xx, yy = np.meshgrid(np.arange(image_size), np.arange(image_size))
        ds1 = 3
        ds2 = 8
        envelope = np.zeros([1,1,image_size, image_size])
        ramp =  (image_size//2-(np.max(np.concatenate([np.abs(xx[None,:]-image_size//2+0.5)/2+(ds2*2),np.abs(yy[None,:]-image_size//2+0.5)], axis=0), axis=(0)))[None,None,:]-ds1)/(ds2-ds1)
        envelope = np.zeros([1,1,image_size, image_size]) + ramp[:,:,ds1-1, 2*ds1-1]
        envelope[:,:,ds1:-ds1, ds1*2:-ds1*2] = ramp[:,:,ds1:-ds1, ds1*2:-ds1*2]
        envelope[:,:,ds2:-ds2, ds2*2:-ds2*2] = ramp[:,:,ds2, 2*ds2]
        envelope -= ramp[:,:,ds1-1, 2*ds1-1]
        envelope /= (ramp[:,:,ds2, 2*ds2] - ramp[:,:,ds1-1, 2*ds1-1] )
        Finv = (np.abs(xx-(image_size//2-0.5)) + np.abs(yy - (image_size//2-0.5)))/(image_size-1)
        A = torch.zeros(2, 1, image_size, image_size).to(device)
        A[0] = Tensor(Finv[None, None, :]).to(device) #1/f
        A[1] = Tensor(envelope).to(device)# envelope
        kernels = None
    else:
        A = None
        kernels = None
    return A, kernels

def get_true_and_noisy_data(image_size, 
                            sigma, 
                            num_imgs_total, 
                            dataset,
                            class_idx,
                            task,
                            objective,
                            front_facing):
    ######### GET DATA ################
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

    # Get noisy imgs
    noisy_imgs = []
    bad = np.array([2,3,35,38,40,48,58,59,60,69]) # for 100

    if front_facing == True and dataset == "CelebA":
        num_imgs_total += np.sum(bad <= num_imgs_total) + 1
        print(num_imgs_total)

    if dataset == 'MNIST':
        if objective == 'learning':
            list_of_transforms = transforms.Compose([transforms.ToTensor(), 
                                                    transforms.Resize(size=(image_size,image_size))])
            test_imgs = datasets.MNIST('./data', train=False, download=False,
                                  transform=list_of_transforms)
        
            if class_idx is not None:
                idx = test_imgs.targets==class_idx
                test_imgs.targets = test_imgs.targets[idx]
                test_imgs.data = test_imgs.data[idx]    

            test_imgs.data = test_imgs.data[:num_imgs_total]
    
        elif objective == 'model-selection':
            test_imgs = []
            num_examples = 1
            list_of_transforms = transforms.Compose([transforms.ToTensor(), 
                                                    transforms.Resize(size=(image_size,image_size))])
            kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

            for i in range(10):
                test_data_i = datasets.MNIST('./data', train=False, download=True,
                                        transform=list_of_transforms)
                idx = test_data_i.targets == i
                test_data_i.targets = test_data_i.targets[idx][:num_examples]
                test_data_i.data = test_data_i.data[idx][:num_examples,:,:]
                test_data_i_loader = torch.utils.data.DataLoader(test_data_i,
                                                      batch_size=num_examples, shuffle=False, **kwargs)
                for batch_idx, (data, label) in enumerate(test_data_i_loader):
                    test_imgs.append(data.to(device))

    elif dataset == 'FashionMNIST':
        list_of_transforms = transforms.Compose([transforms.ToTensor(), 
                                                 transforms.Resize(size=(image_size,image_size))])
        test_imgs = datasets.FashionMNIST('./data', train=False, download=True,
                              transform=list_of_transforms)
    
        idx = test_imgs.targets == 0
        test_imgs.targets = test_imgs.targets[idx][:num_imgs_total]
        test_imgs.data = test_imgs.data[idx][:num_imgs_total]
    elif dataset == 'obama' or dataset == "bond":
        # Path to folder with the dataset
        if dataset == "obama":
            dataset_folder = "./CelebDataProcessed/Barack Obama"#f'{data_root}/img_align_celeba'
        elif dataset == "bond":
            dataset_folder = "./CelebDataProcessed/Daniel Craig"#f'{data_root}/img_align_celeba'
        print(dataset_folder)
        img_folder = f'{dataset_folder}'
        
        # Spatial size of training images, images are resized to this size.
        # Transformations to be applied to each individual image sample
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
             transforms.Grayscale(1)])
        # Load the dataset from file and apply transformations
        test_imgs = CelebADataset(img_folder, transform)

    elif dataset == "sagA" or dataset=="sagA_video":
        path = "/scratch/imaging/projects/image_prior_mf/img64_mad_a0.94_i30_variable_pa1_noscattering.npy"
        x = np.load(path)
        x_01 = (x - x.min())/(x - x.min()).max()
        test_imgs = TensorDataset(Tensor(x_01),torch.ones([x.shape[0],1]))
        
    elif dataset == "m87":
        path = "/scratch/imaging/projects/image_prior_mf/img64_mad_a0.94_i30_variable_pa1_noscattering.npy"
        x = np.load(path)
        x_01 = (x - x.min())/(x - x.min()).max()
        test_imgs = TensorDataset(Tensor(x_01),torch.ones([x.shape[0],1]))
        
    elif dataset == "m87-32":
        path = "/scratch/imaging/projects/image_prior_mf/img64_mad_a0.94_i30_variable_pa1_noscattering.npy"
        x = np.load(path)
        x_01 = (x - x.min())/(x - x.min()).max()
        test_imgs = TensorDataset(Tensor(x_01),torch.ones([x.shape[0],1]))
        
######## GET MEASUREMENT OPERATOR ##################
    As = []
    if 'multi' in task:
        if 'multi-compressed-sensing' in task:
            curr_task = 'compressed-sensing'
        elif 'multi-denoising' in task:
#             print("denoising")
            curr_task = 'denoising'
        A1, kernels = get_measurement_operator(curr_task, dataset, image_size)
        if ('denoising-compressed-sensing' in task):
#             print("cs")
            A2, kernels = get_measurement_operator('compressed-sensing', dataset, image_size)
            print(A2.shape)
        if ('compressed-sensing-phase-retrieval' in task) or ('denoising-phase-retrieval' in task):
#             print("phase retrieval")
            if 'gauss' in task:
                curr_task = 'gauss-phase-retrieval'
            else:
                curr_task = 'phase-retrieval'
            A2, kernels = get_measurement_operator(curr_task, dataset, image_size)
        As = [A1, A2]
        print(A1, A2.shape)
    else:
        As, kernels = get_measurement_operator(task, dataset, image_size)

        
    test_imgs_loader = torch.utils.data.DataLoader(test_imgs,
                                              batch_size=1, 
                                              shuffle=False,
                                              **kwargs)
    if dataset=="m87-32":
        down_samp = transforms.Resize(image_size)
    
    ######## GET NOISY DATA ##################
    true_imgs = []
    noisy_imgs = []
    if dataset == 'CelebA' or dataset == "obama" or dataset == "bond":
        i = 0
        for imgs in enumerate(test_imgs_loader):
            data = imgs[1]
            if i > num_imgs_total:
                break
            else:
                if 'multi' in task:
                    if 'multi-denoising' in task and i <= num_imgs_total // 2:
                        noise = sigma[0]*torch.randn(1,1,image_size,image_size).to(device)
                        x = data.to(device)
                        true_imgs.append(x)
                        y = x + noise
                        noisy_imgs.append(y)
                    elif 'multi-compressed-sensing' in task and i <= num_imgs_total // 2:
                        x = data.to(device)
                        true_imgs.append(x)
                        y = forward_model(As[0], x, 'compressed-sensing')
                        y = y + sigma[1]*torch.randn_like(y).to(device)
                        noisy_imgs.append(y)
                    elif 'denoising-compressed-sensing' in task and i > num_imgs_total // 2:
                        x = data.to(device)
                        true_imgs.append(x)
                        y = forward_model(As[1], x, 'compressed-sensing')
                        y = y + sigma[1]*torch.randn_like(y).to(device)
                        noisy_imgs.append(y)
                    elif 'phase-retrieval' in task and i > num_imgs_total // 2:
                        x = data.to(device)
                        true_imgs.append(x)
                        if 'gauss' in task:
                            task_name = 'gauss-phase-retrieval'
                        else:
                            task_name = 'phase-retrieval'
                        y = forward_model(As[1], x, task_name)
                        y = y + sigma[1]*torch.randn_like(y).to(device)
                        noisy_imgs.append(y)
                    elif 'phase-problem' in task and i > num_imgs_total // 2:
                        x = data.to(device)
                        true_imgs.append(x)
                        y = forward_model(As[1], x, 'phase-problem')
                        y = y + sigma[1]*torch.randn_like(y).to(device)
                        noisy_imgs.append(y)
                else:
                    if task == 'denoising':
                        noise = sigma*torch.randn(1,1,image_size,image_size).to(device)
                        x = data.to(device)
                        true_imgs.append(x)
                        y = x + noise
                        noisy_imgs.append(y)
                    else:
                        x = data.to(device)
                        true_imgs.append(x)
                        y = forward_model(As, x, task)
                        y = y + sigma*torch.randn_like(y).to(device)
                        noisy_imgs.append(y)
            i += 1
    else:
        i = 0
        if objective == 'learning':
            for batch_idx, (data, label) in enumerate(test_imgs_loader):
                if i > num_imgs_total:
                    break
                else:
                    if 'multi' in task:
                        if 'multi-denoising' in task and i <= num_imgs_total // 2:
                            noise = sigma[0]*torch.randn(1,1,image_size,image_size).to(device)
                            x = data.to(device)
                            true_imgs.append(x)
                            y = x + noise
                            noisy_imgs.append(y)
                        elif 'multi-compressed-sensing' in task and i <= num_imgs_total // 2:
                            x = data.to(device)
                            true_imgs.append(x)
                            y = forward_model(As[0], x, 'compressed-sensing')
                            y = y + sigma[1]*torch.randn_like(y).to(device)
                            noisy_imgs.append(y)
                        elif 'denoising-compressed-sensing' in task and i > num_imgs_total // 2:
                            x = data.to(device)
                            true_imgs.append(x)
                            y = forward_model(As[1], x, 'compressed-sensing')
                            y = y + sigma[1]*torch.randn_like(y).to(device)
                            noisy_imgs.append(y)
                        elif 'phase-retrieval' in task and i > num_imgs_total // 2:
                            x = data.to(device)
                            true_imgs.append(x)
                            if 'gauss' in task:
                                task_name = 'gauss-phase-retrieval'
                            else:
                                task_name = 'phase-retrieval'
                            y = forward_model(As[1], x, task_name)
                            y = y + sigma[1]*torch.randn_like(y).to(device)
                            noisy_imgs.append(y)
                        elif 'phase-problem' in task and i > num_imgs_total // 2:
                            x = data.to(device)
                            true_imgs.append(x)
                            y = forward_model(As[1], x, 'phase-problem')
                            y = y + sigma[1]*torch.randn_like(y).to(device)
                            noisy_imgs.append(y)
                    else:
                        if task == 'denoising':
                            noise = sigma*torch.randn(1,1,image_size,image_size).to(device)
                            x = data.to(device)
                            true_imgs.append(x)
                            y = x + noise
                            noisy_imgs.append(y)
                        else:
                            if dataset == "m87-32":
                                data = down_samp(data)
                            else:
                                data = data
                            x = data.to(device)
                            true_imgs.append(x)
                            print(x.shape)
                            print(As.shape)
                            y = forward_model(As, x, task, idx=i, dataset=dataset)
                            if (dataset == "sagA_video" or dataset == "m87") and hasattr(sigma, "__len__"):
                                print("add weird noise to video measurements")
                                print(y.shape)
                                y = y + sigma[i]*torch.randn_like(y).to(device)
                            else:
                                y = y + sigma*torch.randn_like(y).to(device)
                            noisy_imgs.append(y)
                i += 1
                #if task == 'denoising':
                #    noise = sigma*torch.randn(1,1,image_size,image_size).to(device)
                #    x = data.to(device)
                #    true_imgs.append(x)
                #    y = x + noise
                #    noisy_imgs.append(y)
                #else:
                #    x = data.to(device)
                #    true_imgs.append(x)
                #    y = forward_model(A, x, task)
                #    y = y + sigma*torch.randn_like(y).to(device)
                #    noisy_imgs.append(y)
        elif objective == 'model-selection':
            for batch_idx, data in enumerate(test_imgs_loader):
                if task == 'denoising':
                    noise = sigma*torch.randn(1,1,image_size,image_size).to(device)
                    x = data.to(device)
                    true_imgs.append(x)
                    y = x + noise
                    noisy_imgs.append(y)
                else:
                    x = data.to(device)
                    true_imgs.append(x)
                    y = forward_model(As, x, task)
                    y = y + sigma*torch.randn_like(y).to(device)
                    noisy_imgs.append(y)
    return true_imgs, noisy_imgs, As, sigma, kernels
    

