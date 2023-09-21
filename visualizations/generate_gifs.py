import os

import ehtplot.color
import imageio
import matplotlib.pyplot as plt
import numpy as np

from utils.data_utils import _get_envelope
from utils.vis_utils import *
from figures import figures_dir


def true_gif(out_dir):
    """Generates gif of true video.

    Parameters
    ----------
    out_dir : str
        Directory to save output to.

    Returns
    -------
    None
        Returns nothing.
    """
    # True images gif
    images = []

    true_path = true_image_path
    x = np.load(true_path)
    x_01 = (x - x.min()) / (x - x.min()).max()

    for k in range(n_imgs_total):
        plt.figure(figsize=(5, 5))
        mean = x_01[k].reshape(true_image_size, true_image_size)
        plt.imshow(mean, cmap="afmhot_10us", vmin=0, vmax=1)
        plt.axis("off")
        plt.savefig(f'{out_dir}/gif/files/true{str(k).zfill(3)}.png',
                    bbox_inches='tight', pad_inches=0)
        plt.close()

    for k in range(n_imgs_total):
        images.append(imageio.imread(f"{out_dir}/gif/files/true{str(k).zfill(3)}.png"))
    out = f"{out_dir}/gif/true.gif"
    imageio.mimsave(out, images, loop=0)


def target_gif(out_dir, folder_name):
    """Generates gif of target video.

    Parameters
    ----------
    out_dir : str
        Directory to save output to.
    folder_name : str
        Name of folder of our reconstructed results (not full path).
        In this function, mainly used to extract the name of the dataset, as
        this influences the intrinsic resolution of the telescope.

    Returns
    -------
    None
        Returns nothing.
    """
    true_images, true_image_norm = true_img_list(list(range(n_imgs_total)), 1, normalize=True)

    dataset = dataset_from_dir(folder_name)
    object_name = object_from_dataset(dataset)

    target_ims = get_ehtim_ims(true_images, object_name, blur=True)
    target_images = [np.reshape(im.imvec, (im.ydim, im.xdim)) for im in target_ims]

    # Target images gif
    images = []

    for k in range(n_imgs_total):
        plt.figure(figsize=(5, 5))
        target = target_images[k].reshape(true_image_size, true_image_size) * true_image_norm
        plt.imshow(target, cmap="afmhot_10us", vmin=0, vmax=1)
        plt.axis("off")
        plt.savefig(f'{out_dir}/gif/files/target{str(k).zfill(3)}.png',
                    bbox_inches='tight', pad_inches=0)
        plt.close()

    for k in range(n_imgs_total):
        images.append(imageio.imread(f"{out_dir}/gif/files/target{str(k).zfill(3)}.png"))
    out = f"{out_dir}/gif/target.gif"
    imageio.mimsave(out, images, loop=0)


def dirty_gif(out_dir):
    """Generates gif of dirty video.

    This is hardcoded for the m87 dataset right now, to change, change value
    of folder_name variable in this script.

    Parameters
    ----------
    out_dir : str
        Directory to save output to.

    Returns
    -------
    None
        Returns nothing.
    """
    dirty_images = dirty_img_list(dataset_from_dir(folder_name), list(range(n_imgs_total)))

    # Dirty images gif
    images = []
    for k in range(n_imgs_total):
        plt.figure(figsize=(5, 5))
        target = dirty_images[k].reshape(true_image_size, true_image_size)
        plt.imshow(target, cmap="afmhot_10us")
        plt.axis("off")
        plt.savefig(f'{out_dir}/gif/files/dirty{str(k).zfill(3)}.png',
                    bbox_inches='tight', pad_inches=0)
        plt.close()

    for k in range(n_imgs_total):
        images.append(imageio.imread(f"{out_dir}/gif/files/dirty{str(k).zfill(3)}.png"))
    out = f"{out_dir}/gif/dirty.gif"
    imageio.mimsave(out, images, loop=0)


def mean_gif(out_dir, folder_name, envelope_params=None, epoch=None):
    """Generates gif of our mean reconstruction video.

    Parameters
    ----------
    out_dir : str
        Directory to save output to.
    folder_name : str
        Name of folder of our reconstructed results (not full path).
    envelope_params : None or tuple
        None if no envelope is used, else, a tuple of ({'sq', 'circ'}, int, int)
        describing the parameters of an envelope to place on the gif.
    epoch : None or int
        Epoch of reconstruction results to use (usually around 150k is good).
        None to use latest epoch available in folder_name.

    Returns
    -------
    None
        Returns nothing.
    """
    if envelope_params is not None:
        etype, ds1, ds2 = envelope_params
        envelope = _get_envelope(true_image_size, etype=etype, ds1=ds1, ds2=ds2)

    # Reconstructed images gif
    images = []

    for k in range(n_imgs_total):
        plt.figure(figsize=(5, 5))
        mean = recon_img_list(folder_name, k, epoch=epoch).mean(axis=0).squeeze()
        if envelope_params is not None:
            mean = envelope * mean
        plt.imshow(mean, cmap="afmhot_10us", vmin=0, vmax=1)
        plt.axis("off")
        plt.savefig(f'{out_dir}/gif/files/recon{str(k).zfill(3)}.png',
                    bbox_inches='tight', pad_inches=0)
        plt.close()

    for k in range(n_imgs_total):
        images.append(imageio.imread(f"{out_dir}/gif/files/recon{str(k).zfill(3)}.png"))
    out = f"{out_dir}/gif/recon.gif"
    imageio.mimsave(out, images, loop=0)


def std_gif(out_dir, folder_name, epoch=None):
    """Generates gif of our reconstruction standard deviation.

    Parameters
    ----------
    out_dir : str
        Directory to save output to.
    folder_name : str
        Name of folder of our reconstructed results (not full path).
    epoch : None or int
        Epoch of reconstruction results to use (usually around 150k is good).
        None to use latest epoch available in folder_name.

    Returns
    -------
    None
        Returns nothing.
    """
    images = []

    for k in range(n_imgs_total):
        plt.figure(figsize=(5, 5))
        std = recon_img_list(folder_name, k, epoch=epoch).std(axis=0).squeeze()
        plt.imshow(std, cmap="afmhot_10us")
        plt.axis("off")
        plt.savefig(f'{out_dir}/gif/files/std{str(k).zfill(3)}.png',
                    bbox_inches='tight', pad_inches=0)
        plt.close()

    for k in range(n_imgs_total):
        images.append(imageio.imread(f"{out_dir}/gif/files/std{str(k).zfill(3)}.png"))
    out = f"{out_dir}/gif/std.gif"
    imageio.mimsave(out, images)


def baseline_gif(dataset, baseline_name, ext='mp4'):
    """Generates gif of a baseline's reconstruction video.

    Saves output in baseline_results/[dataset]/[baseline_name]/gif.

    Parameters
    ----------
    dataset : {'m87', 'sagA', 'sagA_video'}
        Dataset used for baseline.
    baseline_name : str
        Name of folder (not full path) for desired baseline.
        Assumes folder already contains reconstructions as a result of
        running baseline scripts (e.g. rml_baselines.py).
        Assumes root directory is baseline_results.
    ext : str
        File extension for video (e.g. gif, mp4). Note that gif sometimes
        doesn't look the best because it can't express the full color range,
        in which case mp4 or other ffmpeg format is recommended.

    Returns
    -------
    None
        Returns nothing.
    """
    out_dir = os.path.join(baselines_dir, dataset, baseline_name)
    subdirs = os.path.join(out_dir, 'gif', 'files')
    if not os.path.exists(subdirs):
        os.makedirs(subdirs)

    # Baseline reconstruction gif
    baseline_images, true_image_norm = baseline_img_list(baseline_name, dataset, ambientgan_epoch=25896)

    for k in range(n_imgs_total):
        plt.figure(figsize=(5, 5))
        plt.imshow(baseline_images[k].squeeze() * true_image_norm, cmap="afmhot_10us", vmin=0, vmax=1)
        plt.axis("off")
        plt.savefig(f'{out_dir}/gif/files/{baseline_name}_{str(k).zfill(3)}.png',
                    bbox_inches='tight', pad_inches=0)
        plt.close()

    images = []
    for k in range(n_imgs_total):
        images.append(imageio.imread(f"{out_dir}/gif/files/{baseline_name}_{str(k).zfill(3)}.png"))
    out = f"{out_dir}/gif/{baseline_name}.{ext}"
    if ext == 'gif':
        imageio.mimsave(out, images, loop=0)
    else:
        imageio.mimsave(out, images, format='FFMPEG')


def phase_shift_gif(out_dir, other_img_path):
    """Video with amp of true black hole video, but phase of another image.

    Parameters
    ----------
    out_dir : str
        Directory to save output to.
    other_img_path : str
        Path to other image (recommend a square image i.e. same aspect ratio
        as black hole video).

    Returns
    -------
    None
        Returns nothing.
    """
    # simple image scaling to (nR x nC) size
    def scale(im, nR, nC):
        nR0 = len(im)  # source number of rows
        nC0 = len(im[0])  # source number of columns
        return [[im[int(nR0 * r / nR)][int(nC0 * c / nC)]
                 for c in range(nC)] for r in range(nR)]

    def rgb2gray(rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    # get true images
    true_path = true_image_path
    x = np.load(true_path)
    x_01 = (x - x.min()) / (x - x.min()).max()

    # get other img
    other_img = imageio.imread(other_img_path)
    other_img = rgb2gray(other_img)
    other_img = (other_img - other_img.min()) / (other_img - other_img.min()).max()
    other_img = scale(other_img, true_image_size, true_image_size)

    for k in range(n_imgs_total):
        plt.figure(figsize=(5, 5))

        bh_img = x_01[k].reshape(true_image_size, true_image_size)
        fft_bh = np.fft.fft2(bh_img)
        fft_other = np.fft.fft2(other_img)
        combined = np.abs(fft_bh) * np.exp(1j * np.angle(fft_other))
        combined = np.fft.ifft2(combined)
        combined = np.real(combined)

        plt.imshow(combined, cmap="afmhot_10us", vmin=0, vmax=1)
        plt.axis("off")
        plt.savefig(f'{out_dir}/gif/files/phase-shift_{str(k).zfill(3)}.png',
                    bbox_inches='tight', pad_inches=0)
        plt.close()

    images = []
    for k in range(n_imgs_total):
        images.append(imageio.imread(f"{out_dir}/gif/files/phase-shift_{str(k).zfill(3)}.png"))
    out = f"{out_dir}/gif/phase-shift.gif"
    imageio.mimsave(out, images, loop=0)



if __name__ == '__main__':
    folder_name = "m8764_gmm_closure-phase_deepdecoder_n_imgs_totalimgs_Nonenoise_std_dropout0.0001_layer_size150x6_latent40_seed100_eps0.001_gamma0.01_sigma-cpNone_cphases-min_centroid1e+05-1e+04"  # gamma = 0.01

    object_name = object_from_dataset(dataset_from_dir(folder_name))
    folder_out_dir = os.path.join('results_batched', folder_name, 'figures')
    for dir_ in (folder_out_dir, figures_dir):
        subdirs = os.path.join(dir_, 'gif', 'files')
        if not os.path.exists(subdirs):
            os.makedirs(subdirs)

    # true_gif(figures_dir)
    # target_gif(figures_dir, folder_name)
    # dirty_gif(figures_dir)
    # mean_gif(folder_out_dir, folder_name)
    # std_gif(folder_out_dir, folder_name)
    # phase_shift_gif(figures_dir, 'utils/yuumi.png')

    baseline_names = ['dip-centroid10', 'tv-1', 'tv2-1', 'simple-1']
    for baseline_name in baseline_names:
        baseline_gif(object_name, baseline_name)
