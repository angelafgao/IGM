import json
import os.path

import ehtim as eh
import ehtplot.color
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import interpolate

from utils.vis_utils import *
from utils.figure_utils import *

figures_dir = 'figures'
figures_files_dir = 'figures/files'
# m87_folder = 'm8764_gmm_closure-phase_deepdecoder_60imgs_Nonenoise_std_dropout0.0001_layer_size150x6_latent40_seed100_eps0.001_gamma0.001_sigma-cpNone_cphases-min_centroid100000.0-a'
m87_folder = "m8764_gmm_closure-phase_deepdecoder_60imgs_Nonenoise_std_dropout0.0001_layer_size150x6_latent40_seed100_eps0.001_gamma0.01_sigma-cpNone_cphases-min_centroid1e+05-1e+04"  # gamma = 0.01
sagA_folder = 'sagA_video64_gmm_closure-phase_deepdecoder_60imgs_Nonenoise_std_dropout0.0001_layer_size150x6_latent40_seed100_eps0.001_gamma10.0_sigma-cpNone_cphases-min_centroid1e+10-2e+04'


def svgs_uv_m87(cbar=True):
    [obs], _ = true_obs_list('m87', [0], normalize=False)
    cmap_ticks = range(0, 24, 4)

    for i, amt in enumerate([0.01, 0.3, 0.7, 1]):
        cmap = 'Purples'  # if i == 0 else 'rainbow'
        fig = plot_uv_coverage(obs, cbar=cbar, cmap=cmap, cmap_ticks=cmap_ticks, amount_plot=amt)
        fig.savefig(os.path.join(figures_dir, f'uv_m87_{amt}.svg'), format='svg')
        plt.close(fig)


def _svgs_true_img(img_idxs):
    """Generate svgs and time-angle plots of true video.

    Saves output in figures_files_dir.

    Parameters
    ----------
    img_idxs : list of int
        Indices of frames for which to generate svgs.

    Returns
    -------
    None
        Returns nothing.
    """
    # get true and target images
    true_images, true_img_norm = true_img_list(list(range(n_imgs_total)), 1, normalize=True)

    # create svgs of chosen frames
    for idx in img_idxs:
        plt.figure()
        plt.imshow(true_images[idx].squeeze() * true_img_norm, vmin=0, vmax=1, cmap='afmhot_10us')
        plt.axis('off')
        plt.savefig(os.path.join(figures_dir, f'true{str(idx).zfill(3)}.svg'),
                    format='svg', bbox_inches='tight', pad_inches=0)
        plt.close()

    # time-angle plots
    [time_angle_arr] = get_time_angle_arrs(['truth'], [true_images], figures_files_dir)
    plot_bh(time_angle_arr, os.path.join(figures_files_dir, 'true_time-angle.svg'), true_img_norm=true_img_norm)


def svgs_true_img():
    _svgs_true_img([0, 20, 50])


def _svgs_target_img(object_name, img_idxs):
    """Generates svgs and time-angle plots of target video.

    Saves output in figures_files_dir.

    Parameters
    ----------
    object_name : {'m87', 'sagA'}
        Name of source object.
    img_idxs : list of int
        Indices of frames for which to generate svgs.

    Returns
    -------
    None
        Returns nothing.
    """
    # get true and target images
    true_images, true_img_norm = true_img_list(list(range(n_imgs_total)), 1, normalize=True)
    target_ims = get_ehtim_ims(true_images, object_name, blur=True)
    target_images = [np.reshape(im.imvec, (im.ydim, im.xdim)) for im in target_ims]

    # create svgs of chosen frames
    for i, idx in enumerate(img_idxs):
        plt.figure()
        plt.imshow(target_images[idx].squeeze() * true_img_norm, vmin=0, vmax=1, cmap='afmhot_10us')
        plt.axis('off')
        plt.savefig(os.path.join(figures_files_dir, f'target{str(idx).zfill(3)}.svg'),
                    format='svg', bbox_inches='tight', pad_inches=0)
        plt.close()

    # time-angle plots
    [time_angle_arr] = get_time_angle_arrs(['target'], [target_images], figures_files_dir)
    plot_bh(time_angle_arr, os.path.join(figures_files_dir, 'target_time-angle.svg'), true_img_norm=true_img_norm)


def svgs_target_img():
    _svgs_target_img('m87', [0, 20, 50])


def _svgs_dirty_img(folder_name, img_idxs):
    """Generates svgs and time-angle plots of dirty video.

    Saves output in figures_files_dir.

    Parameters
    ----------
    folder_name : str
        Name of folder of reconstructed results (not full path).
    img_idxs : list of int
        Indices of frames for which to generate svgs.

    Returns
    -------
    None
        Returns nothing.
    """
    # get dirty images
    dirty_images = dirty_img_list(dataset_from_dir(folder_name), list(range(n_imgs_total)))

    # create svgs of chosen frames
    for idx in img_idxs:
        plot_bh(dirty_images[idx],
                os.path.join(figures_files_dir, f'dirty{str(idx).zfill(3)}.svg'),
                scale=False)

    # time-angle plots
    [time_angle_arr] = get_time_angle_arrs(['dirty'], [dirty_images], figures_files_dir)
    plot_bh(time_angle_arr, os.path.join(figures_files_dir, 'dirty_time-angle.svg'), scale=False)


def svgs_dirty_img():
    _svgs_dirty_img(m87_folder, [0, 20, 50])


def _svgs_recon_samples(folder_name, img_idxs, n_samples=3, epoch=None):
    """Generate svgs for individual samples of our reconstruction.

    Saves output in figures_files_dir.

    Parameters
    ----------
    folder_name : str
        Name of folder of reconstructed results (not full path).
    img_idxs : list of int
        Indices of frames for which to generate svgs.
    n_samples : int
        Number of samples for which to generate svgs, for each frame.
    epoch : int
        Epoch of reconstruction results to use (usually around 150k is good).
        None to use latest epoch available in folder_name.

    Returns
    -------
    None
        Returns nothing.
    """
    dataset = dataset_from_dir(folder_name)

    # get samples + create svgs
    for i, img_idx in enumerate(img_idxs):
        image_list = recon_img_list(folder_name, img_idx, epoch=epoch)
        for j in range(n_samples):
            img_sample = image_list[j].squeeze()

            plt.figure()
            plt.imshow(img_sample, cmap='afmhot_10us', vmin=0, vmax=1)
            plt.axis('off')
            plt.savefig(os.path.join(figures_files_dir,
                                     f'recon-sample_{dataset}_{str(img_idx).zfill(3)}_samp{str(j).zfill(3)}.svg'),
                        format='svg', bbox_inches='tight', pad_inches=0)
            plt.close()


def svgs_recon_sample():
    _svgs_recon_samples(m87_folder, [0, 1, n_imgs_total - 1], n_samples=3, epoch=150000)


def _svgs_recon_mean_std(folder_name, img_idxs, epoch=None):
    """Generate svgs and time-angle plots for our mean and std reconstruction.

    Saves output in figures_files_dir.

    Parameters
    ----------
    folder_name : str
        Name of folder of reconstructed results (not full path).
    img_idxs : list of int
        Indices of frames for which to generate svgs.
    epoch : int
        Epoch of reconstruction results to use (usually around 150k is good).
        None to use latest epoch available in folder_name.

    Returns
    -------
    None
        Returns nothing.
    """
    dataset = dataset_from_dir(folder_name)

    # get mean + std of reconstructed images
    means = []
    stds = []
    for img_idx in range(n_imgs_total):
        image_list = recon_img_list(folder_name, img_idx, epoch=epoch)
        img_mean = image_list.mean(axis=0).squeeze()
        img_std = image_list.std(axis=0).squeeze()
        means.append(img_mean)
        stds.append(img_std)

    # create svgs for mean + std frames
    for img_idx in img_idxs:
        plot_bh(means[img_idx],
                os.path.join(figures_files_dir, f'recon-mean_{dataset}_{str(img_idx).zfill(3)}.svg'))
        plot_bh(stds[img_idx],
                os.path.join(figures_files_dir, f'recon-std_{dataset}_{str(img_idx).zfill(3)}.svg'),
                scale=False)

    # time-angle plots for mean + std
    [time_angle_arr] = get_time_angle_arrs(['mean'], [means], figures_files_dir)
    plot_bh(time_angle_arr, os.path.join(figures_files_dir, f'mean_{dataset}_time-angle.svg'))

    [time_angle_arr] = get_time_angle_arrs(['std'], [stds], figures_files_dir)
    plot_bh(time_angle_arr, os.path.join(figures_files_dir, f'std_{dataset}_time-angle.svg'))


def svgs_recon_mean_std():
    _svgs_recon_mean_std(m87_folder, [0, 20, 50], epoch=150000)


def _svgs_baseline(dataset, img_idxs, baseline_names):
    """Generate svgs and time-angle plots for list of specified baselines.

    Saves output in figures_files_dir.

    Parameters
    ----------
    dataset : {'m87', 'sagA', 'sagA_video'}
        Dataset used for baseline.
    img_idxs : list of int
        Indices of frames for which to generate svgs.
    baseline_names : list of str
        List of names of folder (not full path) for desired baselines.
        Assumes folder already contains reconstructions as a result of
        running baseline scripts (e.g. rml_baselines.py).
        Assumes root directory is baseline_results.

    Returns
    -------
    None
        Returns nothing.
    """
    _, true_image_norm = true_img_list(list(range(n_imgs_total)), 1, normalize=True)

    # get baseline images
    all_img_idxs = list(range(n_imgs_total))
    baseline_images = [[] for _ in baseline_names]
    for i, baseline_name in enumerate(baseline_names):
        if not baseline_name.startswith('ambientgan'):
            for img_idx in all_img_idxs:
                path = os.path.join(baselines_dir, dataset, baseline_name, f'{str(img_idx).zfill(3)}.npy')
                baseline_recon = np.load(path).squeeze()
                baseline_images[i].append(baseline_recon)
        else:
            path = os.path.join(baselines_dir, dataset, baseline_name,
                                'final_long_bh2025_recon_ambGAN_25896_epoch_real_noise.npy')
            baseline_recons = np.load(path).squeeze()
            for img_idx in all_img_idxs:
                baseline_recon = baseline_recons[img_idx]
                baseline_images[i].append(baseline_recon)

    # create svgs of chosen frames
    for name, img_list in zip(baseline_names, baseline_images):
        scale_factor = 1 if name.startswith('dip') or name.startswith('ambientgan') else true_image_norm
        for img_idx in img_idxs:
            plt.figure()
            plt.imshow(img_list[img_idx].squeeze() * scale_factor, cmap='afmhot_10us', vmin=0, vmax=1)
            plt.axis('off')
            plt.savefig(os.path.join(figures_files_dir,
                                     f'baseline-{name}_{str(img_idx).zfill(3)}.svg'),
                        format='svg', bbox_inches='tight', pad_inches=0)
            plt.close()

    # get time-angle plots
    time_angle_arrs = get_time_angle_arrs(baseline_names, baseline_images, figures_files_dir,
                                          recompute=False)

    # create svgs of time-angle plots
    for i, (baseline_name, time_angle_arr) in enumerate(zip(baseline_names, time_angle_arrs)):
        scale_factor = 1 if baseline_name.startswith('dip') or baseline_name.startswith('ambientgan') else true_image_norm
        plot_bh(time_angle_arr,
                os.path.join(figures_files_dir, f'baseline-{baseline_name}_time-angle.svg'),
                scale=True,
                true_img_norm=scale_factor)


def svgs_baseline():
    # baseline_names = ['simple-1', 'tv-1', 'tv2-1', 'dip-centroid100000.0']
    baseline_names = ['ambientgan']
    _svgs_baseline('m87', [0, 20, 50], baseline_names)


def _fig_sample_recons(folder_name, img_idxs, n_samples=3, epoch=None):
    """
    |         | truth | target | mean | sample 1 | ... | sample n |
    | frame 1 |
    | ...     |
    | frame n |
    """
    font_size = 16
    plt.rc('font', size=font_size)
    plt.rc('axes', titlesize=font_size, labelsize=font_size)

    dataset = dataset_from_dir(folder_name)
    object_name = object_from_dataset(dataset)
    flux = obs_params[object_name]['flux']

    # get true + target images
    true_images, true_img_norm = true_img_list(img_idxs, flux, normalize=True)
    image_size = true_images[0].shape[0]

    target_ims = get_ehtim_ims(true_images, object_name, blur=True)
    target_images = [np.reshape(im.imvec, (im.ydim, im.xdim)) for im in target_ims]

    # create figure
    n_frames = len(img_idxs)
    n_rows = n_frames
    n_cols = n_samples + 3
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows), constrained_layout=True)

    if epoch is None:
        in_path = latest_epoch_path(os.path.join('results_batched', folder_name, 'files'))
    else:
        in_path = f'results_batched/{folder_name}/files/{str(epoch).zfill(7)}'

    # place images in figure
    for i, img_idx in enumerate(img_idxs):
        image_list = np.load(os.path.join(in_path, f'xsamples_{str(img_idx).zfill(3)}.npy'))
        img_mean = image_list.mean(axis=0)
        img_samples = image_list[:n_samples]

        row = i
        ax[row][0].imshow(true_images[i] * true_img_norm, vmin=0, vmax=1, cmap='afmhot_10us')
        ax[row][1].imshow(target_images[i] * true_img_norm, vmin=0, vmax=1, cmap='afmhot_10us')
        ax[row][2].imshow(img_mean.reshape(image_size, image_size), vmin=0, vmax=1, cmap='afmhot_10us')
        for j in range(n_samples):
            ax[row][j+3].imshow(img_samples[j].reshape(image_size, image_size), vmin=0, vmax=1, cmap='afmhot_10us')
            ax[row][j+3].axis('off')

    # set titles + axis labels
    col_titles = ('Truth', 'Target', 'Ours (mean)', 'Ours (samples)')
    for i, title in enumerate(col_titles):
        ax[0][i].set_title(title)
    for i, idx in enumerate(img_idxs):
        ax[i][0].set_ylabel(f'Frame {idx}')
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    # fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, f'recon_samples_{dataset}.svg'), format='svg', transparent=True)
    print(f'saved sample_recons_{dataset}.svg')


def fig_sample_recons():
    img_idxs = [0, 20, 50]
    _fig_sample_recons(m87_folder, img_idxs, n_samples=4, epoch=150000)


def _fig_recon(folder_name, img_idxs, epoch=None):
    """
             | frame 1 | ...  | frame n | time-angle |
    | truth  |
    | dirty  |
    | target |
    | mean   |
    | stdev  |
    """
    font_size = 20
    plt.rc('font', size=font_size)
    plt.rc('axes', titlesize=font_size, labelsize=font_size)

    dataset = dataset_from_dir(folder_name)
    object_name = object_from_dataset(dataset)
    flux = obs_params[object_name]['flux']

    all_imgs_idxs = range(n_imgs_total)

    # get true and dirty images
    true_images, true_img_norm = true_img_list(all_imgs_idxs, flux, normalize=True)
    image_size = true_images[0].shape[0]

    target_ims = get_ehtim_ims(true_images, object_name, blur=True)
    target_images = [np.reshape(im.imvec, (im.ydim, im.xdim)) for im in target_ims]
    dirty_images = dirty_img_list(dataset, all_imgs_idxs)

    # get mean and std. of reconstructed images
    if epoch is None:
        in_path = latest_epoch_path(os.path.join('results_batched', folder_name, 'files'))
    else:
        in_path = f'results_batched/{folder_name}/files/{str(epoch).zfill(7)}'

    mean_images = []
    std_images = []
    for i, img_idx in enumerate(all_imgs_idxs):
        image_list = np.load(os.path.join(in_path, f'xsamples_{str(img_idx).zfill(3)}.npy'))
        img_mean = image_list.mean(axis=0)
        img_std = image_list.std(axis=0)
        mean_images.append(img_mean.reshape(image_size, image_size))
        std_images.append(img_std.reshape(image_size, image_size))

    # get time-angle plots
    time_angle_arrs = get_time_angle_arrs(
        ('truth', 'dirty', 'target', 'mean', 'std'),
        (true_images, dirty_images, target_images, mean_images, std_images),
        figures_files_dir
    )
    cx, cy = time_angle_ring(image_size)

    # create figure
    n_frames = len(img_idxs)
    n_cols = n_frames + 1
    n_rows = 5
    time_angle_ratio = time_angle_arrs[0].shape[1] / time_angle_arrs[0].shape[0]
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 1.75 * n_rows), constrained_layout=True,
                           gridspec_kw={'width_ratios': [1 for _ in range(n_cols - 1)] + [time_angle_ratio]})

    # place images in figure
    for j, img_idx in enumerate(img_idxs):
        ax[0][j].imshow(true_images[img_idx] * true_img_norm, vmin=0, vmax=1, cmap='afmhot_10us')
        ax[1][j].imshow(dirty_images[img_idx] * true_img_norm, cmap='afmhot_10us')
        ax[2][j].imshow(target_images[img_idx] * true_img_norm, vmin=0, vmax=1, cmap='afmhot_10us')
        ax[3][j].imshow(mean_images[img_idx].reshape(image_size, image_size), vmin=0, vmax=1, cmap='afmhot_10us')
        ax[4][j].imshow(std_images[img_idx].reshape(image_size, image_size), cmap='afmhot_10us')

    ax[0][0].plot(cx, cy, c='white', linewidth=2)

    ax[0][n_cols - 1].imshow(time_angle_arrs[0] * true_img_norm, vmin=0, vmax=1, cmap='afmhot_10us')
    ax[1][n_cols - 1].imshow(time_angle_arrs[1] * true_img_norm, cmap='afmhot_10us')
    ax[2][n_cols - 1].imshow(time_angle_arrs[2] * true_img_norm, vmin=0, vmax=1, cmap='afmhot_10us')
    ax[3][n_cols - 1].imshow(time_angle_arrs[3], vmin=0, vmax=1, cmap='afmhot_10us')
    ax[4][n_cols - 1].imshow(time_angle_arrs[4], cmap='afmhot_10us')

    # set titles + axis labels
    col_titles = [f'Frame {idx}' for idx in img_idxs] #+ ['angle vs. time']
    row_titles = ['Truth', 'Dirty', 'Target', 'Our Mean', 'Our Std.']
    for i, title in enumerate(col_titles):
        ax[0][i].set_title(title)
    for i, title in enumerate(row_titles):
        ax[i][0].set_ylabel(title)
    ax[0][n_cols-1].set_title(r'time $\longrightarrow$')
    ax[0][n_cols-1].set_ylabel(r'$\longleftarrow$ angle')

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    # fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, f'recon_{dataset}.svg'), format='svg')
    print(f'saved recon_{dataset}.svg')


def fig_recon():
    img_idxs = [0, 20, 50]
    _fig_recon(m87_folder, img_idxs, epoch=150000)


def _tab_intrinsic_resolution(folder_name, epoch=None):
    """Generates LaTeX table for intrinsic resolution.

    Prints LaTeX table + saves table in figures_dir.

    Parameters
    ----------
    folder_name : str
        Name of folder of reconstructed results (not full path).
    epoch : int
        Epoch of reconstruction results to use (usually around 150k is good).
        None to use latest epoch available in folder_name.

    Returns
    -------
    None
        Returns nothing.
    """
    dataset = dataset_from_dir(folder_name)
    object_name = object_from_dataset(dataset)

    num_imgs = n_imgs_total
    path = true_image_path
    bh_true = np.load(path)
    bh_renorm = lambda x: (x - bh_true.min()) / (bh_true - bh_true.min()).max()
    print(bh_true.min(), bh_true.max(), bh_renorm(bh_true).min(), bh_renorm(bh_true).max())

    # get true images
    true_images, true_img_norm = true_img_list(list(range(n_imgs_total)), 1, normalize=True)
    image_size = true_images[0].shape[0]
    true_im_list = get_ehtim_ims(true_images, object_name, blur=False)

    # get reconstructed images
    if epoch is None:
        in_path = latest_epoch_path(os.path.join('results_batched', folder_name, 'files'))
    else:
        in_path = f'results_batched/{folder_name}/files/{str(epoch).zfill(7)}'
    recon_images = []
    for i, img_idx in enumerate(list(range(n_imgs_total))):
        image_list = np.load(os.path.join(in_path, f'xsamples_{str(img_idx).zfill(3)}.npy'))
        image_list = image_list / true_img_norm * obs_params[object_name]['flux']
        img_mean = image_list.mean(axis=0)
        recon_images.append(img_mean)
    ours_m87 = np.stack(recon_images).reshape(n_imgs_total, image_size, image_size)

    psnrs = []
    pnccrs = []
    target = np.zeros([num_imgs, image_size, image_size])
    for i in [0, 5, 10, 15, 20, 25, 30, 35]:
        # mov = eh.movie.load_hdf5(
        #     '/scratch/imaging/projects/image_prior_mf/IGM-journal/utils/mad_a0.94_i30_variable_pa1_noscattering.hdf5')
        # mov.frames = images_new.reshape([60,64*5*64*5])
        npix = image_size
        targetfov = obs_params[object_name]['fov']
        #     im_list = np.zeros([60, npix, npix])
        #     blur_im_list = np.zeros([60, npix, npix])
        zbl = obs_params[object_name]['flux']
        prior_fwhm = num_imgs * eh.RADPERUAS  #
        idx = 0
        blurred = np.zeros([num_imgs, image_size, image_size])
        for im in true_im_list:
            #         im_list[idx] = im.regrid_image(targetfov, npix).imvec.reshape([npix, npix])
            blur_im_list = im.blur_circ(i * eh.RADPERUAS).regrid_image(targetfov, npix).imvec.reshape([npix, npix])
            blurred[idx] = bh_renorm(blur_im_list)
            idx += 1
            if (idx >= 60):
                break

        if i == 25:
            target = np.copy(blurred)
        #         if idx == 0:
        #             print(blur_im_list.min(), blur_im_list.max(), blurred[idx].min(), blurred[idx].max())
        #         print(ours_m87.min(), ours_m87.max(), blurred.min(), blurred.max())
        psnrs.append(psnr_all(ours_m87, blurred))
        pnccrs.append(pnccrs_all(ours_m87, blurred))
    #         print(pnccrs[-1])
    print([0, 5, 10, 15, 20, 25, 30, 35], psnrs, pnccrs)
    df = pd.DataFrame([psnrs, pnccrs], columns=[0, 5, 10, 15, 20, 25, 30, 35], index=["PSNR", "NCC"])
    latex_table = df.to_latex(formatters={"name": str.upper}, float_format="{:.3f}".format)
    print(latex_table)
    out_path = os.path.join(figures_dir, f'intrinsic-res_{dataset}.txt')
    with open(out_path, 'w') as f:
        f.write(latex_table)


def tab_intrinsic_resolution():
    _tab_intrinsic_resolution(m87_folder, epoch=150000)


def _tab_baselines(folder_name, baseline_names, epoch=None):
    """Generates LaTeX table for comparing ours + baseline PSNR/NCC.

    Table takes the form:
    |         | target | ours | baseline 1 | ... | baseline n |
    | psnr    |
    | pncc    |
    Prints LaTeX table + saves table in figures_dir.

    Parameters
    ----------
    folder_name : str
        Name of folder of reconstructed results (not full path).
    baseline_names : list of str
        List of names of folder (not full path) for desired baselines.
        Assumes folder already contains reconstructions as a result of
        running baseline scripts (e.g. rml_baselines.py).
        Assumes root directory is baseline_results.
    epoch : None or int
        Epoch of reconstruction results to use (usually around 150k is good).
        None to use latest epoch available in folder_name.

    Returns
    -------
    None
        Returns nothing.
    """
    dataset = dataset_from_dir(folder_name)
    object_name = object_from_dataset(dataset)
    flux = obs_params[object_name]['flux']

    all_imgs_idxs = range(n_imgs_total)

    # get true and target images
    true_images, true_image_norm = true_img_list(all_imgs_idxs, flux, normalize=True)
    image_size = true_images[0].shape[0]

    target_ims = get_ehtim_ims(true_images, object_name, blur=True)
    target_images = [np.reshape(im.imvec, (im.ydim, im.xdim)) for im in target_ims]

    # get our reconstructed images
    if epoch is None:
        in_path = latest_epoch_path(os.path.join('results_batched', folder_name, 'files'))
    else:
        in_path = f'results_batched/{folder_name}/files/{str(epoch).zfill(7)}'

    recon_images = []
    for i, img_idx in enumerate(all_imgs_idxs):
        image_list = np.load(os.path.join(in_path, f'xsamples_{str(img_idx).zfill(3)}.npy'))
        # img_sample = image_list[0]
        img_mean = image_list.mean(axis=0)
        recon_images.append(img_mean)

    # get baseline images
    baseline_images = [[] for _ in baseline_names]
    for i, baseline_name in enumerate(baseline_names):
        if not baseline_name.startswith('ambientgan'):
            for img_idx in all_imgs_idxs:
                path = os.path.join(baselines_dir, object_name, baseline_name, f'{str(img_idx).zfill(3)}.npy')
                baseline_recon = np.load(path)
                if baseline_name.startswith('dip'):
                    baseline_recon /= true_image_norm
                baseline_images[i].append(baseline_recon)
        else:
            path = os.path.join(baselines_dir, object_name, baseline_name,
                                'final_long_bh2025_recon_ambGAN_25896_epoch_real_noise.npy')
            baseline_recons = np.load(path).squeeze()
            for img_idx in all_imgs_idxs:
                baseline_recon = baseline_recons[img_idx] / true_image_norm
                baseline_images[i].append(baseline_recon)

    # get psnrs + pnccs
    psnrs = []
    pnccs = []
    target_images = [img * true_image_norm for img in target_images]
    targets = np.stack(target_images)
    recon_images = [img / true_image_norm for img in recon_images]
    for img_list in (recon_images, *baseline_images):
        img_list = [img * true_image_norm for img in img_list]
        imgs = np.stack(img_list).reshape(n_imgs_total, image_size, image_size)
        psnr = psnr_all(imgs, targets)
        pncc = pnccrs_all(imgs, targets)
        psnrs.append(psnr)
        pnccs.append(pncc)

    # get baseline titles
    baseline_titles = list(map(lambda x: x.split('-')[0], baseline_names))
    baseline_titles = [dict_baseline_titles[s] for s in baseline_titles]

    # make table
    df = pd.DataFrame([psnrs, pnccs], columns=['Ours'] + baseline_titles, index=["PSNR", "NCC"])
    latex_table = df.to_latex(formatters={"name": str.upper}, float_format="{:.3f}".format)
    print(latex_table)
    out_path = os.path.join(figures_dir, f'baselines_{dataset}.txt')
    with open(out_path, 'w') as f:
        f.write(latex_table)


def tab_baselines():
    baseline_names = ['simple-1', 'tv-1', 'tv2-1', 'dip-centroid100000.0', 'ambientgan']
    _tab_baselines(m87_folder, baseline_names, epoch=150000)


def _fig_baselines(folder_name, img_idxs, baseline_names, epoch=None):
    """
    |         | target | ours | baseline 1 | ... | baseline n |
    | frame 1 |
    | ...     |
    | frame m |
    | psnr    |
    | pncc    |
    """
    font_size = 24
    plt.rc('font', size=font_size)
    plt.rc('axes', titlesize=font_size, labelsize=font_size)

    dataset = dataset_from_dir(folder_name)
    object_name = object_from_dataset(dataset)
    flux = obs_params[object_name]['flux']

    all_imgs_idxs = range(n_imgs_total)

    # get true and target images
    true_images, true_image_norm = true_img_list(all_imgs_idxs, flux, normalize=True)
    image_size = true_images[0].shape[0]

    target_ims = get_ehtim_ims(true_images, object_name, blur=True)
    target_images = [np.reshape(im.imvec, (im.ydim, im.xdim)) for im in target_ims]

    # create figure
    n_rows = len(img_idxs) + 2
    n_cols = len(baseline_names) + 2
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 1.7 * n_rows), constrained_layout=True,
                           gridspec_kw={'height_ratios': [1 for _ in range(n_rows - 2)] + [0.35, 0.35]})

    # get our reconstructed images
    if epoch is None:
        in_path = latest_epoch_path(os.path.join('results_batched', folder_name, 'files'))
    else:
        in_path = f'results_batched/{folder_name}/files/{str(epoch).zfill(7)}'

    recon_images = []
    for i, img_idx in enumerate(all_imgs_idxs):
        image_list = np.load(os.path.join(in_path, f'xsamples_{str(img_idx).zfill(3)}.npy'))
        # img_sample = image_list[0]
        img_mean = image_list.mean(axis=0)
        recon_images.append(img_mean)

    # get baseline images
    # baseline_images = {k: [] for k in baseline_names}
    baseline_images = [[] for _ in baseline_names]
    for i, baseline_name in enumerate(baseline_names):
        for j, img_idx in enumerate(all_imgs_idxs):
            path = os.path.join(baselines_dir, object_name, baseline_name, f'{str(img_idx).zfill(3)}.npy')
            baseline_recon = np.load(path)
            # baseline_images[baseline_name].append(baseline_recon)
            baseline_images[i].append(baseline_recon)

    # get psnrs + pnccs
    psnrs = []
    pnccs = []
    target_images = [img * true_image_norm for img in target_images]
    targets = np.stack(target_images)
    recon_images = [img / true_image_norm for img in recon_images]
    for img_list in (recon_images, *baseline_images):
        img_list = [img * true_image_norm for img in img_list]
        imgs = np.stack(img_list).reshape(n_imgs_total, image_size, image_size)
        psnr = psnr_all(imgs, targets)
        pncc = pnccrs_all(imgs, targets)
        psnrs.append(psnr)
        pnccs.append(pncc)

    # place images in figure
    for i, img_idx in enumerate(img_idxs):
        row = i
        # ax[row][0].imshow(true_images[img_idx], cmap='afmhot_10us')
        ax[row][0].imshow(target_images[img_idx], vmin=0, vmax=1, cmap='afmhot_10us')
        ax[row][1].imshow(recon_images[img_idx].squeeze() * true_image_norm, vmin=0, vmax=1, cmap='afmhot_10us')
        for j in range(len(baseline_names)):
            ax[row][j+2].imshow(baseline_images[j][img_idx] * true_image_norm, vmin=0, vmax=1, cmap='afmhot_10us')

    ax[n_rows-2][0].text(0.5, 0.5, '--', va='center', ha='center')
    ax[n_rows-1][0].text(0.5, 0.5, '--', va='center', ha='center')
    for i in range(len(baseline_names) + 1):
        ax[n_rows-2][i+1].text(0.5, 0.5, f'{psnrs[i]:.1f}', va='center', ha='center')
        ax[n_rows-1][i+1].text(0.5, 0.5, f'{pnccs[i]:.3f}', va='center', ha='center')

    # set titles + axis labels
    baseline_titles = list(map(lambda x: x.split('-')[0], baseline_names))
    baseline_titles = [dict_baseline_titles[s] for s in baseline_titles]

    col_titles = ['Target', 'Ours'] + baseline_titles
    row_titles = [f'Frame {idx}' for idx in img_idxs] \
                 + [r'PSNR ($\uparrow$)', r'NCC ($\uparrow$)']
    for i, title in enumerate(col_titles):
        ax[0][i].set_title(title)
    for i, title in enumerate(row_titles):
        ax[i][0].set_ylabel(title)

    ax[n_rows-2][0].yaxis.label.set(rotation='horizontal', ha='right', va='center')
    ax[n_rows-1][0].yaxis.label.set(rotation='horizontal', ha='right', va='center')
    for i in range(n_cols):
        ax[n_rows-2][i].set_frame_on(False)
        ax[n_rows-1][i].set_frame_on(False)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    # fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, f'baselines_{dataset}.svg'), format='svg')
    print(f'saved baselines_{dataset}.svg')


def fig_baselines():
    img_idxs = [0, 20, 50]
    baseline_names = ('simple-1', 'tv-1', 'tv2-1')
    _fig_baselines(m87_folder, img_idxs, baseline_names, epoch=150000)


def _fig_ablations(folder_names, img_idxs, col_titles=None, epochs=None):
    """
    |             | target | (\alpha_1, \beta_1) | ... | (\alpha_n, \beta_n) |
    | frame 1     |
    | ...         |
    | frame m     |
    | \chi_\amp^2 |
    | \chi_\ph^2  |
    """
    font_size = 22
    plt.rc('font', size=font_size)
    plt.rc('axes', titlesize=font_size, labelsize=font_size)

    dataset = dataset_from_dir(folder_names[0])
    object_name = object_from_dataset(dataset)
    flux = obs_params[object_name]['flux']

    all_imgs_idxs = range(n_imgs_total)
    if epochs is None:
        epochs = [None for _ in range(len(folder_names))]
    epochs = [latest_epoch(name) if epochs[i] is None else epochs[i] for i, name in enumerate(folder_names)]

    # get true and target images
    true_images, true_image_norm = true_img_list(all_imgs_idxs, flux, normalize=True)

    target_ims = get_ehtim_ims(true_images, object_name, blur=True)
    target_images = [np.reshape(im.imvec, (im.ydim, im.xdim)) for im in target_ims]

    # get our reconstructed images + mean chi^2 for each results folder
    obs_list, _ = true_obs_list(dataset, all_imgs_idxs, normalize=True)
    all_recon_images = []
    chi2amp_list, chi2cphase_list = [], []
    for folder_name, epoch in zip(folder_names, epochs):
        # set up path to store chi^2 to avoid recomputation in future
        folder_figure_path = os.path.join('results_batched', folder_name, 'figures', 'chi2')
        chi2_file = f'chi2-mean_epoch{epoch}.npy'
        chi2_path = os.path.join(folder_figure_path, chi2_file)
        if not os.path.exists(folder_figure_path):
            os.makedirs(folder_figure_path)
        recompute_chi2 = not os.path.exists(chi2_path)

        # get chi^2 and recon image means
        chi2amp_mean, chi2cphase_mean = 0, 0  # mean over all frames for one folder
        recon_images = []
        for i, img_idx in enumerate(all_imgs_idxs):
            image_list = recon_img_list(folder_name, img_idx, epoch=epoch)
            img_mean = image_list.mean(axis=0)
            recon_images.append(img_mean)

            if recompute_chi2:
                print(f'chi^2 frame {img_idx} / {folder_name}')
                chi2amp, chi2cphase = chi2_mean(image_list, object_name, obs_list[i],
                                                 true_img_norm=true_image_norm, n_samples=10)
                chi2amp_mean += chi2amp / n_imgs_total
                chi2cphase_mean += chi2cphase / n_imgs_total

        # store chi^2 value if recomputed
        if recompute_chi2:
            np.save(chi2_path, [chi2amp_mean, chi2cphase_mean])
        else:
            chi2amp_mean, chi2cphase_mean = np.load(chi2_path)

        all_recon_images.append(recon_images)
        chi2amp_list.append(chi2amp_mean)
        chi2cphase_list.append(chi2cphase_mean)

    # create figure
    n_rows = len(img_idxs) + 2
    n_cols = len(folder_names) + 1
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 1.6 * n_rows), constrained_layout=True,
                           gridspec_kw={'height_ratios': [1 for _ in range(n_rows - 2)] + [0.35, 0.35]})

    # place images in figure
    for row, img_idx in enumerate(img_idxs):
        # ax[row][0].imshow(true_images[img_idx], cmap='afmhot_10us')
        ax[row][0].imshow(target_images[img_idx] * true_image_norm, vmin=0, vmax=1, cmap='afmhot_10us')
        for j, recon_images in enumerate(all_recon_images):
            ax[row][j+1].imshow(recon_images[img_idx].squeeze(), vmin=0, vmax=1, cmap='afmhot_10us')

    ax[n_rows-2][0].text(0.5, 0.5, '--', va='center', ha='center')
    ax[n_rows-1][0].text(0.5, 0.5, '--', va='center', ha='center')
    for i in range(len(folder_names)):
        ax[n_rows-2][i+1].text(0.5, 0.5, f'{chi2amp_list[i]:.3f}', va='center', ha='center')
        ax[n_rows-1][i+1].text(0.5, 0.5, f'{chi2cphase_list[i]:.3f}', va='center', ha='center')

    # set titles + axis labels
    if col_titles is None:
        col_titles = []
        original_gamma = None
        original_centroid_wt = None
        for i, folder_name in enumerate(folder_names):
            path = os.path.join('results_batched', folder_name, 'args.json')
            with open(path, 'r') as f:
                data = json.load(f)
            gamma = data["gamma"]
            if data["centroid"] is not None:
                centroid_wt = int(float(data["centroid"][0]))
            else:
                centroid_wt = 0

            gstr, cstr = '', ''
            if i == 0:
                original_gamma = gamma
                original_centroid_wt = centroid_wt
                gstr, cstr = r'\bf', r'\bf'
            else:
                if gamma != original_gamma:
                    gstr = r'\bf'
                if centroid_wt != original_centroid_wt:
                    cstr = r'\bf'

            title = r'$' + gstr + r'{' + fr'\alpha={gamma:.0e}'.replace('-0', '-') + r'}$' + '\n'
            if data["centroid"] is not None:
                title += r'$' + cstr + r'{' + fr'\beta={centroid_wt:.0e}'.replace('+0', '') + r'}$'
            else:
                title += r'$\bf{\beta=0}$'
            col_titles.append(title)

    # col_titles = ['truth', 'target'] + col_titles
    col_titles = ['Target'] + col_titles
    row_titles = [f'Frame {idx}' for idx in img_idxs] + [r'$\chi_{amp.}^2$', r'$\chi_{ph.}^2$']
    for i, title in enumerate(col_titles):
        ax[0][i].set_title(title)
    for i, title in enumerate(row_titles):
        ax[i][0].set_ylabel(title)

    for i in range(n_cols):
        ax[n_rows-2][i].set_frame_on(False)
        ax[n_rows-1][i].set_frame_on(False)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    # fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, f'ablations_{dataset}.svg'), format='svg', transparent=True)
    print(f'saved ablations_{dataset}.svg')


def fig_ablations():
    folder_names = [m87_folder] + [
        "m8764_gmm_closure-phase_deepdecoder_60imgs_Nonenoise_std_dropout0.0001_layer_size150x6_latent40_seed100_eps0.001_gamma0.01_sigma-cpNone_cphases-min",  # no centroid
        "m8764_gmm_closure-phase_deepdecoder_60imgs_Nonenoise_std_dropout0.0001_layer_size150x6_latent40_seed100_eps0.001_gamma0.01_sigma-cpNone_cphases-min_centroid1e+04-1e+04",  # centroid = 1e4
        "m8764_gmm_closure-phase_deepdecoder_60imgs_Nonenoise_std_dropout0.0001_layer_size150x6_latent40_seed100_eps0.001_gamma0.01_sigma-cpNone_cphases-min_centroid1e+06-1e+04",  # centroid = 1e6
        "m8764_gmm_closure-phase_deepdecoder_60imgs_Nonenoise_std_dropout0.0001_layer_size150x6_latent40_seed100_eps0.001_gamma0.001_sigma-cpNone_cphases-min_centroid100000.0-a",  # gamma = 0.001
        "m8764_gmm_closure-phase_deepdecoder_60imgs_Nonenoise_std_dropout0.0001_layer_size150x6_latent40_seed100_eps0.001_gamma0.1_sigma-cpNone_cphases-min_centroid1e+05-1e+04",  # gamma = 0.1
    ]
    folder_names = [
        "m8764_gmm_closure-phase_deepdecoder_60imgs_Nonenoise_std_dropout0.0001_layer_size150x6_latent40_seed100_eps0.001_gamma0.001_sigma-cpNone_cphases-min_centroid1e+06-1e+04_fixedv2",  # gamma=0.001, centroid=1e6
        "m8764_gmm_closure-phase_deepdecoder_60imgs_Nonenoise_std_dropout0.0001_layer_size150x6_latent40_seed100_eps0.001_gamma0.0001_sigma-cpNone_cphases-min_centroid1e+05-1e+04_fixedv2",  # gamma=0.0001, centroid=1e5
        "m8764_gmm_closure-phase_deepdecoder_60imgs_Nonenoise_std_dropout0.0001_layer_size150x6_latent40_seed100_eps0.001_gamma0.001_sigma-cpNone_cphases-min_centroid1e+05-1e+04_fixedv2",  # gamma=0.001, centroid=1e5
        "m8764_gmm_closure-phase_deepdecoder_60imgs_Nonenoise_std_dropout0.0001_layer_size150x6_latent40_seed100_eps0.001_gamma0.01_sigma-cpNone_cphases-min_centroid1e+06-1e+04_fixedv2",  # gamma=0.01, centroid=1e6
        "m8764_gmm_closure-phase_deepdecoder_60imgs_Nonenoise_std_dropout0.0001_layer_size150x6_latent40_seed100_eps0.001_gamma0.01_sigma-cpNone_cphases-min_centroid1e+07-1e+04_fixedv2",  # gamma=0.01, centroid=1e7
    ]
    epochs = [150000 for _ in range(len(folder_names))]
    img_idxs = [0, 20, 50]
    _fig_ablations(folder_names, img_idxs, epochs=epochs)


if __name__ == '__main__':
    for dir_ in [figures_dir, figures_files_dir]:
        if not os.path.exists(dir_):
            os.mkdir(dir_)
    # svgs_uv_m87()
    # svgs_true_img()
    # svgs_target_img()
    # svgs_dirty_img()
    # svgs_recon_mean_std()
    # svgs_baseline()
    # tab_baselines()

    # fig_sample_recons()
    # fig_recon()
    # tab_intrinsic_resolution()
    # fig_baselines()
    fig_ablations()
