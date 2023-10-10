import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import interpolate

from utils.vis_utils import *


def plot_uv_coverage(obs, ax=None, fontsize=14, s=None, cmap='rainbow',
                     add_conjugate=True, xlim=(-9.5, 9.5), ylim=(-9.5, 9.5),
                     shift_inital_time=True, cbar=True, cmap_ticks=[0, 4, 8, 12],
                     time_units='Hrs', ax_label=True, amount_plot=1):
    """
    Plot the uv coverage as a function of observation time.
    x axis: East-West frequency
    y axis: North-South frequency

    Parameters
    ----------
    obs: ehtim.Obsdata
        ehtim Observation object
    ax: matplotlib axis,
        A matplotlib axis object for the visualization.
    fontsize: float, default=14,
        x/y-axis label fontsize.
    s: float,
        Marker size of the scatter points
    cmap : str, default='rainbow'
        A registered colormap name used to map scalar data to colors.
    add_conjugate: bool, default=True,
        Plot the conjugate points on the uv plane.
    xlim, ylim: (xmin/ymin, xmax/ymax), default=(-9.5, 9.5)
        x-axis range in [Giga lambda] units
    shift_inital_time: bool,
        If True, observation time starts at t=0.0
    cmap_ticks: list,
        List of the temporal ticks on the colorbar
    time_units: str,
        Units for the colorbar
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    giga = 10 ** 9
    u = np.concatenate([obsdata['u'] for obsdata in obs.tlist()]) / giga
    v = np.concatenate([obsdata['v'] for obsdata in obs.tlist()]) / giga
    t = np.concatenate([obsdata['time'] for obsdata in obs.tlist()])

    n_plot = int(amount_plot * (len(u) - 1))
    n_plot = 1 if n_plot == 0 else n_plot
    u, v, t = u[:n_plot], v[:n_plot], t[:n_plot]

    if shift_inital_time:
        t -= t.min()

    if add_conjugate:
        u = np.concatenate([u, -u])
        v = np.concatenate([v, -v])
        t = np.concatenate([t, t])

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()

    if time_units == 'mins':
        t *= 60.0

    sc = ax.scatter(u, v, c=t, cmap=plt.cm.get_cmap(cmap), s=s)
    if ax_label == True:
        ax.set_xlabel(r'East-West Freq (u) $[G \lambda]$', fontsize=fontsize)
        ax.set_ylabel(r'North-South Freq (v) $[G \lambda]$', fontsize=fontsize)
    ax.invert_xaxis()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')

    if cbar is True:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3.5%', pad=0.2)
        cbar = fig.colorbar(sc, cax=cax, ticks=cmap_ticks)
        cbar.set_ticklabels(['{} {}'.format(tick, time_units) for tick in cbar.get_ticks()])
    fig.tight_layout()
    return fig


def time_angle_ring(npix):
    nz = 40
    theta = np.linspace(0, 2 * np.pi, nz)
    rad = 13.5
    cy = -rad * np.sin(theta) + (npix // 2 - 3.5)
    cx = rad * np.cos(theta) + (npix // 2 - 1)
    return cx, cy


def time_angle(img_list, verbose=True):
    nz = 40

    npix = img_list[0].shape[0]
    n_imgs = len(img_list)

    img_arr = np.zeros([n_imgs, npix, npix])
    for i in range(n_imgs):
        img_arr[i] = img_list[i]

    theta = np.linspace(0, 2 * np.pi, nz)
    rad = 13.5
    cy = -rad * np.sin(theta) + (npix // 2 - 3.5)
    cx = rad * np.cos(theta) + (npix // 2 - 1)

    x = np.arange(npix)
    xx, yy = np.meshgrid(x, x)

    flattened = np.zeros([n_imgs, nz])
    # %%
    for idx in range(n_imgs):
        f = interpolate.interp2d(xx, yy, img_arr[idx], kind='linear')
        znew = f(cx, cy)
        flattened[idx] = np.diagonal(znew)
        if idx % 10 == 0 and verbose:
            print(f'frame {idx}')

    flattened_temp = np.copy(flattened)
    flattened_temp[flattened > img_arr.max()] = np.nan
    flattened_temp[flattened < img_arr.min()] = np.nan

    x = np.arange(nz)
    y = np.arange(n_imgs)
    # mask invalid values
    xx, yy = np.meshgrid(x, y)
    # get only the valid values
    flattened_temp = np.ma.masked_invalid(flattened_temp)
    xx, yy = np.meshgrid(x, y)
    # get only the valid values
    x1 = xx[~flattened_temp.mask]
    y1 = yy[~flattened_temp.mask]
    newarr = flattened_temp[~flattened_temp.mask]

    flattened_fixed = interpolate.griddata((x1, y1), newarr.ravel(),
                                           (xx, yy),
                                           method='cubic')
    return flattened_fixed


def get_time_angle_arrs(names, img_lists, root_dir, recompute=False):
    time_angle_arrs = []
    for i, (name, img_list) in enumerate(zip(names, img_lists)):
        time_angle_path = os.path.join(root_dir, f'time-angle_{name}.npy')
        if not recompute and os.path.exists(time_angle_path):
            time_angle_arr = np.load(time_angle_path)
        else:
            time_angle_arr = time_angle(img_list).transpose()
            np.save(time_angle_path, time_angle_arr)
        time_angle_arrs.append(time_angle_arr)
    return time_angle_arrs


def psnr(x_hat, x_true, maxv=1.):
    x_hat = x_hat.flatten()
    x_true = x_true.flatten()
    mse = np.mean(np.square(x_hat - x_true))
    psnr_ = 10. * np.log(maxv ** 2 / mse) / np.log(10.)
    return psnr_


def psnr_all(x_all, x_true):
    #     print(x_all.shape)
    psnrs = np.zeros(x_all.shape[0])
    for i in range(x_all.shape[0]):
        # plt.imshow(np.concatenate([x_all[i], x_true[i]], axis=1), cmap="gray")
        psnrs[i] = psnr(x_all[i], x_true[i])
    #     print(psnrs)
    return np.mean(psnrs)


def pnccrs_all(x_all, x_true):
    #     print(x_all.shape)
    pccrs = np.zeros(x_all.shape[0])
    for i in range(x_all.shape[0]):
        # plt.imshow(np.concatenate([x_all[i], x_true[i]], axis=1), cmap="gray")

        norm_x_all = np.linalg.norm(x_all[i])
        x_all_normed = x_all[i] / norm_x_all
        norm_x_true = np.linalg.norm(x_true[i])
        x_true_normed = x_true[i] / norm_x_true
        c = scipy.signal.correlate2d(x_all_normed, x_true_normed, mode='full')
        pccrs[i] = np.max(c)
    #         pccrs[i] = np.max(scipy.signal.correlate2d(x_all[i], x_true[i], boundary='symm', mode='same'))/np.sum(x_true[i]**2)
    #         print(np.sum(x_true[i]**2))
    return np.mean(pccrs)


def chi2_mean(img_list, object_name, obs, n_samples=10, true_img_norm=1):
    img_list = img_list[:n_samples]
    flux = obs_params[object_name]['flux']
    img_list = [img.squeeze() / true_img_norm * flux for img in img_list]
    im_list = get_ehtim_ims(img_list, object_name, blur=False)

    ttype = 'fast'
    sysnoise = 0
    syscnoise = 0
    cp_uv_min = .1e9
    maxset = False
    snrcut_dict = {key: 0. for key in ['vis', 'amp', 'cphase', 'logcamp', 'camp']}

    chi2amp_mean, chi2cphase_mean = 0, 0
    for i, im in enumerate(im_list):
        chi2amp_mean += obs.chisq(im, dtype='amp', ttype=ttype, systematic_noise=sysnoise,
                                  maxset=maxset, snrcut=snrcut_dict['amp']) \
                        / n_samples
        chi2cphase_mean += obs.chisq(im, dtype='cphase', ttype=ttype, systematic_noise=sysnoise,
                                     systematic_cphase_noise=syscnoise,
                                     maxset=maxset, cp_uv_min=cp_uv_min,
                                     snrcut=snrcut_dict['cphase']) \
                           / n_samples
    return chi2amp_mean, chi2cphase_mean


def plot_bh(img, out_path, scale=True, true_img_norm=1, **kwargs):
    vmin, vmax = (0, 1) if scale else (None, None)
    plt.figure()
    plt.imshow(img.squeeze() * true_img_norm, cmap='afmhot_10us', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0, **kwargs)
    plt.close()
