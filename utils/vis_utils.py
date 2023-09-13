import glob
import os

import ehtim as eh
import numpy as np

true_image_path = './utils/img64_mad_a0.94_i30_variable_pa1_noscattering.npy'
true_image_size = 64  # NOTE: could be directly computed instead
n_imgs_total = 60

baselines_dir = 'baseline_results'
dict_baseline_titles = {'tv': 'TV-RML', 'tv2': 'TSV-RML', 'simple': 'MEM-RML', 'dip': 'DIP', 'ambientgan': 'AmbientGAN'}

obs_params = {
    'm87': {
       'ra': 12.513728717168174,
       'dec': 12.39112323919932,
       'fov': 128.0 * eh.RADPERUAS,  # chosen to ring that is 40 uas for the example 64x64 image
       'flux': 1,
       'array_path': './utils/EHT2025.txt',
       'nt': 1
    },
    'sagA': {
       'ra': 17.761121055553343,
       'dec': -29.00784305556,
       'fov': 160.0 * eh.RADPERUAS,  # chosen to ring that is 50 uas for the example 64x64 image
       'flux': 2,
       'array_path': './utils/ngEHT.txt',
       'nt': 64
    }
}


def dataset_from_dir(folder_name):
    if folder_name.startswith('sagA_video'):
        return 'sagA_video'
    elif folder_name.startswith('sagA'):
        return 'sagA'
    elif folder_name.startswith('m87'):
        return 'm87'
    else:
        raise ValueError('invalid folder_name')


def object_from_dataset(dataset):
    return 'm87' if dataset == 'm87' else 'sagA'


def latest_epoch_path(in_path, pattern='*', verbose=True):
    epoch_path = sorted(glob.glob(os.path.join(in_path, pattern)))[-1]
    if verbose:
        print(f'using epoch {os.path.split(epoch_path)[-1]}')
    return epoch_path


def latest_epoch(folder_name):
    in_path = latest_epoch_path(os.path.join('results_batched', folder_name, 'files'), verbose=False)
    return int(os.path.basename(os.path.normpath(in_path)))


def true_img_list(img_idxs, flux, normalize=True):
    # NOTE: parameter normalize should probably be renamed
    true_images = np.load(true_image_path)
    true_img_norm = 1 / (true_images - true_images.min()).max() if normalize else 1
    true_images = (true_images - true_images.min()) / (true_images - true_images.min()).max()
    # true_img_norm = np.max([np.abs(image).sum() for image in true_images]) if normalize else 1
    true_images = [true_images[i] for i in img_idxs]
    true_images = [image / true_img_norm * flux for image in true_images]
    return true_images, true_img_norm


def get_ehtim_ims(img_list, object_name, blur=False):
    params = obs_params[object_name]
    fov, ra, dec = params['fov'], params['ra'], params['dec']
    image_size = img_list[0].shape[0]
    im_list = [eh.image.Image(image, psize=fov / float(image_size), ra=ra, dec=dec, source=object_name)
               for image in img_list]
    if blur:
        im_list = [im.blur_circ(10.0 * eh.RADPERUAS) for im in im_list]
    return im_list


def true_obs_list(dataset, img_idxs, normalize=True, add_th_noise=False, verbose=False):
    # Get ehtim observations of true images
    object_name = 'm87' if dataset == 'm87' else 'sagA'
    image_size = 64

    if object_name == 'm87':
        ra = 12.513728717168174
        dec = 12.39112323919932
        fov = 128.0 * eh.RADPERUAS  # chosen to ring that is 40 uas for the example 64x64 image
        flux = 1
        array_path = './utils/EHT2025.txt'
        nt = 1
    elif object_name == 'sagA':
        ra = 17.761121055553343
        dec = -29.00784305556
        fov = 160.0 * eh.RADPERUAS  # chosen to ring that is 50 uas for the example 64x64 image
        flux = 2
        array_path = './utils/ngEHT.txt'
        nt = 64
    else:
        raise ValueError('invalid object_name')

    array = eh.array.load_txt(array_path)
    params = {
        'mjd': 57850,  # night of april 6-7, 2017
        'timetype': 'UTC',
        # 'tstart': 4.0,  # start of observations
        # 'tstop': 15.5,  # end of observation
        'tint': 60.0,  # integration time (secs)
        'bw': 1856000000.0,
    }
    tstart = 4.0  # start of observations
    tstop = 15.5  # end of observation

    true_images, true_img_norm = true_img_list(img_idxs, flux, normalize=normalize)

    true_images = [eh.image.Image(image, psize=fov / float(image_size), ra=ra, dec=dec,
                                  rf=226191789062.5, mjd=params['mjd'], source=object_name) \
                   for image in true_images]
    if dataset == 'm87':
        m87_obs = eh.obsdata.load_uvfits('./utils/eht2025.uvfits')
        obs_list = [im.observe_same(m87_obs, add_th_noise=add_th_noise, phasecal=True, ampcal=True,
                                    ttype='fast', verbose=verbose)
                    for im in true_images]
    elif dataset == 'sagA':
        obs_list = [im.observe(array, **params, tadv=(tstop - tstart) * 3600.0 / nt,
                               tstart=tstart, tstop=tstop,
                               ttype='fast', add_th_noise=add_th_noise, phasecal=True, ampcal=True,
                               verbose=verbose)
                    for im in true_images]
    elif dataset == 'sagA_video':
        tadv = (tstop - tstart) * 3600.0 / nt
        obs_list = [im.observe(array, **params, tadv=tadv,
                               tstart=tstart + i * tadv / 3600.0, tstop=tstart + (i + 1) * tadv / 3600.0,
                               ttype='fast', add_th_noise=add_th_noise, phasecal=True, ampcal=True,
                               verbose=verbose)
                    for i, im in enumerate(true_images)]
    else:
        raise ValueError('invalid dataset')

    return obs_list, true_img_norm


def dirty_img_list(dataset, img_idxs):
    true_images, _ = true_img_list(img_idxs, 1, normalize=True)
    image_size = true_images[0].shape[0]
    true_ims = get_ehtim_ims(true_images, object_from_dataset(dataset), blur=False)

    obs_list, _ = true_obs_list(dataset, img_idxs, normalize=True)
    dirty_images = []
    for i, obs in enumerate(obs_list):
        vis, _, A_vis = eh.imaging.imager_utils.chisqdata_vis(obs, true_ims[i], mask=[])
        dirty_image = A_vis.conj().T @ vis
        dirty_images.append(np.abs(dirty_image.reshape(image_size, image_size)))

    return dirty_images


def recon_img_list(folder_name, img_idx, epoch=None):
    if epoch is None:
        in_path = latest_epoch_path(os.path.join('results_batched', folder_name, 'files'), verbose=False)
    else:
        in_path = os.path.join('results_batched', folder_name, 'files', f'{str(epoch).zfill(7)}')
    image_list = np.load(os.path.join(in_path, f'xsamples_{str(img_idx).zfill(3)}.npy'))
    return image_list


def baseline_img_list(baseline_name, dataset, img_idxs=None,
                      ambientgan_epoch=25896):
    if img_idxs is None:
        img_idxs = list(range(n_imgs_total))
    _, true_image_norm = true_img_list(img_idxs, 1, normalize=True)

    baseline_images = []
    if not baseline_name.startswith('ambientgan'):
        for img_idx in img_idxs:
            path = os.path.join(baselines_dir, dataset, baseline_name, f'{str(img_idx).zfill(3)}.npy')
            baseline_recon = np.load(path)
            if baseline_name.startswith('dip'):
                baseline_recon /= true_image_norm
            baseline_images.append(baseline_recon)
    else:
        path = os.path.join(baselines_dir, dataset, baseline_name,
                            f'final_long_bh2025_recon_ambGAN_{ambientgan_epoch}_epoch_real_noise.npy')
        baseline_recons = np.load(path).squeeze()
        for img_idx in img_idxs:
            baseline_recon = baseline_recons[img_idx] / true_image_norm
            baseline_images.append(baseline_recon)
    return baseline_images, true_image_norm
