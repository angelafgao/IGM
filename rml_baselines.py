import os

import ehtim as eh
from ehtim.imaging.imager_utils import imager_func
import ehtplot.color
import numpy as np

from utils.vis_utils import *


def ehtim_baseline(dataset, img_idxs=None, stype='simple', alpha=1,
                   show_updates=False, **kwargs):
    """Reconstruct images using an RML baseline.

    Saves output in baseline_results/[stype]-[alpha].
    Uses the eht-imaging imager_func under the hood.

    Parameters
    ----------
    dataset : {'m87', 'sagA', 'sagA_video'}
        Dataset to assume for reconstruction.
    img_idxs : None or list of int
        Indices of frames (if None, all frames) to reconstruct.
    stype : str
        Type of RML regularizer for imager_func (e.g. 'tv', 'tv2').
    alpha : float
        Weight on
    show_updates : bool
        Whether to display progress of minimizer.
    kwargs : optional
        Additional parameters to pass to imager_func.

    Returns
    -------
    None
        Returns nothing.
    """
    if img_idxs is None:
        img_idxs = list(range(n_imgs_total))

    obs_list, _ = true_obs_list(dataset, img_idxs, normalize=True, add_th_noise=True)
    object_name = object_from_dataset(dataset)
    fov, flux = obs_params[object_name]['fov'], obs_params[object_name]['flux']
    image_size = true_image_size

    # Run baseline optimization
    out_path = f'baseline_results/{dataset}/{stype}-{alpha}'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # random_im = np.random.uniform(0, 1, (image_size, image_size)) * 1e-3
    # empty_im = np.ones((image_size, image_size)) * 1e-4
    fwhm = 50 * eh.RADPERUAS
    # prior = eh.image.Image(empty_im, psize=fov / float(image_size),
    #                        ra=ra, dec=dec, source=object_name)
    for i, idx in enumerate(img_idxs):
        print("--------------------------------------------")
        print(f'{dataset} / baseline {stype} / frame {idx}')
        print("--------------------------------------------")

        obs = obs_list[i]
        res = obs.res()

        prior = eh.image.make_square(obs_list[i], image_size, fov)
        prior = prior.add_gauss(flux, (fwhm, fwhm, 0, 0, 0))
        # prior = prior.add_gauss(flux * 1e-3, (fwhm, fwhm, 0, fwhm, fwhm))

        outim = imager_func(obs, prior, prior, flux,
                            d1='amp', d2='cphase', s1='simple',
                            alpha_s1=alpha, show_updates=show_updates,
                            **kwargs)
        fig = outim.display(cfun='afmhot_10us')

        outim = outim.blur_circ(res)
        outim = imager_func(obs, outim, outim, flux,
                            d1='amp', d2='cphase', s1=stype,
                            alpha_s1=alpha, show_updates=show_updates,
                            **kwargs)
        fig = outim.display(cfun='afmhot_10us')

        outim = outim.blur_circ(res / 2.0)
        outim = imager_func(obs, outim, outim, flux,
                            d1='amp', d2='cphase', s1=stype,
                            alpha_s1=alpha, show_updates=show_updates,
                            **kwargs)

        fig = outim.display(cfun='afmhot_10us')
        fig.savefig(os.path.join(out_path, f'{str(idx).zfill(3)}.png'))
        outim_arr = np.reshape(outim.imvec, (outim.ydim, outim.xdim))
        np.save(os.path.join(out_path, f'{str(idx).zfill(3)}.npy'), outim_arr)
        print("--------------------------------------------")


def main():
    stypes = ['tv', 'tv2', 'simple']
    datasets = ['m87']
    for (dataset, s) in [(d, s) for d in datasets for s in stypes]:
        ehtim_baseline(dataset, stype=s, alpha=1, maxit=100)


if __name__ == '__main__':
    main()
