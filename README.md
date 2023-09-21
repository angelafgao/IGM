# Ill-Posed Image Reconstruction without an Explicit Image Prior

![overview image](https://github.com/angelafgao/igm/blob/main/mnistdenoising64.png)

## Run examples
For model selection examples, see TBD.

For joint inference for image reconstruction posteriors and IGMs, see ```test_scripts.sh```. There are examples for the denoising example above as well as examples for denoising celebrity faces and black hole compressed sensing.

## Dependencies
General requirements for PyTorch release:
* [pytorch](https://pytorch.org/)

Follow this [installation guide ](https://github.com/tianweiy/SeqMRI/blob/main/docs/INSTALL.md) to build the singularity container to run these scripts.

You will need to install [ehtim](https://achael.github.io/eht-imaging/) as well as [ehtplot](https://github.com/liamedeiros/ehtplot) for visualizing the black hole examples. For more details on how to create your own measurements and use the ehtim package, see [this tutorial](https://github.com/yvette256/ehtim-tutorial).

## Citations
```
@inproceedings{gao2023image,
  title={Image Reconstruction without Explicit Priors},
  author={Gao, Angela F and Leong, Oscar and Sun, He and Bouman, Katherine L},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

```
@article{leong2023ill,
      title={Ill-Posed Image Reconstruction Without an Image Prior},
      author={Leong, Oscar and Gao, Angela F and Sun, He and Bouman, Katherine L},
      journal={arXiv preprint arXiv:2304.05589},
      year={2023}}
```