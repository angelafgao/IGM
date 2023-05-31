# mnist denoising experiments
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 5 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 35 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 75 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 150 --dataset 'MNIST' --sigma 0.5 --image_size 64 --batchGD --class_idx 8 --GMM_EPS 1e-3 --latent_type gmm

# celeb denoising experiments
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 95 --dataset 'bond' --sigma 0.1 --image_size 64 --batchGD  --GMM_EPS 1e-3 --latent_type gmm
python igm_multi_inv_learning_new.py --task 'denoising' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 95 --dataset 'obama' --sigma 0.1 --image_size 64 --batchGD  --GMM_EPS 1e-3 --latent_type gmm


# black hole compressed sensing experiments with fixed noise value for all frequencies
python igm_multi_inv_learning_new.py --task 'compressed-sensing' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 60 --dataset 'sagA' --sigma 2 --image_size 64 --batchGD  --GMM_EPS 1e-3 --latent_type gmm
python igm_multi_inv_learning_new.py --task 'compressed-sensing' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 60 --dataset 'm87' --sigma 2 --image_size 64 --batchGD  --GMM_EPS 1e-3 --latent_type gmm
python igm_multi_inv_learning_new.py --task 'compressed-sensing' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 60 --dataset 'sagA_video' --sigma 2 --image_size 64 --batchGD  --GMM_EPS 1e-3 --latent_type gmm

# realistic noise scenario for black hole compressed sensing
python igm_multi_inv_learning_new.py --task 'compressed-sensing' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 60 --dataset 'sagA_video' --image_size 64 --batchGD  --GMM_EPS 1e-3 --latent_type gmm
python igm_multi_inv_learning_new.py --task 'compressed-sensing' --latent_dim 40 --generator_type 'deepdecoder' --num_imgs 60 --dataset 'm87' --image_size 64 --batchGD  --GMM_EPS 1e-3 --latent_type gmm
