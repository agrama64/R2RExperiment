# R2RExperiment

This repository aims to reproduce the Recorrupted-to-Recorrupted data augmentation technique for unsupervised image processing.

Link to Paper: https://openaccess.thecvf.com/content/CVPR2021/papers/Pang_Recorrupted-to-Recorrupted_Unsupervised_Deep_Learning_for_Image_Denoising_CVPR_2021_paper.pdf

supervised_model.py - This file applies a standard denoising CNN to the MNIST dataset. It uses ground-truth images for its loss function in training and is thus a supervised NN. This file is provided for comparison.

noisy_noisy.py - This file applies a denoising CNN using the recorrupted-to-recorrupted data augmentation technique. This is an unsupervised algorithm where the input is one noisy image. From this image, the model applies two different recorruptions: one for the inputs and one for the targets. The loss is then determined between the model applied on the recorrupted input and the recorrupted target.

