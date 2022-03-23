# Adversarial Regularization for Convolution Filters

## Authors

 Dmitrii Gavrilev, Evgeniy Garsiya, Lina Bashaeva, Farid Davletshin, Dmitriy Gilyov

TA: Nikita Balabin

## Project Description

Discriminators in GAN learning helps to distinguish real images from generated. Also we know that convolution filters are 2D matrices. If we train the model on the large amount of data we will see that filters have some patterns. However, if we train the model on the small data, the filters looks random even after training. 
The question is: *Can we apply discriminator to regularize the new filters while training on small data?*

## Results

Kernels without regularization / Kernels with regularization

![image](https://user-images.githubusercontent.com/64730991/159608379-b3487a6e-0073-4b5a-ac4f-94927005059b.png)


## Repository organization

 - `model_training.ipynb` contains all the necessary code to train a CNN model without regularization on large dataset (in our case, CIFAR-100).
 - Directory `models` contains weights for models trained on CIFAR-100 (they differ only in random initialization).
 - `training_without_regularization.ipynb` contains training model on a small subset of CIFAR-10 (100 samples).
 - `adversarial_regularization.ipynb` is needed to reproduce our main results. In this notebook, we train a model with adversarial regularization induced by a discriminator on the filters from a 7x7 convolutional layer.
 - Directory `additional_experiments` contains the rest of the experiments.
