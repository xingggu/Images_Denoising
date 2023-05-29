# Images_Denoising
In this task will train a U-Net for image denoising using PyTorch. As dataset, we use the Berkeley Segmentation Dataset (BSDS300). This dataset contains 300 clean color images. For simplicity, we consider Gaussian noise and convert the images to grayscale. The attached file `unet.py` contains a PyTorch implemenation of the U-Net for you to use (This file was downloaded from: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/models/unet.py). 

We use all 200 images in `./BSDS300/images/train` for training. The first 50 images in `./BSDS300/images/test` are used for validation and the remaining 50 for testing. The data preprocessing steps for creating a dataset. It involves converting RGB images to grayscale, scaling pixel values, adding Gaussian noise, and dividing images into non-overlapping chunks. The goal is to generate pairs of noisy and clean images for training a model.
 1. Implement a `torch.utils.data.Dataset` that generates pairs of noisy images and clean ground truth.
 2. Convert the images from the BSDS dataset (which are RGB) to **grayscale**.
 3. Scale the pixel values of the grayscale images to the range of $[0, 1]$.
 4. Add zero-mean Gaussian noise to the clean images. The variance of the noise is specified as `noise_var`.
 5. To reduce computational cost, the images are divided into non-overlapping chunks of a specified size `(chunk_size x chunk_size)`.
 6. If the dimensions of the image are not divisible by chunk_size, the image is cropped until it becomes divisible. For example, an image of size $(512, 512)$ would be split into 16 non-overlapping chunks of size $(128, 128)$.
 7. The final result is a dataset containing multiple chunks derived from each original image. For instance, if there are 200 train images and each is split into $(128, 128)$ chunks, the dataset would consist of 1200 chunks in total.
 
Choose a MSE loss for training and implement the trainig loop, and use PSNR every epoch to validate model does not overfit. Before computing the model's loss, we need to denormalize the model output to map the pixels back to the original range, since all pixels are normalized before being passed into the model.
