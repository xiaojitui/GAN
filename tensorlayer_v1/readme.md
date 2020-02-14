This is a Super-Resolution GAN (Generative Adversarial Network) model used to increase the image resolution (x4). 
This is developed based on: https://github.com/tensorlayer/srgan/


The datasets used for training are pairs of low-resolution (LR) images (i.e. shape = [h, w, c]) and high-resolution (HR) images (i.e. shape = [4*h, 4*w, c]). 

To train the model, use main.py and '--mode' = 'train'

To evaluate, use main.py and '--mode' = 'evaluate'
(1) pick one LR image from valid folder
(2) by G, to generate a HR image
(3) compare LR, generated_HR, and bicubic_HR

