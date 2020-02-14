#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time
import random
import numpy as np
import scipy, multiprocessing
import tensorflow as tf
import tensorlayer as tl
from model import get_G, get_D
from config import config


# In[ ]:





# In[ ]:


# tl.files.load_file_list
# tl.vis.read_images
# tl.models.vgg19
# tl.cost.mean_squared_error
# tl.cost.sigmoid_cross_entropy


# In[ ]:





# In[1]:


def loadparams():
    batch_size = config.TRAIN.batch_size  # 8, 16
    # use 8 if your GPU memory is small, and change [4, 4] in tl.vis.save_images to [2, 4]
    
    lr_init = config.TRAIN.lr_init # 1e-4
    beta1 = config.TRAIN.beta1 # 0.9
    
    ## initialize G
    n_epoch_init = config.TRAIN.n_epoch_init # 100
    
    ## adversarial learning (SRGAN)
    n_epoch = config.TRAIN.n_epoch # 2000
    lr_decay = config.TRAIN.lr_decay # 0.9
    decay_every = config.TRAIN.decay_every # 1000
    
    shuffle_buffer_size = 128

    # create folders to save result images and trained models
    save_dir = "samples"
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "models"
    tl.files.exists_or_mkdir(checkpoint_dir)


# In[ ]:





# In[ ]:


'''
final return is:
train_ds = lr_patch, hr_patch
hr_patch is 384*384, all hr imgs
lr_patch is the resize of hr img to 96*96, will be use to generate back some fake hr
both are generated from hr imgs
'''

def get_train_data():
    # load dataset
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
        # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
        # valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
        # valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the entire train set.
    train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
        # for im in train_hr_imgs:
        #     print(im.shape)
        # valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
        # for im in valid_lr_imgs:
        #     print(im.shape)
        # valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
        # for im in valid_hr_imgs:
        #     print(im.shape)
        
    # dataset API and augmentation
    def generator_train():
        for img in train_hr_imgs:
            yield img
    
    def _map_fn_train(img):
        # hr: 384*384 then normalize, then random flip
        hr_patch = tf.image.random_crop(img, [384, 384, 3])
        hr_patch = hr_patch / (255. / 2.)
        hr_patch = hr_patch - 1.
        hr_patch = tf.image.random_flip_left_right(hr_patch)
        
        # lr: 96*96
        lr_patch = tf.image.resize(hr_patch, size=[96, 96])
        return lr_patch, hr_patch
    
    train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32))
    # process for each img
    train_ds = train_ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())
        # train_ds = train_ds.repeat(n_epoch_init + n_epoch)
    
    # load train data
    train_ds = train_ds.shuffle(shuffle_buffer_size)
    train_ds = train_ds.prefetch(buffer_size=2)
    train_ds = train_ds.batch(batch_size)
        # value = train_ds.make_one_shot_iterator().get_next()
    
    return train_ds


# In[ ]:


'''
=================

overal algorithm:
1. first, train G for 100 epoch, by G_loss = mse loss of (real_hr vs fake_lr) # pix vs pix compare
2. sencond, train D and G together.
(a) D_loss = sigmoid loss of (real_hr_logits vs 1) + sigmoid loss of (fake_hr_logits vs 0)
(b) G_loss = sigmoid loss of (fake_hr_logits vs 1) + mse loss of (fake_hr vs real_hr) + mse loss of (fake_hr_vgg_feature vs real_hr_vgg_feature)


==================

in initial learning:
1. by G, use lr_patch, to generate some fake hr, as fake_hr
2. then compare with real hr, and calculate mse loss of (fake_hr vs real_hr) # pix vs pix 
# this is used to improve G


==================
later learning:

1. by G, use lr_patch, to generate some fake hr, as fake_hr

2. calculate D_logits
(a). by D, calculate logits of fake_hr as fake_hr_logits
(b). by D, calculate logits of real_hr as real_hr_logits

3. calculate D_loss
(a) sigmoid loss of (real_hr_logits vs 1)
(b) sigmoid loss of (fake_hr_logits vs 0)
(C) sum above two as total D_loss
# use sigmoid loss as: z = label, x = logits, loss = z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))

4. calculate g_loss
# g_loss =  g_gan_loss + mse_loss + vgg_loss
(a) g_gan_loss = 1e-3 * sigmoid loss of (fake_hr_logits vs 1) 
(b) mse_loss = mse loss of (fake_hr vs real_hr) # pix vs pix 
(c) vgg loss = 2e-6 * mse loss of (fake_hr_feature vs real_hr_feature) # feature vs feature
#  calculate vgg feature of fake_hr as fake_hr_feature, and vgg feature of real_hr as real_hr_feature

'''


# In[ ]:





# In[ ]:


def train():
    G = get_G((batch_size, 96, 96, 3))
    D = get_D((batch_size, 384, 384, 3))
    VGG = tl.models.vgg19(pretrained=True, end_with='pool4', mode='static')

    lr_v = tf.Variable(lr_init)
    g_optimizer_init = tf.optimizers.Adam(lr_v, beta_1=beta1) # for init train
    g_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)
    d_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)

    # 3 training steps
    G.train()
    D.train()
    VGG.train()

    # load data
    train_ds = get_train_data()

    # ==============================================
    ## initialize learning (G) ~ 100 epochs
    n_step_epoch = round(n_epoch_init // batch_size)
    for epoch in range(n_epoch_init):
        for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
            if lr_patchs.shape[0] != batch_size: # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            with tf.GradientTape() as tape:
                fake_hr_patchs = G(lr_patchs) 
                mse_loss = tl.cost.mean_squared_error(fake_hr_patchs, hr_patchs, is_mean=True)
            grad = tape.gradient(mse_loss, G.trainable_weights)
            g_optimizer_init.apply_gradients(zip(grad, G.trainable_weights))
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse: {:.3f} ".format(
                epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss))
        if (epoch != 0) and (epoch % 10 == 0):
            tl.vis.save_images(fake_hr_patchs.numpy(), [2, 4], os.path.join(save_dir, 'train_g_init_{}.png'.format(epoch)))
    
    # ==============================================
    ## adversarial learning (G, D)
    n_step_epoch = round(n_epoch // batch_size)
    for epoch in range(n_epoch):
        for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
            if lr_patchs.shape[0] != batch_size: # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                fake_patchs = G(lr_patchs)
                logits_fake = D(fake_patchs)
                logits_real = D(hr_patchs)
                feature_fake = VGG((fake_patchs+1)/2.) # the pre-trained VGG uses the input range of [0, 1]
                feature_real = VGG((hr_patchs+1)/2.)
                d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real))
                d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake))
                d_loss = d_loss1 + d_loss2
                g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake))
                mse_loss = tl.cost.mean_squared_error(fake_patchs, hr_patchs, is_mean=True)
                vgg_loss = 2e-6 * tl.cost.mean_squared_error(feature_fake, feature_real, is_mean=True)
                g_loss = mse_loss + vgg_loss + g_gan_loss
            grad = tape.gradient(g_loss, G.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
            grad = tape.gradient(d_loss, D.trainable_weights)
            d_optimizer.apply_gradients(zip(grad, D.trainable_weights))
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss(mse:{:.3f}, vgg:{:.3f}, adv:{:.3f}) d_loss: {:.3f}".format(
                epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss, vgg_loss, g_gan_loss, d_loss))

        # update the learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            lr_v.assign(lr_init * new_lr_decay)
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)

        if (epoch != 0) and (epoch % 10 == 0):
            tl.vis.save_images(fake_patchs.numpy(), [2, 4], os.path.join(save_dir, 'train_g_{}.png'.format(epoch)))
            G.save_weights(os.path.join(checkpoint_dir, 'g.h5'))
            D.save_weights(os.path.join(checkpoint_dir, 'd.h5'))


# In[ ]:


'''
==================

how to evaluate?
1. pick one lr img from valid folder
2. by G, to generate a hr img
3. compare lr, generated_hr, and bicubic_hr

==================
'''

def evaluate():
    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## if your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)

    ###========================== DEFINE MODEL ============================###
    imid = 64  # 0: item1  81: item2 53: item3  64: item4
    valid_lr_img = valid_lr_imgs[imid]
    valid_hr_img = valid_hr_imgs[imid]
    # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
    valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]
    # print(valid_lr_img.min(), valid_lr_img.max())

    G = get_G([1, None, None, 3])
    G.load_weights(os.path.join(checkpoint_dir, 'g.h5'))
    G.eval()

    valid_lr_img = np.asarray(valid_lr_img, dtype=np.float32)
    valid_lr_img = valid_lr_img[np.newaxis,:,:,:]
    size = [valid_lr_img.shape[1], valid_lr_img.shape[2]]

    out = G(valid_lr_img).numpy()

    print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    print("[*] save images")
    tl.vis.save_image(out[0], os.path.join(save_dir, 'valid_gen.png'))
    tl.vis.save_image(valid_lr_img[0], os.path.join(save_dir, 'valid_lr.png'))
    tl.vis.save_image(valid_hr_img, os.path.join(save_dir, 'valid_hr.png'))

    out_bicu = scipy.misc.imresize(valid_lr_img[0], [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
    tl.vis.save_image(out_bicu, os.path.join(save_dir, 'valid_bicubic.png'))


# In[ ]:





# In[ ]:


## test (if __name__ == '__main__')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train, evaluate')

    args = parser.parse_args()
    tl.global_flag['mode'] = args.mode # 'train' or 'evaluate' 

    loadparams()

    if tl.global_flag['mode'] == 'train':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")




