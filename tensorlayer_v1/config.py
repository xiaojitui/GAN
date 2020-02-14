#!/usr/bin/env python
# coding: utf-8

# In[1]:


from easydict import EasyDict as edict
import json


# In[2]:


config = edict()
config.TRAIN = edict()


# In[4]:


## Adam
config.TRAIN.batch_size = 8 
# [16] use 8 if your GPU memory is small, and use [2, 4] in tl.vis.save_images / use 16 for faster training
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9


# In[5]:


## initialize G
config.TRAIN.n_epoch_init = 100
# config.TRAIN.lr_decay_init = 0.1
# config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)


# In[6]:


## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 2000
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)


# In[7]:


## train set location


#### data pairs: HR vs LRx4
#### HR: shape = [4*x, 4*y, 3]
#### LRx4: shape = [x, y, 3]


config.TRAIN.hr_img_path = 'data/data_train_HR/'
config.TRAIN.lr_img_path = 'data/data_train_LR_bicubic/X4/'

config.VALID = edict()
## test set location
config.VALID.hr_img_path = 'data/data_valid_HR/'
config.VALID.lr_img_path = 'data/data_valid_LR_bicubic/X4/'


# In[8]:



def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")



