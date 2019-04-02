import os, time
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from modules import *
from utils import show_result, show_train_hist
import os
import numpy as np
from skimage.transform import resize

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

adv_img_dir = "data/adv_img/TCla0PCla8.txt"
adv_img_array = np.loadtxt(adv_img_dir, delimiter=",")
img_idx = 50
adv_img = adv_img_array[img_idx,].reshape(28, 28)

img_size = 32
# plt.imshow(adv_img)
# plt.show()
adv_img = resize(adv_img, (img_size, img_size))
adv_img = (adv_img - np.mean(adv_img)) / np.std(adv_img)
adv_tensor = torch.from_numpy(adv_img).float().view(1,1, img_size, img_size).cuda()
# plt.imshow(adv_img)
# plt.show()
#network
G = Generator(128)
E = Encoder(128)
G.load_state_dict(torch.load('run/MNIST_cDCGAN_results/MNIST_cDCGAN_generator_param.pkl'))
E.load_state_dict(torch.load('run/MNIST_AutoEncodeCDCGAN_results/MNIST_encoder_param.pkl'))
G.cuda()
E.cuda()

# Binary Cross Entropy loss
MSE_loss = nn.MSELoss()

# label preprocess
onehot = torch.zeros(10, 10)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1)
fill = torch.zeros([10, 10, img_size, img_size])
for i in range(10):
    fill[i, i, :, :] = 1


G.eval()
E.eval()

x_ = adv_tensor


f, axs = plt.subplots(3, 4)
f.delaxes(axs[2, 3])
axs[0, 0].imshow(x_.squeeze().detach().cpu())
for ax_ind, i in enumerate(torch.range(0, 9), 1):
    lb = i.long()
    y_fill_ = fill[lb].unsqueeze(0)
    y_label_ = onehot[lb].unsqueeze(0)
    x_, y_fill_, y_label_ = Variable(x_.cuda()), Variable(y_fill_.cuda()),  Variable(y_label_.cuda())

    E_result = E(x_, y_fill_)
    G_result = G(E_result, y_label_)
    loss = MSE_loss(G_result, x_)
    
    axs[ax_ind//4, ax_ind%4].imshow(G_result.squeeze().detach().cpu())
    axs[ax_ind//4, ax_ind%4].text(5, 5, '%.3f' % loss.detach().cpu().numpy(), color='white')

plt.show()
