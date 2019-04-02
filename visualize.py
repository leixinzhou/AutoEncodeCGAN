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
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# training parameters
batch_size = 1
# data_loader
img_size = 32
transform = transforms.Compose([
        transforms.Scale(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)
# network
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
for (x_, y_) in  val_loader:
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
    break