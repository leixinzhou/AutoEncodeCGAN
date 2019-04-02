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
batch_size = 128
lr = 0.001
train_epoch = 50

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
E.weight_init(mean=0.0, std=0.02)
G.cuda()
E.cuda()

# Binary Cross Entropy loss
MSE_loss = nn.MSELoss()

# Adam optimizer
E_optimizer = optim.Adam(E.parameters(), lr=lr, betas=(0.5, 0.999))

# results save folder
root = 'run/MNIST_AutoEncodeCDCGAN_results/'
model = 'MNIST_encoder_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

train_hist = {}
train_hist['tr_losses'] = []
train_hist['val_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# label preprocess
onehot = torch.zeros(10, 10)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1)
fill = torch.zeros([10, 10, img_size, img_size])
for i in range(10):
    fill[i, i, :, :] = 1

print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    tr_losses = []
    val_losses = []

    epoch_start_time = time.time()
   
    G.train()
    E.train()
    for x_, y_ in train_loader:
        # train discriminator D
        E.zero_grad()

        mini_batch = x_.size()[0]

        y_fill_ = fill[y_]
        y_label_ = onehot[y_]
        x_, y_fill_, y_label_ = Variable(x_.cuda()), Variable(y_fill_.cuda()),  Variable(y_label_.cuda())


        E_result = E(x_, y_fill_)
        G_result = G(E_result, y_label_)
        loss = MSE_loss(G_result, x_)

        loss.backward()
        E_optimizer.step()

        tr_losses.append(loss.item())

        # train generator G
        E.zero_grad()

    G.eval()
    E.eval()
    for x_, y_ in val_loader:
    # train discriminator D
        mini_batch = x_.size()[0]

        y_fill_ = fill[y_]
        y_label_ = onehot[y_]
        x_, y_fill_, y_label_ = Variable(x_.cuda()), Variable(y_fill_.cuda()),  Variable(y_label_.cuda())


        E_result = E(x_, y_fill_)
        G_result = G(E_result, y_label_)
        loss = MSE_loss(G_result, x_)

        val_losses.append(loss.item())

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('[%d/%d] - ptime: %.2f, loss_tr: %.3f, loss_val: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(tr_losses)),
                                                              torch.mean(torch.FloatTensor(val_losses))))

    train_hist['tr_losses'].append(torch.mean(torch.FloatTensor(tr_losses)))
    train_hist['val_losses'].append(torch.mean(torch.FloatTensor(val_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(E.state_dict(), root + model + 'param.pkl')
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

# show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')

# images = []
# for e in range(train_epoch):
#     img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
#     images.append(imageio.imread(img_name))
# imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)