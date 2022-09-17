# Training Process #
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from train_process import train
from model import AutoEncoder
import matplotlib.pyplot as plt


# hyperparameters
epochs = 40
batch_size = 64
lr = 0.001

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
train_loader = DataLoader(datasets.MNIST('./dataset',
                                         train=True,
                                         download=False,
                                         transform=transforms.ToTensor()),
                          batch_size=batch_size,
                          shuffle=True)
# Model variables setting
model_ae = AutoEncoder().to(device)
optimizer = torch.optim.Adam(model_ae.parameters(), lr=lr)
loss_function = nn.MSELoss().to(device)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[10, 40], gamma=0.5)
ckpt_name = 'AE_checkpoint.pt'

# Training
train_loss_his, model_trained = train(
    model_ae, epochs, train_loader, optimizer, loss_function, batch_size, scheduler, ckpt_name)

torch.save(model_trained, 'mode_AutoEncoder_MNIST.pth')


# Plot loss

plt.figure()
plt.plot(train_loss_his, linewidth=2.5)
plt.legend(['Training loss'], fontsize=20, loc='upper right')
plt.xlabel('epoch', fontweight='bold', fontsize=20)
plt.ylabel('Loss', fontweight='bold', fontsize=20)
