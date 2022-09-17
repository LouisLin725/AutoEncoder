# Encoder and Decoder Respectively #
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from model import Encoder, Decoder
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# hyperparameters
epochs = 40
batch_size = 64
lr = 0.001
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

# loadind training data
train_loader = DataLoader(datasets.MNIST('./dataset', train=True, download=False, transform=transforms.ToTensor()),
                          batch_size=batch_size,
                          shuffle=True)

# model initialization
model_encoder = Encoder().to(device)
model_decoder = Decoder().to(device)

# optimizer initialization
optimizer_En = torch.optim.Adam(model_encoder.parameters(), lr=lr)
optimizer_De = torch.optim.Adam(model_decoder.parameters(), lr=lr)

# loss function assignment
loss_function = nn.MSELoss().to(device)

train_losses_his = []
for epoch in range(epochs):
    # keep track of training and validation loss
    train_losses = []
    print('running epoch: {}'.format(epoch+1))

    # train the model #
    model_encoder.train()
    model_decoder.train()

    for data, target in tqdm(train_loader):
        # flatten
        inputs = data.view(-1, 784).to(device)

        # initialization
        model_encoder.zero_grad()
        model_decoder.zero_grad()

        # forward pass: Creating the codes by passing inputs to the model
        codes = model_encoder(inputs)
        decodes = model_decoder(codes)

        # calculate the batch loss, criterion: loss_fnc
        loss = loss_function(decodes, inputs)

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # perform a single optimization step (parameter update)
        optimizer_En.step()
        optimizer_De.step()

        # update training loss
        train_losses.append(loss.item() * data.size(0))

    # calculate average losses
    train_loss = np.average(train_losses)
    train_losses_his.append(train_loss)

    # print training/validation statistics
    print('\tTraining Loss: {:.6f}'.format(train_loss))

    # Setting Check point
    torch.save({'epoch': epoch,
                'model_state_dict': model_encoder.state_dict(),
                'optimizer_state_dict': optimizer_En.state_dict(),
                'loss': loss_function,
                'train_loss': train_loss,
                }, "Encoder_checkpoint.pt")

    torch.save({'epoch': epoch,
                'model_state_dict': model_decoder.state_dict(),
                'optimizer_state_dict': optimizer_De.state_dict(),
                'loss': loss_function,
                'train_loss': train_loss,
                }, "Decoder_checkpoint.pt")

# saving both models
torch.save(model_encoder, 'model_MNIST_Encoder.pth')
torch.save(model_decoder, 'model_MNIST_Decoder.pth')

# Plot loss
plt.figure()
plt.plot(train_losses_his, linewidth=2.5)
plt.legend(['Training loss'], fontsize=20, loc='upper right')
plt.xlabel('epoch', fontweight='bold', fontsize=20)
plt.ylabel('Loss', fontweight='bold', fontsize=20)
