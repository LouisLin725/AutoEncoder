# Settings
import torch
import torch
from torchvision import datasets
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np

# plot setting
plt.rcParams['image.cmap'] = 'gray'

# Show images


def show_images(images):
    bound = int(np.ceil(np.sqrt(images.shape[0])))
    for index, image in enumerate(images):
        plt.subplot(bound, bound, index+1)
        plt.imshow(image.reshape(28, 28))
        plt.axis('off')


# Device selection
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

# Load model
model_ae = torch.load('mode_AutoEncoder_MNIST.pth')
model_ae.eval()

# DataLoader for testing data
test_loader = torch.utils.data.DataLoader(datasets.MNIST('./dataset', train=False, download=False, transform=transforms.ToTensor()),
                                          batch_size=20)
# Testing
with torch.no_grad():
    for i, (data, target) in enumerate(test_loader):
        inputs = data.view(-1, 28*28)
        print(inputs.shape)
        print('Original Images')
        show_images(inputs)
        plt.show()

        # Forward
        codes, outputs = model_ae(inputs.to(device))
        print(outputs.shape)

        # outputs = outputs.detach().cpu()
        print('Restructured Image by AE')
        show_images(outputs)
        plt.show()
        if i == 0:
            break
