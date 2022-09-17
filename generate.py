# Utilization #
import numpy as np
import torch
import matplotlib.pyplot as plt


pos = np.array([[-1.5, -1.5], [1.5, 1.5], [10, 10], [-100, 100]])
codes = torch.FloatTensor([[-1.5, -1.5], [1.5, 1.5], [0, 0], [-100, 100]])

device = torch.device('cpu')
model_decoder = torch.load('model_MNIST_Decoder.pth',
                           map_location=torch.device('cpu'))
model_decoder.eval()

outputs = model_decoder(codes.to(device))
outputs = outputs.detach().cpu()

# Settings
plt.rcParams['image.cmap'] = 'gray'

# Show images


def show_images(images):
    bound = int(np.ceil(np.sqrt(images.shape[0])))
    for index, image in enumerate(images):
        plt.subplot(bound, bound, index+1)
        plt.imshow(image.reshape(28, 28))
        plt.axis('off')


print('Generated Images by Decoder')
show_images(outputs)
plt.show()
