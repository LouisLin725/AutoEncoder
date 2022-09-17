# Utilization #
import numpy as np
import torch
import matplotlib.pyplot as plt


pos = np.array([[-1.0, 1.0], [0.0, 1.0], [0.3, 0.3], [-0.7, -0.5], 
                [0, 0.25], [0.1, -0.1], [-0.5, -0.75], [0.35, 0.13],
                [-0.48, -0.25], [1.0, 0.0]])
codes = torch.FloatTensor(pos)

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
