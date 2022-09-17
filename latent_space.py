# Visualization #
import torch
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt

axis_x = []
axis_y = []
answers = []

# Load model
model_encoder = torch.load('model_MNIST_Encoder.pth')
model_decoder = torch.load('model_MNIST_Decoder.pth')


# DataLoader of testing data
test_loader = torch.utils.data.DataLoader(datasets.MNIST('./dataset', train=False, download=False, transform=transforms.ToTensor()),
                                          batch_size=20)
# Device Selection
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

# evaluation mode
model_encoder.eval()
model_decoder.eval()

with torch.no_grad():
    for i, (data, target) in enumerate(test_loader):

        # flatten
        inputs = data.view(-1, 28*28).to(device)
        answers += target.tolist()

        # Forward-pass with no gradient
        model_encoder.zero_grad()
        model_decoder.zero_grad()

        # encode and decode
        codes = model_encoder(inputs)
        decoded = model_decoder(codes)
        codes = codes.detach().cpu()

        # project to latent space
        axis_x += codes[:, 0].tolist()
        axis_y += codes[:, 1].tolist()

print("The codes' shape: ")
print(codes.shape)

answers = np.array(answers)
print(answers)
axis_x = np.array(axis_x)
axis_y = np.array(axis_y)

fig, ax = plt.subplots()
for i in range(len(answers)):
    ax.text(axis_x[i], axis_y[i], str(answers[i]),
            color=plt.cm.Set1(answers[i]))
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()
