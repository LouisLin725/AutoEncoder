import torch.nn as nn

# Model structure
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        '''
        Encoder, using Tanh to make sure 
        the value is between [-1, 1]
        '''
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, 2),
            nn.Tanh())

    def forward(self, inputs):
        codes = self.encoder(inputs)

        return codes


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Decoder, so as encoder spec.
        self.decoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 64),
            nn.Tanh(),
            nn.Linear(64, 256),
            nn.Tanh(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        outputs = self.decoder(inputs)

        return outputs


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = Encoder()
        # Decoder
        self.decoder = Decoder()

    def forward(self, inputs):
        codes = self.encoder(inputs)
        decodes = self.decoder(codes)

        return codes, decodes
