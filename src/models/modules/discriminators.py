import torch
import torch.nn as nn
import numpy as np


class Discriminator(nn.Module):
    def __init__(
            self,
            n_classes: int,
            channels: int,
            img_size: int,
    ):
        super().__init__()
        self.img_shape = (channels, img_size, img_size)
        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(self.img_shape)), 512), # 10+32*32 = 1034 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity

class CNNDiscriminator(nn.Module):
    def __init__(
            self,
            n_classes: int,
            channels: int,
            img_size: int,
    ):
        super().__init__()
        self.img_shape = (channels, img_size, img_size)
        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.encode = nn.Sequential(nn.Linear(1034, 512), nn.LeakyReLU(0.2, inplace=True), nn.Linear(512, 256))
        self.model = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1), #output = (16, 16, 4)
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1), #output = (8, 8, 16)
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1), #output = (4, 4, 64)
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fclayers = nn.Sequential(nn.Linear(4*4*64, 128),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1))

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce inputs
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        d_in = self.encode(d_in)
        d_in = d_in.view(d_in.size(0), -1, 16, 16)
        validity = self.model(d_in)
        validity = validity.view(validity.size(0), -1)
        validity = self.fclayers(validity)
        return validity