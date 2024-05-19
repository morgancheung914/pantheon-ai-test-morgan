import torch
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(
            self,
            n_classes: int,
            latent_dim: int,
            channels: int,
            img_size: int,
    ):
        super().__init__()
        self.img_shape = (channels, img_size, img_size)
        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
  
        return img

class CNNGenerator(nn.Module):
    def __init__(
            self,
            n_classes: int,
            latent_dim: int,
            channels: int,
            img_size: int,
    ):
        super().__init__()
        self.img_shape = (channels, img_size, img_size)
        self.label_emb = nn.Embedding(n_classes, n_classes)


        def projection(in_feat, out_feat, normalize=True):
            if normalize:
                layers = [nn.Linear(in_feat, 128),
                nn.BatchNorm1d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(128, out_feat),
                nn.BatchNorm1d(out_feat, 0.8),
                nn.LeakyReLU(0.2, inplace=True)
                ]
            else:
                layers = [nn.Linear(in_feat, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(128, out_feat),
                nn.LeakyReLU(0.2, inplace=True)
                ]
            return layers
        self.model_1 = nn.Sequential(
            *projection(latent_dim + n_classes, 64*8*8, normalize = True), #project to higher dimension
        )

        self.model_2 = nn.Sequential(nn.ConvTranspose2d(64, 16, kernel_size=3, stride=1, padding = 1), #each transpose conv doubles side length 
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.ConvTranspose2d(16, 4, kernel_size=3, stride=2, padding = 1, output_padding = 1),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.ConvTranspose2d(4, 1, kernel_size=3, stride=2, padding= 1, output_padding = 1),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Tanh())
    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        out_1 = self.model_1(gen_input)
        out_1 = out_1.view(-1, 64, 8, 8) #reshape to (8*8) with 16 channels
        img = self.model_2(out_1)
        
        return img
