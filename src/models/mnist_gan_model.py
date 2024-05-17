from typing import Union, Dict, Any, Tuple, Optional

import wandb
import torch
import torch.nn as nn
from torch import Tensor
from pytorch_lightning import LightningModule


class MNISTGANModel(LightningModule):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        latent_dim: int,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.adversarial_loss = torch.nn.MSELoss()

    def forward(self, z, labels) -> Tensor:
        return self.generator(z, labels)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2)
        )
        return [opt_g, opt_d], []

    def training_step(self, batch, batch_idx, optimizer_idx) -> Union[Tensor, Dict[str, Any]]:
        log_dict, loss = self.step(batch, batch_idx, optimizer_idx) #used for both generator and discriminator
        self.log_dict({"/".join(("train", k)): v for k, v in log_dict.items()})
        return loss

    def validation_step(self, batch, batch_idx) -> Union[Tensor, Dict[str, Any], None]:
        log_dict, loss = self.step(batch, batch_idx)
        self.log_dict({"/".join(("val", k)): v for k, v in log_dict.items()})
        return None

    def test_step(self, batch, batch_idx) -> Union[Tensor, Dict[str, Any], None]:
        # TODO: if you have time, try implementing a test step
        raise NotImplementedError

    def step(self, batch, batch_idx, optimizer_idx=None) -> Tuple[Dict[str, Tensor], Optional[Tensor]]:
        # TODO: implement the step method of the GAN model.
        #     : This function should return both a dictionary of losses
        #     : and current loss of the network being optimised.
        #     :
        #     : When training with pytorch lightning, because we defined 2 optimizers in
        #     : the `configure_optimizers` function above, we use the `optimizer_idx` parameter
        #     : to keep a track of which network is being optimised.

        imgs, labels = batch

        batch_size = imgs.shape[0]

        log_dict = {}
        loss = None

        # TODO: Create adversarial ground truths

        #Ground truth for real images
        real_ground_truth = torch.ones((imgs.shape[0], 1))

        #Ground truth for fake images
        fake_ground_truth = torch.zeros((imgs.shape[0], 1))

        # TODO: Create noise and labels for generator input

        #Assuming 1:1 amount for real and fake images
        generator_labels = torch.randint(0, 10, size=(imgs.shape[0], ))

        generator_noise = torch.normal(0, 1, size=(imgs.shape[0], self.latent_dim))

        if optimizer_idx == 0 or not self.training:
            # TODO: generate images and calculate the adversarial loss for the generator
            # HINT: when optimizer_idx == 0 the model is optimizing the generator
            

            # TODO: Generate a batch of images
            fake_imgs = self.generator(generator_noise, generator_labels)
            
            # TODO: Calculate loss to measure generator's ability to fool the discriminator

            #Pass fake images into discriminator 
            fake_validity = torch.reshape(self.discriminator(fake_imgs, generator_labels), shape = (batch_size, 1))
           
            #Compare the validity with the ground truth
            loss = self.adversarial_loss(fake_validity, 1 - fake_ground_truth)
            log_dict['gen_loss'] = loss

   

        if optimizer_idx == 1 or not self.training:
            # TODO: generate images and calculate the adversarial loss for the discriminator
            # HINT: when optimizer_idx == 1 the model is optimizing the discriminator

            # TODO: Generate a batch of images
            fake_imgs = self.generator(generator_noise, generator_labels)
            
            # TODO: Calculate loss for real images
            real_validity = torch.reshape(self.discriminator(imgs, labels), shape = (batch_size, 1))
            
            real_loss = self.adversarial_loss(real_validity, real_ground_truth)

            # TODO: Calculate loss for fake images
            fake_validity = torch.reshape(self.discriminator(fake_imgs, generator_labels), shape = (batch_size, 1))


            fake_loss = self.adversarial_loss(fake_validity, fake_ground_truth)

            # TODO: Calculate total discriminator loss
            loss = real_loss + fake_loss
            log_dict['disc_loss'] = loss

        return log_dict, loss

    def on_epoch_end(self):
        # TODO: implement functionality to log predicted images to wandb
        #     : at the end of each epoch

        # TODO: Create fake images
        #10 random images
        generator_labels = torch.arange(10,)
        generator_noise = torch.normal(0, 1, size=(10, self.latent_dim))
        fake_imgs = self.generator(generator_noise, generator_labels)
        fake_imgs = fake_imgs.permute(0, 2, 3, 1)

        
        wandb_img = []
        for img_idx in range(fake_imgs.shape[0]):
            img = fake_imgs[img_idx,:,:,:]
            img = torch.squeeze(img)
            wandb_img.append(wandb.Image(img, caption=f"Number: {img_idx}"))
        for logger in self.trainer.logger:
            if type(logger).__name__ == "WandbLogger":
                # TODO: log fake images to wandb (https://docs.wandb.ai/guides/track/log/media)
                #     : replace `None` with your wandb Image object

                logger.experiment.log({"gen_imgs": wandb_img})

