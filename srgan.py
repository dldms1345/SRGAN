import os
import imageio
import sys
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid

from models import Generator, Discriminator, VGG_Network
from datasets import ImageDataset

cuda = torch.cuda.is_available()
device = torch.device("cuda:2" if cuda else "cpu")

G = Generator().to(device)
D = Discriminator().to(device)
vgg_net = VGG_Network().to(device)
# Losses
# BCELoss <- real(1) fake(0)이라서 sigma(n=1~N)-log(D(G(Ilr))) 표현 가능
criterion_GAN = nn.BCELoss()
# 유클리드 거리(Ihr을 vgg에, Ilr을 G에 넣은 결과를 vgg에)(MSE)
criterion_content = nn.MSELoss()

G_optimizer = Adam(G.parameters(), lr=0.00001, betas=(0.9, 0.999))
D_optimizer = Adam(D.parameters(), lr=0.00001, betas=(0.9, 0.999))

dataset = ImageDataset(crop_size=96, scale_factor=4, hr_root='/home/dataset/DIV2K/DIV2K_train_HR')
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

# pretrained data?
# scheduler

# Training
# *The number of batches is equal to number of iterations for one epoch.
total_epochs = 150
for epoch in range(total_epochs):
    print('Epoch {}/{}'.format(epoch, total_epochs - 1))
    print('-' * 10)

    for i, imgs in enumerate(dataloader):
        # Configure model input
        lr_imgs = Variable(imgs[0]).to(device)
        hr_imgs = Variable(imgs[1]).to(device)

        ### train Generator
        G_optimizer.zero_grad()

        # Generate a high resolution image from low resolution input
        fake_imgs = G(lr_imgs)
        
        fake_imgs_ = make_grid(fake_imgs, nrow=1, normalize=True)
        save_image(fake_imgs_, "images/tmp/%d.png" % i, normalize=False)
        # print(fake_imgs.size())

        # 각 fake image의 점수(tanh)
        fake_out = D(G(lr_imgs))
        # print(fake_out)
        real_out = D(hr_imgs)
        # print(real_out)
        # print("--------------------------------------------")
        # Adversarial ground truths
        #valid = Variable(Tensor(np.ones()), requires_grad=False)
        #fake = Variable(Tensor(np.zeros((lr_imgs.size(0), 1))), requires_grad=False)
        valid = Variable(torch.ones_like(real_out).to(device), requires_grad=False)
        fake = Variable(torch.zeros_like(fake_out).to(device), requires_grad=False)

        # Adversarial loss
        adversarial_loss = criterion_GAN(D(G(lr_imgs)), valid)

        # Content loss
        content_loss = criterion_content(vgg_net(G(lr_imgs)), vgg_net(hr_imgs))

        # Total loss
        perceptual_loss = content_loss + 1e-3 * adversarial_loss

        perceptual_loss.backward()
        G_optimizer.step()

        ### train Discriminator
        D_optimizer.zero_grad()

        # Loss of real and fake images
        loss_real = criterion_GAN(D(hr_imgs), valid)
        loss_fake = criterion_GAN(D(G(lr_imgs)), fake)
        # Total loss
        D_loss = (loss_real + loss_fake) / 2

        D_loss.backward()
        D_optimizer.step()

        # --------------
        #  Log Progress
        # --------------
        '''
        sys.stdout.write(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, total_epochs, i, len(dataloader), D_loss.item(), perceptual_loss.item())
        )
        '''
        # sample_interval : interval between saving image samples
        # checkpoint_interval : interval between model checkpoints
        sample_interval = 100
        checkpoint_interval = 10
        batches_done = epoch * len(dataloader) + i
        fake_imgs = G(lr_imgs)
        if batches_done % sample_interval == 0:
            # Save image grid with upsampled inputs and SRGAN outputs
            lr_imgs = nn.functional.interpolate(lr_imgs, scale_factor=4)
            fake_imgs = make_grid(fake_imgs, nrow=1, normalize=True)
            lr_imgs = make_grid(lr_imgs, nrow=1, normalize=True)
            img_grid = torch.cat((lr_imgs, fake_imgs), -1)
            save_image(img_grid, "images/%d.png" % batches_done, normalize=False)

    if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G.state_dict(), "saved_models/G_%d.pth" % epoch)
        torch.save(D.state_dict(), "saved_models/discriminator_%d.pth" % epoch)
        print("The end")
