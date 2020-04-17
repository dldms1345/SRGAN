import os
import imageio
import sys
import numpy as np
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid

from models import Generator, Discriminator, VGG_Network
from datasets import ImageDataset, ValDataset

from math import log10

cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if cuda else "cpu")

G = Generator().to(device)
D = Discriminator().to(device)
vgg_net = VGG_Network().to(device)


# Losses
# BCELoss <- real(1) fake(0)이라서 sigma(n=1~N)-log(D(G(Ilr))) 표현 가능
criterion_GAN = nn.BCELoss()
# 유클리드 거리(Ihr을 vgg에, Ilr을 G에 넣은 결과를 vgg에)(MSE)
criterion_content = nn.MSELoss()

pre_G_optimizer = Adam(G.parameters(), lr=0.0001, betas=(0.9, 0.999))
G_optimizer = Adam(G.parameters(), lr=0.0001, betas=(0.9, 0.999))
D_optimizer = Adam(D.parameters(), lr=0.0001, betas=(0.9, 0.999))


crop_size=96
scaling_factor=4
dataset = ImageDataset(crop_size=96, scale_factor=4, hr_root='/home/dataset/DIV2K/DIV2K_train_HR')
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

pre_mode = True
# Checking if the pretrained model exist
if len(os.listdir('/home/eh1404/works/SRGAN/SRGAN/pretrained_model')) != 0 and pre_mode == False:
    print("pretrained generator model exist")
    G.load_state_dict(torch.load("pretrained_model/pretrained_G.pth"))
else:
    G.load_state_dict(torch.load("pretrained_model/pretrained_G.pth"))
    G.train()
    # pretraining
    pre_epochs = 400
    for epoch in range(pre_epochs):
        print('Pretrain Epoch {}/{}'.format(epoch, pre_epochs - 1))
        print('-' * 10)

        for i, imgs in enumerate(dataloader):
            # Configure model input
            lr_imgs = imgs[0].to(device)
            hr_imgs = imgs[1].to(device)

            ### train Generator
            pre_G_optimizer.zero_grad()
            G_loss = criterion_content(G(lr_imgs), hr_imgs)
            G_loss.backward()
            G_optimizer.step()
        if epoch % 20 == 0:
            sys.stdout.write(
                "[Epoch %d/%d] [Batch %d/%d] [G loss: %f]"
                % (epoch, pre_epochs, i, len(dataloader), G_loss.item())
            )
        
            fake_imgs_ = make_grid(G(lr_imgs), nrow=1, normalize=True)
            save_image(fake_imgs_, "images/train/pre%d.png" % (epoch+200), normalize=False)
            fake_imgs_ = make_grid(lr_imgs, nrow=1, normalize=True)
            save_image(fake_imgs_, "images/train/pre%d_lr.png" % (epoch+200), normalize=False)
    
        G.eval()
        # validation
        with torch.no_grad():
            valing_results = {'mse': 0, 'psnr': 0, 'batch_sizes': 0}
            val_images = []
            val_dataset = ValDataset(4)
            val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
            for i, imgs in enumerate(val_dataloader):
                batch_size = 1
                valing_results['batch_sizes'] += batch_size
                val_lr = imgs[0].to(device)
                val_hr = imgs[1].to(device)
                val_lr_upscale = imgs[2].to(device)
                val_sr = G(val_lr)
                batch_mse = ((val_sr - val_hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
                print("mse: {}, psnr: {}".format(valing_results['mse'], valing_results['psnr']))
                if epoch % 20 == 0:       
                    fake_imgs_ = make_grid(val_sr, nrow=1, normalize=True)
                    real_imgs_ = make_grid(val_lr_upscale, nrow=1, normalize=True)
                    img_grid = torch.cat((real_imgs_, fake_imgs_), -1)
                    save_image(img_grid, "images/val/pre%d-%d.png" % (epoch+200, i), normalize=False)
        # save model
        if epoch % 100 == 99:
            torch.save(G.state_dict(), "pretrained_model/pretrained_G.pth")

#G.load_state_dict(torch.load("saved_models/G_40.pth"))
#D.load_state_dict(torch.load("saved_models/discriminator_40.pth"))
# Training
# *The number of batches is equal to number of iterations for one epoch.

total_epochs = 200
start = time.time()
for epoch in range(total_epochs):
    print('Epoch {}/{}'.format(epoch, total_epochs - 1))
    print('-' * 10)
    G.train()
    for i, imgs in enumerate(dataloader):
        # Configure model input
        lr_imgs = imgs[0].to(device)
        hr_imgs = imgs[1].to(device)
        
        ### train Generator
        G_optimizer.zero_grad()

        # Generate a high resolution image from low resolution input
        fake_imgs = G(lr_imgs)
    
        # 각 fake image의 점수(tanh)
        fake_out = D(G(lr_imgs))
        real_out = D(hr_imgs)
        
        # Adversarial ground truths
        valid = Variable(torch.zeros_like(real_out).to(device), requires_grad=False)
        fake = Variable(torch.ones_like(fake_out).to(device), requires_grad=False)

        # Adversarial loss
        adversarial_loss = criterion_GAN(D(G(lr_imgs)), valid)

        # Content loss
        content_loss = criterion_content(vgg_net(G(lr_imgs)), vgg_net(hr_imgs))
        #content_loss = criterion_content(G(lr_imgs), hr_imgs)

        # Total loss
        perceptual_loss = 0.006 * content_loss + 1e-3 * adversarial_loss
        # perceptual_loss = content_loss + 1e-3 * adversarial_loss

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

        # sample_interval : interval between saving image samples
        # checkpoint_interval : interval between model checkpoints
        sample_interval = 100
        checkpoint_interval = 10
        validation_interval = 100
        batches_done = epoch * len(dataloader) + i
        fake_imgs = G(lr_imgs)
        if batches_done % sample_interval == 0:
            #  Log Progress
            sys.stdout.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [adversarial loss: %f] [content loss: %f] [percep loss: %f]\n"
                % (epoch, total_epochs, i, len(dataloader), D_loss.item(), adversarial_loss.item(), content_loss.item(), perceptual_loss.item())
            )
            # Save image grid with upsampled inputs and SRGAN outputs
            lr_imgs = nn.functional.interpolate(lr_imgs, scale_factor=4)
            fake_imgs = make_grid(fake_imgs, nrow=1, normalize=True)
            lr_imgs = make_grid(lr_imgs, nrow=1, normalize=True)
            img_grid = torch.cat((lr_imgs, fake_imgs), -1)
            save_image(img_grid, "images/train/vgg54/%d.png" % batches_done, normalize=False)

        if batches_done % validation_interval == 0:
            G.eval()
            # validation
            with torch.no_grad():
                valing_results = {'mse': 0, 'psnr': 0, 'batch_sizes': 0}
                val_images = []
                val_dataset = ValDataset(4)
                val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
                for i, imgs in enumerate(val_dataloader):
                    batch_size = 1
                    valing_results['batch_sizes'] += batch_size
                    val_lr = imgs[0].to(device)
                    val_hr = imgs[1].to(device)
                    val_lr_upscale = imgs[2].to(device)
                    val_sr = G(val_lr)
                    batch_mse = ((val_sr - val_hr) ** 2).data.mean()
                    valing_results['mse'] += batch_mse * batch_size
                    valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
                    print("mse: {}, psnr: {}".format(valing_results['mse'], valing_results['psnr']))
                    
                    real_imgs_ = make_grid(val_hr, nrow=1, normalize=True)
                    fake_imgs_ = make_grid(val_sr, nrow=1, normalize=True)
                    bipol_imgs_ = make_grid(val_lr_upscale, nrow=1, normalize=True)
                    img_grid = torch.cat((real_imgs_, fake_imgs_, bipol_imgs_), -1)
                    save_image(img_grid, "images/val/vgg54/main%d-%d.png" % (epoch, i), normalize=False)

    
    print("training time: %dsec" % (time.time() - start))
    if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G.state_dict(), "saved_models/vgg54/G_%d.pth" % epoch)
        torch.save(D.state_dict(), "saved_models/vgg54/discriminator_%d.pth" % epoch)