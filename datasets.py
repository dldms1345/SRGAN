# hr img /home/dataset/DIV2K/DIV2K_train_HR
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np

def get_crop_size(crop_size, scale_factor):
    return crop_size - (crop_size % scale_factor)

class ImageDataset(Dataset):
    def __init__(self, crop_size, scale_factor, hr_root='/home/dataset/DIV2K/DIV2K_train_HR'):
        super(ImageDataset, self).__init__()
        self.hr_root = hr_root
        self.scale_factor = scale_factor
        imgs = sorted(os.listdir(self.hr_root))
        self.filenames = [os.path.join(self.hr_root, x) for x in imgs]
        self.crop_size = get_crop_size(crop_size, scale_factor)

        self.hr_transform = transforms.Compose([
            transforms.RandomCrop(self.crop_size),
            transforms.ToTensor()])

        self.lr_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.crop_size // self.scale_factor, Image.BICUBIC),
            transforms.ToTensor()])

    def __getitem__(self, idx):
        img = Image.open(self.filenames[idx])
        hr_img = self.hr_transform(img)
        lr_img = self.lr_transform(hr_img)
        return lr_img, hr_img
    
    def __len__(self):
        return len(self.filenames)


class ValDataset(Dataset):
    def __init__(self, upscale_factor, dataset_dir='/home/dataset/DIV2K/DIV2K_valid_HR'):
        super(ValDataset, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = get_crop_size(min(w, h), self.upscale_factor)
        lr_scale = transforms.Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = transforms.Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = transforms.CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        sr_image = hr_scale(lr_image)
        return transforms.ToTensor()(lr_image), transforms.ToTensor()(hr_image), transforms.ToTensor()(sr_image)

    def __len__(self):
        return len(self.image_filenames)
