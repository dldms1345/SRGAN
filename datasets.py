# hr img /home/dataset/DIV2K/DIV2K_train_HR
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np

def get_crop_size(crop_size, scale_factor):
    return crop_size - (crop_size % scale_factor)

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
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
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        self.lr_transform = transforms.Compose([
            transforms.RandomCrop(self.crop_size),
            transforms.Resize(self.crop_size // self.scale_factor, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])


    def __getitem__(self, idx):
        img = Image.open(self.filenames[idx])
        hr_img = self.hr_transform(img)
        lr_img = self.lr_transform(img)
        return lr_img, hr_img
    
    def __len__(self):
        return len(self.filenames)