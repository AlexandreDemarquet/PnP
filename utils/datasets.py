import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.startswith(('clean_', 'noisy_'))]
        self.image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths) // 2

    def __getitem__(self, idx):
        clean_image_path = self.image_paths[idx * 2]
        noisy_image_path = self.image_paths[idx * 2 + 1]
        clean_image = Image.open(clean_image_path).convert('RGB')
        noisy_image = Image.open(noisy_image_path).convert('RGB')
        clean_image = self.transform(clean_image)
        noisy_image = self.transform(noisy_image)
        return noisy_image, clean_image
    
