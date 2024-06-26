import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import os
import re
import random


image_size = (128, 128)  # Taille des images voulues, un CenterCrop() sera ensuite réalisé pour obtenir cette taille
mean = 0  # Moyenne du bruit
std = 0.25  # Écart-type du bruit
kernel_size = 4  # Taille du noyau pour le downsampling
mask_threshold = 0.5  # Seuil pour le masque d'inpainting
downsample_factor = 4  # Facteur de downsampling pour le downsampling

# Fonction pour ajouter du bruit gaussien et sauvegarder les images bruitées
def add_gaussian_noise_and_save(input_dir, output_dir, image_size=(128, 128), mean=0, std=0.1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Liste des fichiers dans le répertoire source
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    # Décompter les images déjà présentes dans le répertoire de sortie
    existing_files = [f for f in os.listdir(output_dir) if re.match(r'noisy_\d+\.\w+', f)]
    existing_count = len(existing_files)
    
    transform = transforms.Compose([
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])
    
    for i, image_file in enumerate(image_files):
        if i % 100 == 0:
            print(f'Processing image {i + 1} of {len(image_files)}...')

        # Charger et transformer l'image
        image_path = os.path.join(input_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        clean_image = transform(image)
        
        # Générer le bruit gaussien
        noise = torch.randn_like(clean_image) * std + mean
        
        # Ajouter le bruit à l'image
        noisy_image = torch.clamp(clean_image + noise, 0, 1)
        
        # Convertir en PIL et sauvegarder l'image
        noisy_image_pil = transforms.ToPILImage()(noisy_image)
        noisy_image_pil.save(os.path.join(output_dir, f'noisy_{existing_count + i + 1}.png'))

        # Convertir en PIL et sauvegarder l'image clean
        clean_image_pil = transforms.ToPILImage()(clean_image)
        clean_image_pil.save(os.path.join(output_dir, f'clean_{existing_count + i + 1}.png'))
    
    print("Process completed. All noisy images are saved.")


def inpainting_and_save(input_dir, output_dir, image_size=(128, 128), mask_threshold=0.5, device='cpu'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Liste des fichiers dans le répertoire source
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    # Décompter les images déjà présentes dans le répertoire de sortie
    existing_files = [f for f in os.listdir(output_dir) if re.match(r'noisy_\d+\.\w+', f)]
    existing_count = len(existing_files)
    
    transform = transforms.Compose([
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])
    
    for i, image_file in enumerate(image_files):
        if i % 100 == 0:
            print(f'Processing image {i + 1} of {len(image_files)}...')
        
        # Charger et transformer l'image
        image_path = os.path.join(input_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        clean_image = transform(image).to(device)
        
        # Générer le masque
        h, w = clean_image.shape[1], clean_image.shape[2]
        mask = torch.rand(1, 1, h, w).to(device)
        mask = mask < mask_threshold  # Masque avec des valeurs True ou False
        
        # Appliquer le masque pour créer des zones manquantes
        inpainted_image = clean_image.clone()  # Copie de l'image originale
        inpainted_image[:, mask[0, 0, :, :]] = 0  # Masquage des parties sélectionnées
        
        # Pour l'exemple, remplissons les zones masquées par la moyenne des pixels voisins
        # Note: Cette méthode est simpliste, pour des résultats plus avancés, utilisez des techniques d'inpainting sophistiquées.
        for c in range(inpainted_image.shape[0]):
            channel = inpainted_image[c]
            masked_region = mask[0, 0, :, :]
            if masked_region.sum() > 0:
                mean_value = clean_image[c, ~masked_region].mean()
                channel[masked_region] = mean_value
            inpainted_image[c] = channel
        
        # Convertir en PIL et sauvegarder l'image inpainted
        inpainted_image_pil = transforms.ToPILImage()(inpainted_image.cpu())
        inpainted_image_pil.save(os.path.join(output_dir, f'noisy_{existing_count + i + 1}.png'))
        
        # Convertir en PIL et sauvegarder l'image clean
        clean_image_pil = transforms.ToPILImage()(clean_image.cpu())
        clean_image_pil.save(os.path.join(output_dir, f'clean_{existing_count + i + 1}.png'))
    
    print("Process completed. All inpainted images are saved.")


def downsampling_and_save(input_dir, output_dir, image_size, downsample_factor, device='cpu'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    existing_files = [f for f in os.listdir(output_dir) if re.match(r'noisy_\d+\.\w+', f)]
    existing_count = len(existing_files)
    
    transform = transforms.Compose([
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])

    for i, image_file in enumerate(image_files):
        if i % 100 == 0:
            print(f'Processing image {i + 1} of {len(image_files)}...')
        
        image_path = os.path.join(input_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        clean_image = transform(image).to(device)
        
        # Ajouter une dimension batch
        clean_image = clean_image.unsqueeze(0)
        
        # Downsampling
        downsampled_image = F.avg_pool2d(clean_image, downsample_factor, downsample_factor)
        downsampled_image = F.interpolate(downsampled_image, size=image_size, mode='nearest')
        
        # Retirer la dimension batch
        downsampled_image = downsampled_image.squeeze(0)
        clean_image = clean_image.squeeze(0)
        
        # Convertir en PIL et sauvegarder l'image downsampled
        downsampled_image_pil = transforms.ToPILImage()(downsampled_image.cpu())
        downsampled_image_pil.save(os.path.join(output_dir, f'noisy_{existing_count + i + 1}.png'))
        
        # Convertir en PIL et sauvegarder l'image clean
        clean_image_pil = transforms.ToPILImage()(clean_image.cpu())
        clean_image_pil.save(os.path.join(output_dir, f'clean_{existing_count + i + 1}.png'))

    print("Process completed. All downsampled images are saved.")


# Exemple d'utilisation
input_directory = 'E:/KaggleDownloads/imagenet-object-localization-challenge'  # Répertoire des images sources
output_directory = f"C:/Users/jugou/superresolution_images{downsample_factor}"  # Répertoire pour sauvegarder les images bruitées

downsampling_and_save(input_directory, output_directory, image_size, downsample_factor, device='cuda')