import os
import cv2
from torch.utils.data import Dataset, DataLoader
import torch
class DatasetLoader(Dataset):
    # def __init__(self, image_dir, mask_dir, transform=None):
    #     self.imageDir = image_dir
    #     self.maskDir = mask_dir
        # self.transform = transform
        # self.imageFilenames = sorted([
        #     f for f in os.listdir(image_dir)
        #     if f.startswith('fire') and f.endswith('.png')
        # ])
    def __init__(self,images,masks):
        self.images = images
        self.masks = masks
        
    def __len__(self):
        #return len(self.imageFilenames)
        return len(self.images)

    # def __getitem__(self, index):
    #     imageName = self.imageFilenames[index]
    #     maskName = imageName.replace('.png', '_gt.png')
        
    #     imagePath = os.path.join(self.imageDir, imageName)
    #     maskLocation = os.path.join(self.maskDir, maskName)

    #     image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    #     #image = cv2.imread(imagePath, cv2.COLOR_BGR2RGB)
    #     mask = cv2.imread(maskLocation, cv2.IMREAD_GRAYSCALE)

    #     if self.transform:
    #         image = self.transform(image)
    #         mask = self.transform(mask)
        
    #     return image, mask
    def __getitem__(self, index):
        image = self.images[index].transpose((2, 0, 1)) / 255.0
        mask = self.masks[index].transpose((2, 0, 1)) / 255.0
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
