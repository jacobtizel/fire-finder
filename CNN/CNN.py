import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import numpy as np
import cv2
import os
#from UNet import UNet
from CNN.DatasetLoader import DatasetLoader

def loadData(imageDir, maskDir, imageSize=128, batchSize=16):
    transform = transforms.Compose([
        transforms.ToPILImage(), #Load Image as PIL image
        transforms.Resize((imageSize, imageSize)), #Resize to 128x128, no larger or gpu memory explodes
        transforms.ToTensor(),#Convert to tensor for cuda
    ])

    dataset = DatasetLoader(imageDir, maskDir, transform=transform) #Create a dataset for the dataset loader
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True) #Use Torch's dataset loader
    return dataloader

def loadDataColor(imageDir,maskDir,imageSize = 128):
        images = []
        masks = []
        for filename in os.listdir(imageDir):
            imagePath = os.path.join(imageDir, filename)
            maskName = filename.replace('.png', '_gt.png')
            maskPath = os.path.join(maskDir, maskName)

            image = cv2.imread(imagePath)
            image = cv2.resize(image, (imageSize, imageSize))

            mask = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (imageSize, imageSize))
            mask = np.expand_dims(mask, axis=-1)

            images.append(image)
            masks.append(mask)

        return np.array(images), np.array(masks)
    
    
    
# Training Loop
def trainModel(model, dataloader, epochs=50, lr=0.001, device='cuda'):
    print(device) #Check what device is being used
    model.to(device) #Send torch stuff to device
    optimizer = torch.optim.SGD(model.parameters(), lr=lr) #Use Stochastic Gradient Descent
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train() #Make sure model is in training
        epoch_loss = 0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device) #Move images and masks to device
            
            optimizer.zero_grad() #Resets optimizer state
            outputs = model(images) #Get outputs based on model's response to images

            loss = criterion(outputs, masks) #Evaluate loss based on BCELoss
            loss.backward() #Backwards Propogate
            optimizer.step() #Take optimizer step

            epoch_loss += loss.item() #Calculate epoch loss

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}')

    torch.save(model.state_dict(), 'unet_segmentation_torch.pth')
         
def visualizePredictions(model, dataloader, device='cuda'):
    model.eval()
    model.to(device)
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            fixedImages = []
            # for image in images:
            #     fixedImages.append(np.transpose(image, axes = (1,2,0)))
            # images = fixedImages
            
            outputs = model(images)
            outputs = outputs.cpu().squeeze().numpy()
            masks = masks.cpu().squeeze().numpy()
            images = images.cpu().squeeze().numpy()
            
            plt.figure(figsize=(12, 5))
            for i in range(3):
                #image = (np.transpose(images[i], axes = (1,2,0)))
                #output = (np.transpose(outputs[i], axes = (1,2,0)))
                #image = cv2.colorChange(image,cv2.COLOR_BGR2RGB)
                plt.subplot(3, 3, i * 3 + 1)
                #plt.imshow(images[i])
                plt.imshow(images[i], cmap='gray')
                plt.title('Input Image')
                plt.axis('off')
                
                plt.subplot(3, 3, i * 3 + 2)
                plt.imshow(masks[i], cmap='gray')
                plt.title('Ground Truth')
                plt.axis('off')
                
                plt.subplot(3, 3, i * 3 + 3)
                #plt.imshow(outputs[i])
                plt.imshow(outputs[i], cmap='gray')
                plt.title('Prediction')
                plt.axis('off')
            plt.show()
            break

def saveSegmentedImages(model, images, output_dir, device='cuda'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, image in enumerate(images):
            imageTensor = torch.tensor(image.transpose((2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(device)
            
            prediction = model(imageTensor).squeeze().cpu().numpy()
            prediction = (prediction > 0.5).astype(np.uint8) 
            
            # RGBMask = np.zeros_like(image)
            # RGBMask[:, :, 1] = prediction  # Green mask
            #pred_3ch = np.repeat(prediction[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
            # overlay = cv2.addWeighted(image, 0.7, RGBMask, 0.3, 0)
            overlay = cv2.bitwise_and(image.astype(np.uint8),image.astype(np.uint8), mask=prediction)
            
            
            
            cv2.imwrite(os.path.join(output_dir, f"segmented_{i}.png"), overlay)