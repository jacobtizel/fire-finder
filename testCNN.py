import time
from CNN.CNN import loadData, trainModel, visualizePredictions,saveSegmentedImages,loadDataColor
from CNN.UNet import UNet
from CNN.DatasetLoader import DatasetLoader
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split



train = True
startTime = time.time() 
# Paths to dataset
imageDir = 'BoWFireDataset\\dataset\\img\\fire'
maskDir = 'BoWFireDataset\\dataset\\gt\\fire-masks'

# Load data
#dataloader = loadData(imageDir, maskDir)
images, masks = loadDataColor(imageDir,maskDir)

masks = masks/255
#Split Model into training and validation data
trainingImages, validationImages, trainingMasks, validationMasks = train_test_split(images, masks, test_size=0.1, random_state=70)

trainingDataset = DatasetLoader(trainingImages,trainingMasks)
validationDataset = DatasetLoader(validationImages,validationMasks)

trainingDatasetLoader = DataLoader(trainingDataset,batch_size=80, shuffle=True)
validationDatasetLoader = DataLoader(validationDataset,batch_size= 20, shuffle= False)

# Create U-Net model
model = UNet(inputChannels=3, outputChannels=1)
if train:
# Train model
    trainModel(model, trainingDatasetLoader, epochs=50, lr=0.0001)
    stopTime = time.time()
    print(f'Training took {stopTime-startTime:.2f} Seconds.\n')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('unet_segmentation_torch.pth', map_location=device))

# Visualize predictions
#visualizePredictions(model, dataloader)
saveSegmentedImages(model,validationImages,'CNN\\CNNImages')