import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import uuid



class PreProcessor:
    
    def __init__(self,isTrainData=True):
        self.isTrainData = True
    
    def execute(self,isTrainData=True,dirPath='Dataset/data'):
        if not os.path.exists(dirPath):
            raise ValueError(f"Dataset Directory '{dirPath}' not found.")
        path = 'train' if isTrainData else 'test'
        
        print("====== Pre-processing data from {} =======".format('/'.join([dirPath,path])))
        self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        self.dataset = ImageFolder('/'.join([dirPath,path]) , transform=self.transform)

        self.dataloader = torch.utils.data.DataLoader(self.dataset)

        
    def save(self, saveDir="Dataset/data"):
        folderName = "PreProcessedData"
        self.saveDataDir = "/".join([saveDir,folderName,"train"]) if self.isTrainData else "/".join([saveDir,folderName,"test"]) 
        print("====== save data from {} =======".format(self.saveDataDir))
        
        if not os.path.exists(self.saveDataDir):
            os.makedirs(self.saveDataDir)
            os.makedirs("/".join([self.saveDataDir,"infected"]))
            os.makedirs("/".join([self.saveDataDir,"notinfected"]))
            
        for input, label in self.dataloader:
            if label[0] == 0:
                save_image(input[0], self.saveDataDir+"/notinfected/image_{}_notinfected.png".format(str(uuid.uuid4())))
            else:
                save_image(input[0], self.saveDataDir+"/infected/image_{}_infected.png".format(str(uuid.uuid4())))
                    
              




 









    
    



