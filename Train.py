import os
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import torchvision.models as models
import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Trainer:
    
    def __init__(self):
        print("+++++++++++++ Training initiated +++++++++++++\n")
        print("Started at= {}".format(datetime.datetime.now()))
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # self.loadData()
        # clear
        # self.getModel()
            
    def getModel(self, num_classes = 2, 
                               feature_extraction = True):
    
        self.model = models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.num_classes = len(self.dataset.classes)
        self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(self.in_features, num_classes)
        self.model.roi_heads.mask_predictor = None
        
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(self.params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)

        self.model = self.model.to(self.device)
        return self.model
    
    def loadData(self, dirPath="Dataset/PreProcessedData/train"):
        print("Loading train data...")
        if not os.path.exists(dirPath):
            raise ValueError("Dataset Directory {} not found.".format(dirPath))
        else:
            self.dataset = ImageFolder(dirPath)
            self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=32, shuffle=True)
            print(len(self.dataset))
        
    
    def execute(self):
        pass
    
    
    def trial(self):

        print("======= executing trail =========")
        