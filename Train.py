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
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

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
        model = torch.hub.load('facebookresearch/detectron2', 'mask_rcnn_R_50_FPN_3x', pretrained=True)
        model.eval()

        # Define the image
        image_path = 'Dataset/data/PreProcessedData/train/notinfected/image_0c97b15d-7246-4d73-ab96-bc720330346d_notinfected.png'
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image / 255.0).permute(2, 0, 1).float().unsqueeze(0)

        with torch.no_grad():
            outputs = model(image.unsqueeze(0))
            cyst_mask = outputs[0]['instances'].pred_masks[0].numpy()

        # Extract cyst from ultrasound image
        cyst = image.squeeze().detach().numpy() * cyst_mask

        plt.subplot(1, 2, 1)
        plt.imshow(image.squeeze().detach().numpy(), cmap='gray')
        plt.title('Ultrasound Image')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(cyst, cmap='gray')
        plt.title('Extracted Cyst')
        plt.axis('off')
        plt.show()
        # for i in range(0,epochs):
        #     for images, targets, image_ids in self.dataloader:
        #         images = list(image.to(self.device) for image in images)
        #         targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        #         loss_dict = model(images, targets)

        #         losses = sum(loss for loss in loss_dict.values())
        #         loss_value = losses.item()

                
        #         optimizer.zero_grad()
        #         losses.backward()
        #         optimizer.step()
    
    
    def trial(self):

        image_path = 'Dataset/data/PreProcessedData/train/notinfected/image_0c97b15d-7246-4d73-ab96-bc720330346d_notinfected.png'
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image / 255.0).permute(2, 0, 1).float().unsqueeze(0)

        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)
        outputs = predictor(image)

        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow(out.get_image()[:, :, ::-1])
        cv2.imwrite(out.get_image(),"Dataset/output.jpg")