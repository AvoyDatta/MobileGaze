import argparse
import os
import shutil
import time, math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import torch.utils.model_zoo as model_zoo
from torch.autograd.variable import Variable

'''
Pytorch model for the MobileGaze.

Author: Avoy Datta (avoy.datta@stanford.edu), 2019
Adapted from iTracker-Pytorch Implementation
}

'''

squeezenet_outs = 256
sn_pretrained = True

class MobileGazeImageModel(nn.Module):
    # Used for both eyes (with shared weights) and the face (with unqiue weights)
    def __init__(self):
        super(MobileGazeImageModel, self).__init__()
        self.features = torchvision.models.squeezenet1_1(pretrained=sn_pretrained)
        updated_sn_conv = nn.Conv2d(512, squeezenet_outs, kernel_size=1)
        self.features.classifier = nn.Sequential(
                    nn.Dropout(p=0.5),
                    updated_final_conv,
                    nn.ReLU(inplace=True),
                    nn.AvgPool2d(13, stride=1)
                ) #Updates the classification layer of SqueezeNet
        #SN should now return a vector of scores of dim squeezenet_outs
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
#             nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
#             nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(inplace=True),
                
#         )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

class FaceImageModel(nn.Module):
    
    def __init__(self):
        super(FaceImageModel, self).__init__()
        self.conv = MobileGazeImageModel()
        self.fc = nn.Sequential(
            nn.Linear(squeezenet_outs, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class FaceGridModel(nn.Module):
    # Model for the face grid pathway
    def __init__(self, gridSize = 25):
        super(FaceGridModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(gridSize * gridSize, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



class MobileGazeModel(nn.Module):


    def __init__(self):
        super(ITrackerModel, self).__init__()
        self.eyeModel = MobileGazeImageModel()
        self.faceModel = FaceImageModel()
        self.gridModel = FaceGridModel() #Same as iTracker
        # Joining both eyes
        self.eyesFC = nn.Sequential(
            nn.Linear(2*squeezenet_outs, 128), #(num_eyes * (act_size/eye))
            nn.ReLU(inplace=True),
            )
        # Joining everything
        self.fc = nn.Sequential(
            nn.Linear(128+64+128, 128), #(in dims = eyesFC(128) + faceImage(64) + faceGrid(128))
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            )

    def forward(self, faces, eyesLeft, eyesRight, faceGrids):
        # Eye nets
        xEyeL = self.eyeModel(eyesLeft)
        xEyeR = self.eyeModel(eyesRight)
        # Cat and FC
        xEyes = torch.cat((xEyeL, xEyeR), 1)
        xEyes = self.eyesFC(xEyes)

        # Face net
        xFace = self.faceModel(faces)
        xGrid = self.gridModel(faceGrids)

        # Cat all
        x = torch.cat((xEyes, xFace, xGrid), 1)
        x = self.fc(x)
        
        return x
