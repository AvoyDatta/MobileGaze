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
import torchsummary
import thop
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
        self.sn = torchvision.models.squeezenet1_1(pretrained=sn_pretrained, num_classes=1000)
#         print(self.sn.state_dict()['classifier.1.weight'].size())
#         print(self.sn.state_dict()['classifier.1.bias'].size())

        updated_sn_conv = nn.Conv2d(512, squeezenet_outs, kernel_size=1)
        self.sn.classifier = nn.Sequential(
                    nn.Dropout(p=0.5),
                    updated_sn_conv,
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1,1)) #Shrinks size down to width (N, 256, 1, 1)        
        )
        self.features = self.sn
        self.features.num_classes = squeezenet_outs 
        
        self.batchnorm = nn.BatchNorm1d(squeezenet_outs)
        #Updates the classification layer of SqueezeNet
        #SN should now return a vector of scores of dim squeezenet_outs
#         print([v.size() for k, v in self.features.state_dict().items()])
#         wts, bias = torch.empty(512, squeezenet_outs), torch.empty(squeezenet_outs)
#         self.features.state_dict()['classifier.1.weight'].copy_(nn.init.kaiming_uniform_(wts))
#         self.features.state_dict()['classifier.1.bias'].copy_(nn.init.constant_(bias, 0)) 
#         print([v.size() for k, v in self.features.state_dict().items()])
        
#         print(self.features.state_dict()['classifier.1.weight'].size())
#         print(self.features.state_dict()['classifier.1.bias'].size())

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.batchnorm(x)
        return x

class FaceImageModel(nn.Module):
    
    def __init__(self):
        super(FaceImageModel, self).__init__()
        self.conv = MobileGazeImageModel() #BNed
        self.fc = nn.Sequential(
            nn.Linear(squeezenet_outs, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64)
            )

    def forward(self, x):
        x = self.conv(x) #(N, 256)
        x = self.fc(x) #(N, 64)
        return x

class FaceGridModel(nn.Module):
    # Model for the face grid pathway
    def __init__(self, gridSize = 25):
        super(FaceGridModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(gridSize * gridSize, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128)
            )
        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        
    def forward(self, x):
        x = x.view(x.size(0), 1, 25, 25) #(N, 1, 25, 25)
        x = self.conv(x) #Doesnt affect size of input
        x = x.view(x.size(0), -1) #(N, 625)
        x = self.fc(x) #(N, 128)
        return x



class MobileGaze(nn.Module):


    def __init__(self):
        super(MobileGaze, self).__init__()
        self.eyeModel = MobileGazeImageModel()
            
        self.faceModel = FaceImageModel()
        self.gridModel = FaceGridModel() #Same as iTracker
        # Joining both eyes
        self.eyesFC = nn.Sequential(
            nn.Linear(2*squeezenet_outs, 128), #(num_eyes * (act_size/eye))
            nn.LeakyReLU(inplace=True),
#             nn.BatchNorm1d(128)
            )
        # Joining everything
        self.fc = nn.Sequential(
            nn.Linear(128+64+128, 128), #[in dims = eyesFC(128) + faceImage(64) + faceGrid(128)]
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 2),
            )

    def forward(self, faces, eyesLeft, eyesRight, faceGrids, model_stats=None):
        # Eye nets
        xEyeL = self.eyeModel(eyesLeft) #(N, 256)
        xEyeR = self.eyeModel(eyesRight) #(N, 256)
        # Cat and FC
        xEyes = torch.cat((xEyeL, xEyeR), 1) #(N, 512)
        xEyes_fc = self.eyesFC(xEyes) #(N, 128)

        # Face net
        xFace = self.faceModel(faces)
        xGrid = self.gridModel(faceGrids)

        # Cat all
        x_cat = torch.cat((xEyes_fc, xFace, xGrid), 1) #(N, 128 + 64 + 128)
        x = self.fc(x_cat)
        
        if (model_stats != None):
            models = [self.eyeModel, self.eyeModel, self.eyesFC, self.faceModel, self.gridModel, self.fc]
            inputs = [eyesLeft, eyesRight, xEyes, faces, faceGrids, x_cat]
            reused = [1]
            total_flops, total_params = get_model_stats(models, inputs, reused, faces.size(0))
            model_stats['flops'] = total_flops
            model_stats['params'] = total_params
        return x

def get_model_stats(models, inputs, reused, batch_size):
        total_params, total_flops= 0, 0        
        thop_device = 'cuda' if torch.cuda.is_available() else "cpu"

        for idx, sub_model in enumerate(models):
            flops, params = thop.profile(sub_model, input_size = inputs[idx].size(), device=thop_device)
            if idx not in reused: 
                #Skip repeated params
                total_params += params
            total_flops += flops
        total_flops /= (batch_size * 1.) #Divide by batch size to get count for single frame

        return total_flops, total_params
