''' run.py
Executes an inference for images (stored either within the "data/" directory or 
given via command-line argument), yielding a collection of named .bin files 
representing inputs/outputs to convolution layers. 

Usage: 
  python3 run.py <model_type> (<index>) (--zero_type=<zero_type>) 
    (--epsilon=<float>)

Parameters: 
  <model_type>: 
    -resnet18
    -vgg16 
    -alexnet
    -yolov5l

  <index>: the conv operator index in the model to start computation at.
    If not provided, equals 1. To skip the first <n> convolutional operations, 
    set <index> to <n+1>.

  <zero_type>: 
    none: no values are zeroed
    input: the input tensors to each conv2D operation has values zeroed
    weight: the weight tensors to each conv2D operation has values zeroed
    both: both the input and weight tensors to each conv2D operation has 
      values zeroed

  <epsilon>: the threshold that determines if a value is zeroed or not. If 
    zero_type is not "none", then the input and/or weight tensor is scanned 
    before being passed to the conv2D operator. For-each value in the tensor, 
    if the value is less than or equal to epsilon, the value is replaced with 
    zero.
'''

import sys 
import math
import numpy as np
import os
import subprocess
import torch
import cv2
import torch.nn as nn
from imagenet_stubs.imagenet_2012_labels import label_to_name
import glob
from glob import glob
from torch import nn
from torch.nn import functional as F
from torchvision import models as M

from PIL import Image

# For convenience, these are global. Precondition: This program cannot run more
# than one inference. One call of conv2d() will increment CURRENT_INDEX.
MODEL_NAME = '' 
START_INDEX = 1 
CURRENT_INDEX = 1 

# If ZERO_TYPE isn't "none", then if a value in input and/or weight tensors are
# less than or equal to EPSILON, they get set to zero.
ZERO_TYPE = 'none' 
EPSILON = 0


class resnet18:
    def __init__(self): 
        self.model = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1).eval()
        
    
    # Returns (x, control) 
    def __call__(self, dataset):
        resnet18 = self.model 
        x = dataset 
        control = resnet18(x.clone()) 
    
        ## Khoa
        resultFilePath = "cL0_1_1_"
        x = conv2d(x, resultFilePath, resnet18.conv1)
        x = resnet18.bn1(x)
        x = nn.ReLU(inplace=True)(x)
        x = resnet18.maxpool(x)

        # LAYER 1

        identity = x
        resultFilePath = "cL1_0_1_"
        x = conv2d(identity, resultFilePath, resnet18.layer1[0].conv1)
        x = resnet18.get_submodule("layer1.0.bn1")(x)
        x = nn.ReLU(inplace=True)(x)

        resultFilePath = "cL1_0_2_"
        x = conv2d(x, resultFilePath, resnet18.layer1[0].conv2)
        x = resnet18.get_submodule("layer1.0.bn2")(x)

        x += identity
        x = nn.ReLU(inplace=True)(x)

        identity = x

        resultFilePath = "cL1_1_1_"
        x = conv2d(x, resultFilePath, resnet18.layer1[1].conv1)
        x = resnet18.get_submodule("layer1.1.bn1")(x)
        x = nn.ReLU(inplace=True)(x)

        resultFilePath = "cL1_1_2_"
        x = conv2d(x, resultFilePath, resnet18.layer1[1].conv2)
        x = resnet18.get_submodule("layer1.1.bn2")(x)

        x += identity
        x = nn.ReLU(inplace=True)(x)

        # LAYER 2

        identity = x

        resultFilePath = "cL2_0_1_"
        x = conv2d(x, resultFilePath, resnet18.layer2[0].conv1)
        x = resnet18.get_submodule("layer2.0.bn1")(x)
        x = nn.ReLU(inplace=True)(x)

        resultFilePath = "cL2_0_2_"
        x = conv2d(x, resultFilePath, resnet18.layer2[0].conv2)
        x = resnet18.get_submodule("layer2.0.bn2")(x)

        resultFilePath = "cL2_0_down_"
        identity = conv2d(identity, resultFilePath, resnet18.get_submodule("layer2.0.downsample.0"))
        identity = resnet18.get_submodule("layer2.0.downsample.1")(identity)

        x += identity
        x = nn.ReLU(inplace=True)(x)

        identity = x

        resultFilePath = "cL2_1_1_"
        x = conv2d(x, resultFilePath, resnet18.layer2[1].conv1)
        x = resnet18.get_submodule("layer2.1.bn1")(x)
        x = nn.ReLU(inplace=True)(x)

        resultFilePath = "cL2_1_2_"
        x = conv2d(x, resultFilePath, resnet18.layer2[1].conv2)
        x = resnet18.get_submodule("layer2.1.bn2")(x)

        x += identity
        x = nn.ReLU(inplace=True)(x)

        # LAYER 3

        identity = x

        resultFilePath = "cL3_0_1_"
        x = conv2d(x, resultFilePath, resnet18.layer3[0].conv1)
        x = resnet18.get_submodule("layer3.0.bn1")(x)
        x = nn.ReLU(inplace=True)(x)

        resultFilePath = "cL3_0_2_"
        x = conv2d(x, resultFilePath, resnet18.layer3[0].conv2)
        x = resnet18.get_submodule("layer3.0.bn2")(x)

        resultFilePath = "cL3_0_down_"
        identity = conv2d(identity, resultFilePath, resnet18.get_submodule("layer3.0.downsample.0"))
        identity = resnet18.get_submodule("layer3.0.downsample.1")(identity)

        x += identity
        x = nn.ReLU(inplace=True)(x)

        identity = x

        resultFilePath = "cL3_1_1_"
        x = conv2d(x, resultFilePath, resnet18.layer3[1].conv1)
        x = resnet18.get_submodule("layer3.1.bn1")(x)
        x = nn.ReLU(inplace=True)(x)

        resultFilePath = "cL3_1_2_"
        x = conv2d(x, resultFilePath, resnet18.layer3[1].conv2)
        x = resnet18.get_submodule("layer3.1.bn2")(x)

        x += identity
        x = nn.ReLU(inplace=True)(x)

        # LAYER 4

        identity = x

        resultFilePath = "cL4_0_1_"
        x = conv2d(x, resultFilePath, resnet18.layer4[0].conv1)
        x = resnet18.get_submodule("layer4.0.bn1")(x)
        x = nn.ReLU(inplace=True)(x)

        resultFilePath = "cL4_0_2_"
        x = conv2d(x, resultFilePath, resnet18.layer4[0].conv2)
        x = resnet18.get_submodule("layer4.0.bn2")(x)

        resultFilePath = "cL4_0_down_"
        identity = conv2d(identity, resultFilePath, resnet18.get_submodule("layer4.0.downsample.0"))
        identity = resnet18.get_submodule("layer4.0.downsample.1")(identity)

        x += identity
        x = nn.ReLU(inplace=True)(x)

        identity = x

        resultFilePath = "cL4_1_1_"
        x = conv2d(x, resultFilePath, resnet18.layer4[1].conv1)
        x = resnet18.get_submodule("layer4.1.bn1")(x)
        x = nn.ReLU(inplace=True)(x)

        resultFilePath = "cL4_1_2_"
        x = conv2d(x, resultFilePath, resnet18.layer4[1].conv2)
        x = resnet18.get_submodule("layer4.1.bn2")(x)

        x += identity
        x = nn.ReLU(inplace=True)(x)

        # CATEGORIZE

        x = resnet18.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = resnet18.fc(x)
        
        return x, control 

 
    def preprocess(self, image):
        return M.ResNet18_Weights.IMAGENET1K_V1.transforms()(image) 
        
        
    def __repr__(self): 
        return repr(self.model)


    def __len__(self):
        return 20 

class LeNet5():
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    #return x
    def __call__(self, data): 
        x = self.conv1(data)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        x = x.view(-1, self.num_flat_features(x))

        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        return x, x


    def num_flat_features(self, x):
        '''
        Get the number of features in a batch of tensors `x`.
        '''
        size = x.size()[1:]
        return np.prod(size)

    #preprocess input file to fit Lenet-5
    def preprocess(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        img = img / 255
        img=cv2.resize(img,(32,32), interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(img).unsqueeze(1)
        return img 

    def __len__(self): 
        return 8 

class vgg16:
    def __init__(self): 
        self.model = M.vgg16(weights=M.VGG16_Weights.IMAGENET1K_V1).eval()
        
    
    # Returns (x, control) 
    def __call__(self, dataset):
        vgg16 = self.model 
        x = dataset 
        control = vgg16(x.clone())
        
        # PART 1
        
        resultFilePath = "cL1_1_"
        x = conv2d(x, resultFilePath, vgg16.features[0])  # Conv2D-1
        x = vgg16.features[1](x)                          # ReLU-2
    
        resultFilePath = "cL1_2_"
        x = conv2d(x, resultFilePath, vgg16.features[2])  # Conv2D-3
        x = vgg16.features[3](x)                          # ReLU-4
        x = vgg16.features[4](x)                          # MaxPool2D-5
        
        # PART 2 
        
        resultFilePath = "cL2_1_"
        x = conv2d(x, resultFilePath, vgg16.features[5])  # Conv2D-6
        x = vgg16.features[6](x)                          # ReLU-7
        
        resultFilePath = "cL2_2_"
        x = conv2d(x, resultFilePath, vgg16.features[7])  # Conv2D-8
        x = vgg16.features[8](x)                          # ReLU-9
        x = vgg16.features[9](x)                          # MaxPool2D-10
        
        # PART 3 
        
        resultFilePath = "cL3_1_"
        x = conv2d(x, resultFilePath, vgg16.features[10]) # Conv2D-11
        x = vgg16.features[11](x)                         # ReLU-12
        
        resultFilePath = "cL3_2_"
        x = conv2d(x, resultFilePath, vgg16.features[12]) # Conv2D-13
        x = vgg16.features[13](x)                         # ReLU-14
        
        resultFilePath = "cL3_3_"
        x = conv2d(x, resultFilePath, vgg16.features[14]) # Conv2D-15
        x = vgg16.features[15](x)                         # ReLU-16
        x = vgg16.features[16](x)                         # MaxPool2D-17
        
        # PART 4
        
        resultFilePath = "cL4_1_"
        x = conv2d(x, resultFilePath, vgg16.features[17]) # Conv2D-18
        x = vgg16.features[18](x)                         # ReLU-19
        
        resultFilePath = "cL4_2_"
        x = conv2d(x, resultFilePath, vgg16.features[19]) # Conv2D-20
        x = vgg16.features[20](x)                         # ReLU-21
        
        resultFilePath = "cL4_3_"
        x = conv2d(x, resultFilePath, vgg16.features[21]) # Conv2D-22
        x = vgg16.features[22](x)                         # ReLU-23
        x = vgg16.features[23](x)                         # MaxPool2D-24
        
        # PART 5

        resultFilePath = "cL5_1_"
        x = conv2d(x, resultFilePath, vgg16.features[24]) # Conv2D-25
        x = vgg16.features[25](x)                         # ReLU-26
        
        resultFilePath = "cL5_2_"
        x = conv2d(x, resultFilePath, vgg16.features[26]) # Conv2D-27
        x = vgg16.features[27](x)                         # ReLU-28
        
        resultFilePath = "cL5_3_"
        x = conv2d(x, resultFilePath, vgg16.features[28]) # Conv2D-29
        x = vgg16.features[29](x)                         # ReLU-30
        x = vgg16.features[30](x)                         # MaxPool2D-31

        # CATEGORIZE
        
        x = vgg16.avgpool(x) 
        x = torch.flatten(x, start_dim=1) 
        x = vgg16.classifier(x) # linear, relu, linear, relu, linear
        
        return x, control 

 
    def preprocess(self, image):
        return M.VGG16_Weights.IMAGENET1K_V1.transforms()(image) 
        
        
    def __repr__(self): 
        return repr(self.model) 


    def __len__(self): 
        return 13 
        
        
class alexnet:
    def __init__(self): 
        self.model = M.alexnet(weights=M.AlexNet_Weights.IMAGENET1K_V1).eval()
        
    
    # Returns (x, control) 
    def __call__(self, dataset):
        alexnet = self.model 
        x = dataset 
        control = alexnet(x.clone())
        
        # PART 1
        
        resultFilePath = "cL1_"
        x = conv2d(x, resultFilePath, alexnet.features[0])  # Conv2D-1
        x = alexnet.features[1](x)                          # ReLU-2
        x = alexnet.features[2](x)                          # MaxPool2D-3 
    
        # PART 2
        
        resultFilePath = "cL2_"
        x = conv2d(x, resultFilePath, alexnet.features[3])  # Conv2D-4
        x = alexnet.features[4](x)                          # ReLU-5
        x = alexnet.features[5](x)                          # MaxPool2D-6 
        
        # PART 3
        
        resultFilePath = "cL3_"
        x = conv2d(x, resultFilePath, alexnet.features[6])  # Conv2D-7
        x = alexnet.features[7](x)                          # ReLU-8
        
        # PART 4
        
        resultFilePath = "cL4_"
        x = conv2d(x, resultFilePath, alexnet.features[8])  # Conv2D-9
        x = alexnet.features[9](x)                          # ReLU-10

        # PART 5
        
        resultFilePath = "cL5_"
        x = conv2d(x, resultFilePath, alexnet.features[10]) # Conv2D-11
        x = alexnet.features[11](x)                         # ReLU-12
        x = alexnet.features[12](x)                         # MaxPool2D-13

        # CATEGORIZE
        
        x = alexnet.avgpool(x) 
        x = torch.flatten(x, start_dim=1) 
        x = alexnet.classifier(x) # linear, relu, linear, relu, linear
        
        return x, control 

 
    def preprocess(self, image):
        return M.AlexNet_Weights.IMAGENET1K_V1.transforms()(image) 
        
        
    def __repr__(self): 
        return repr(self.model)


    def __len__(self): 
        return 5 
        
class Google_Net:
    def __init__(self):
        self.model = M.googlenet(weights=M.GoogLeNet_Weights.IMAGENET1K_V1).eval()
        
    def __call__(self,dataset):
        googlenet=self.model
        # x = googlenet.conv1(dataset)
        control = googlenet(dataset.clone())

        x=conv2d(dataset,"conv1",googlenet.conv1.conv)
        x=googlenet.conv1.bn(x)
        x=F.relu(x, inplace=True)


        x = googlenet.maxpool1(x)


        # x = googlenet.conv2(x)
        x=conv2d(x,"conv2",googlenet.conv2.conv)
        x=googlenet.conv2.bn(x)
        x=F.relu(x, inplace=True)



        # x = googlenet.conv3(x)
        x=conv2d(x,"conv3",googlenet.conv3.conv)
        x=googlenet.conv3.bn(x)
        x=F.relu(x, inplace=True)




        x = googlenet.maxpool2(x)


        # x = googlenet.inception3a(x)
        #branch1
        x1=conv2d(x,"inception3a_branch1_conv",googlenet.inception3a.branch1.conv)
        x1=googlenet.inception3a.branch1.bn(x1)
        x1=F.relu(x1, inplace=True)

        #branch2
        x2=conv2d(x,"inception3a_branch2_conv1",googlenet.inception3a.branch2[0].conv)
        x2=googlenet.inception3a.branch2[0].bn(x2)
        x2=F.relu(x2, inplace=True)

        x2=conv2d(x2,"inception3a_branch2_conv2",googlenet.inception3a.branch2[1].conv)
        x2=googlenet.inception3a.branch2[1].bn(x2)
        x2=F.relu(x2, inplace=True)

        #branch3
        x3=conv2d(x,"inception3a_branch3_conv1",googlenet.inception3a.branch3[0].conv)
        x3=googlenet.inception3a.branch3[0].bn(x3)
        x3=F.relu(x3, inplace=True)

        x3=conv2d(x3,"inception3a_branch3_conv2",googlenet.inception3a.branch3[1].conv)
        x3=googlenet.inception3a.branch3[1].bn(x3)
        x3=F.relu(x3, inplace=True)

        #branch4
        x4=googlenet.inception3a.branch4[0](x)
        x4=conv2d(x4,"inception3a_branch4_conv",googlenet.inception3a.branch4[1].conv)
        x4=googlenet.inception3a.branch4[1].bn(x4)
        x4=F.relu(x4, inplace=True)

        x=torch.cat((x1,x2,x3,x4),dim=1)

        # x = googlenet.inception3b(x)
        x1=conv2d(x,"inception3b_branch1_conv",googlenet.inception3b.branch1.conv)
        x1=googlenet.inception3b.branch1.bn(x1)
        x1=F.relu(x1, inplace=True)

        #branch2
        x2=conv2d(x,"inception3b_branch2_conv1",googlenet.inception3b.branch2[0].conv)
        x2=googlenet.inception3b.branch2[0].bn(x2)
        x2=F.relu(x2, inplace=True)

        x2=conv2d(x2,"inception3b_branch2_conv2",googlenet.inception3b.branch2[1].conv)
        x2=googlenet.inception3b.branch2[1].bn(x2)
        x2=F.relu(x2, inplace=True)

        #branch3
        x3=conv2d(x,"inception3b_branch3_conv1",googlenet.inception3b.branch3[0].conv)
        x3=googlenet.inception3b.branch3[0].bn(x3)
        x3=F.relu(x3, inplace=True)

        x3=conv2d(x3,"inception3b_branch3_conv2",googlenet.inception3b.branch3[1].conv)
        x3=googlenet.inception3b.branch3[1].bn(x3)
        x3=F.relu(x3, inplace=True)

        #branch4
        x4=googlenet.inception3b.branch4[0](x)
        x4=conv2d(x4,"inception3b_branch4_conv",googlenet.inception3b.branch4[1].conv)
        x4=googlenet.inception3b.branch4[1].bn(x4)
        x4=F.relu(x4, inplace=True)

        x=torch.cat((x1,x2,x3,x4),dim=1)

        x = googlenet.maxpool3(x)

        # x = googlenet.inception4a(x)
        x1=conv2d(x,"inception4a_branch1_conv",googlenet.inception4a.branch1.conv)
        x1=googlenet.inception4a.branch1.bn(x1)
        x1=F.relu(x1, inplace=True)

        #branch2
        x2=conv2d(x,"inception4a_branch2_conv1",googlenet.inception4a.branch2[0].conv)
        x2=googlenet.inception4a.branch2[0].bn(x2)
        x2=F.relu(x2, inplace=True)

        x2=conv2d(x2,"inception4a_branch2_conv2",googlenet.inception4a.branch2[1].conv)
        x2=googlenet.inception4a.branch2[1].bn(x2)
        x2=F.relu(x2, inplace=True)

        #branch3
        x3=conv2d(x,"inception4a_branch3_conv1",googlenet.inception4a.branch3[0].conv)
        x3=googlenet.inception4a.branch3[0].bn(x3)
        x3=F.relu(x3, inplace=True)

        x3=conv2d(x3,"inception4a_branch3_conv2",googlenet.inception4a.branch3[1].conv)
        x3=googlenet.inception4a.branch3[1].bn(x3)
        x3=F.relu(x3, inplace=True)

        #branch4
        x4=googlenet.inception4a.branch4[0](x)
        x4=conv2d(x4,"inception4a_branch4_conv",googlenet.inception4a.branch4[1].conv)
        x4=googlenet.inception4a.branch4[1].bn(x4)
        x4=F.relu(x4, inplace=True)
        
        x=torch.cat((x1,x2,x3,x4),dim=1)
        
        # x = googlenet.inception4b(x)
        x1=conv2d(x,"inception4b_branch1_conv",googlenet.inception4b.branch1.conv)
        x1=googlenet.inception4b.branch1.bn(x1)
        x1=F.relu(x1, inplace=True)

        #branch2
        x2=conv2d(x,"inception4b_branch2_conv1",googlenet.inception4b.branch2[0].conv)
        x2=googlenet.inception4b.branch2[0].bn(x2)
        x2=F.relu(x2, inplace=True)

        x2=conv2d(x2,"inception4b_branch2_conv2",googlenet.inception4b.branch2[1].conv)
        x2=googlenet.inception4b.branch2[1].bn(x2)
        x2=F.relu(x2, inplace=True)

        #branch3
        x3=conv2d(x,"inception4b_branch3_conv1",googlenet.inception4b.branch3[0].conv)
        x3=googlenet.inception4b.branch3[0].bn(x3)
        x3=F.relu(x3, inplace=True)

        x3=conv2d(x3,"inception4b_branch3_conv2",googlenet.inception4b.branch3[1].conv)
        x3=googlenet.inception4b.branch3[1].bn(x3)
        x3=F.relu(x3, inplace=True)

        #branch4
        x4=googlenet.inception4b.branch4[0](x)
        x4=conv2d(x4,"inception4b_branch4_conv",googlenet.inception4b.branch4[1].conv)
        x4=googlenet.inception4b.branch4[1].bn(x4)
        x4=F.relu(x4, inplace=True)

        x=torch.cat((x1,x2,x3,x4),dim=1)


        # x = googlenet.inception4c(x)
        x1=conv2d(x,"inception4c_branch1_conv",googlenet.inception4c.branch1.conv)
        x1=googlenet.inception4c.branch1.bn(x1)
        x1=F.relu(x1, inplace=True)

        #branch2
        x2=conv2d(x,"inception4c_branch2_conv1",googlenet.inception4c.branch2[0].conv)
        x2=googlenet.inception4c.branch2[0].bn(x2)
        x2=F.relu(x2, inplace=True)

        x2=conv2d(x2,"inception4c_branch2_conv2",googlenet.inception4c.branch2[1].conv)
        x2=googlenet.inception4c.branch2[1].bn(x2)
        x2=F.relu(x2, inplace=True)

        #branch3
        x3=conv2d(x,"inception4c_branch3_conv1",googlenet.inception4c.branch3[0].conv)
        x3=googlenet.inception4c.branch3[0].bn(x3)
        x3=F.relu(x3, inplace=True)

        x3=conv2d(x3,"inception4c_branch3_conv2",googlenet.inception4c.branch3[1].conv)
        x3=googlenet.inception4c.branch3[1].bn(x3)
        x3=F.relu(x3, inplace=True)

        #branch4
        x4=googlenet.inception4c.branch4[0](x)
        x4=conv2d(x4,"inception4c_branch4_conv",googlenet.inception4c.branch4[1].conv)
        x4=googlenet.inception4c.branch4[1].bn(x4)
        x4=F.relu(x4, inplace=True)

        x=torch.cat((x1,x2,x3,x4),dim=1)


        # x = googlenet.inception4d(x)
        x1=conv2d(x,"inception4d_branch1_conv",googlenet.inception4d.branch1.conv)
        x1=googlenet.inception4d.branch1.bn(x1)
        x1=F.relu(x1, inplace=True)

        #branch2
        x2=conv2d(x,"inception4d_branch2_conv1",googlenet.inception4d.branch2[0].conv)
        x2=googlenet.inception4d.branch2[0].bn(x2)
        x2=F.relu(x2, inplace=True)

        x2=conv2d(x2,"inception4d_branch2_conv2",googlenet.inception4d.branch2[1].conv)
        x2=googlenet.inception4d.branch2[1].bn(x2)
        x2=F.relu(x2, inplace=True)

        #branch3
        x3=conv2d(x,"inception4d_branch3_conv1",googlenet.inception4d.branch3[0].conv)
        x3=googlenet.inception4d.branch3[0].bn(x3)
        x3=F.relu(x3, inplace=True)

        x3=conv2d(x3,"inception4d_branch3_conv2",googlenet.inception4d.branch3[1].conv)
        x3=googlenet.inception4d.branch3[1].bn(x3)
        x3=F.relu(x3, inplace=True)

        #branch4
        x4=googlenet.inception4d.branch4[0](x)
        x4=conv2d(x4,"inception4d_branch4_conv",googlenet.inception4d.branch4[1].conv)
        x4=googlenet.inception4d.branch4[1].bn(x4)
        x4=F.relu(x4, inplace=True)
        
        x=torch.cat((x1,x2,x3,x4),dim=1)

        
        # x = googlenet.inception4e(x)
        x1=conv2d(x,"inception4e_branch1_conv",googlenet.inception4e.branch1.conv)
        x1=googlenet.inception4e.branch1.bn(x1)
        x1=F.relu(x1, inplace=True)

        #branch2
        x2=conv2d(x,"inception4e_branch2_conv1",googlenet.inception4e.branch2[0].conv)
        x2=googlenet.inception4e.branch2[0].bn(x2)
        x2=F.relu(x2, inplace=True)

        x2=conv2d(x2,"inception4e_branch2_conv2",googlenet.inception4e.branch2[1].conv)
        x2=googlenet.inception4e.branch2[1].bn(x2)
        x2=F.relu(x2, inplace=True)

        #branch3
        x3=conv2d(x,"inception4e_branch3_conv1",googlenet.inception4e.branch3[0].conv)
        x3=googlenet.inception4e.branch3[0].bn(x3)
        x3=F.relu(x3, inplace=True)

        x3=conv2d(x3,"inception4e_branch3_conv2",googlenet.inception4e.branch3[1].conv)
        x3=googlenet.inception4e.branch3[1].bn(x3)
        x3=F.relu(x3, inplace=True)

        #branch4
        x4=googlenet.inception4e.branch4[0](x)
        x4=conv2d(x4,"inception4e_branch4_conv",googlenet.inception4e.branch4[1].conv)
        x4=googlenet.inception4e.branch4[1].bn(x4)
        x4=F.relu(x4, inplace=True)

        x=torch.cat((x1,x2,x3,x4),dim=1)


        x = googlenet.maxpool4(x)


        # x = googlenet.inception5a(x)
        x1=conv2d(x,"inception5a_branch1_conv",googlenet.inception5a.branch1.conv)
        x1=googlenet.inception5a.branch1.bn(x1)
        x1=F.relu(x1, inplace=True)

        #branch2
        x2=conv2d(x,"inception5a_branch2_conv1",googlenet.inception5a.branch2[0].conv)
        x2=googlenet.inception5a.branch2[0].bn(x2)
        x2=F.relu(x2, inplace=True)

        x2=conv2d(x2,"inception5a_branch2_conv2",googlenet.inception5a.branch2[1].conv)
        x2=googlenet.inception5a.branch2[1].bn(x2)
        x2=F.relu(x2, inplace=True)

        #branch3
        x3=conv2d(x,"inception5a_branch3_conv1",googlenet.inception5a.branch3[0].conv)
        x3=googlenet.inception5a.branch3[0].bn(x3)
        x3=F.relu(x3, inplace=True)

        x3=conv2d(x3,"inception5a_branch3_conv2",googlenet.inception5a.branch3[1].conv)
        x3=googlenet.inception5a.branch3[1].bn(x3)
        x3=F.relu(x3, inplace=True)

        #branch4
        x4=googlenet.inception5a.branch4[0](x)
        x4=conv2d(x4,"inception5a_branch4_conv",googlenet.inception5a.branch4[1].conv)
        x4=googlenet.inception5a.branch4[1].bn(x4)
        x4=F.relu(x4, inplace=True)

        x=torch.cat((x1,x2,x3,x4),dim=1)


        # x = googlenet.inception5b(x)
        x1=conv2d(x,"inception5b_branch1_conv",googlenet.inception5b.branch1.conv)
        x1=googlenet.inception5b.branch1.bn(x1)
        x1=F.relu(x1, inplace=True)

        #branch2
        x2=conv2d(x,"inception5b_branch2_conv1",googlenet.inception5b.branch2[0].conv)
        x2=googlenet.inception5b.branch2[0].bn(x2)
        x2=F.relu(x2, inplace=True)

        x2=conv2d(x2,"inception5b_branch2_conv2",googlenet.inception5b.branch2[1].conv)
        x2=googlenet.inception5b.branch2[1].bn(x2)
        x2=F.relu(x2, inplace=True)

        #branch3
        x3=conv2d(x,"inception5b_branch3_conv1",googlenet.inception5b.branch3[0].conv)
        x3=googlenet.inception5b.branch3[0].bn(x3)
        x3=F.relu(x3, inplace=True)

        x3=conv2d(x3,"inception5b_branch3_conv2",googlenet.inception5b.branch3[1].conv)
        x3=googlenet.inception5b.branch3[1].bn(x3)
        x3=F.relu(x3, inplace=True)

        #branch4
        x4=googlenet.inception5b.branch4[0](x)
        x4=conv2d(x4,"inception5b_branch4_conv",googlenet.inception5b.branch4[1].conv)
        x4=googlenet.inception5b.branch4[1].bn(x4)
        x4=F.relu(x4, inplace=True)

        x=torch.cat((x1,x2,x3,x4),dim=1)


        x = googlenet.avgpool(x)
        x = torch.flatten(x, 1)
        x = googlenet.dropout(x)
        x = googlenet.fc(x)
        return x,control


    def preprocess(self, image):
        return M.GoogLeNet_Weights.IMAGENET1K_V1.transforms()(image)

    def __repr__(self): 
        return repr(self.model)

class yolo:
    def __init__(self): 
        self.model = torch.hub.load(
            'ultralytics/yolov5', 
            'custom', 
            'models/yolov5l-cls.pt'
        )
        
        # "self.model" alone is a wrapper, so we need to extract the part that's 
        # actually relevant to us. 
        self.model = self.model.model.model.to('cpu').eval()
        
    
    # Returns (x, control) 
    def __call__(self, dataset):
        yolo = self.model
        x = dataset 
        control = yolo(x.clone()) 
        
        # Layer implementations are taken from: 
        #   https://github.com/ultralytics/yolov5/blob/master/models/common.py
        
        def conv(x, layer, name): 
            x = conv2d(x, name, layer.conv) 
            x = layer.act(x) 
            return x 
            
        def bottleneck(x, layer, name): 
            y = conv(x, layer.cv1, name + '0_')
            z = conv(y, layer.cv2, name + '1_') 
            return x + z 
        
        def c3(x, layer, name): 
            a = conv(x, layer.cv1, name + '0_')
            b = a 
            for i in range(len(layer.m)):
                b = bottleneck(b, layer.m[i], f'{name}1_{i}_')
            c = conv(x, layer.cv2, name + '2_') 
            d = torch.cat((b, c), 1) 
            e = conv(d, layer.cv3, name + '3_') 
            return e
        
        def classify(x, layer, name): 
            # (There's only one classifier layer, but we can implement it like 
            # this for the sake of consistency)
            x = conv(x, layer.conv, name) 
            x = layer.pool(x)
            x = x.flatten(1) 
            x = layer.linear(x) 
            return x 
        
        layers = [conv, conv, c3, conv, c3, conv, c3, conv, c3, classify] 
        for i in range(len(layers)):
            x = layers[i](x, yolo[i], f'cL{i}_')
        
        return x, control 

 
    def preprocess(self, image):
        return M.AlexNet_Weights.IMAGENET1K_V1.transforms()(image) 
        
        
    def __repr__(self): 
        return repr(self.model) 


    def __len__(self): 
        return 60 


def save_matrix(x, fpath):
    assert x.dim() == 2

    with open(fpath, "wb") as f:
        f.write(x.size(dim=0).to_bytes(4, byteorder="little"))
        f.write(x.size(dim=1).to_bytes(4, byteorder="little"))
        f.write(x.numpy().tobytes())


def load_matrix(fpath) -> torch.Tensor:
    with open(fpath, "rb") as f:
        h = int.from_bytes(f.read(4), byteorder="little")
        w = int.from_bytes(f.read(4), byteorder="little")

        buf = f.read(h * w * np.dtype(np.float32).itemsize)

        x = torch.from_numpy(np.frombuffer(buf, dtype=np.float32).copy())
        x = x.reshape(h, w)

        return x


def pad_matrix(x) -> torch.Tensor:
    assert x.dim() == 2

    h = x.size(dim=0)
    w = x.size(dim=1)

    if h % 16 != 0:
        padding = 16 - h % 16
        x = F.pad(x, pad=(0, 0, 0, padding), mode="constant", value=0.0)

    if w % 16 != 0:
        padding = 16 - w % 16
        x = F.pad(x, pad=(0, padding, 0, 0), mode="constant", value=0.0)

    return x


def conv2d(x, layer_name, module) -> torch.Tensor:
    global MODEL_NAME 
    global START_INDEX
    global CURRENT_INDEX 

    # Either the (1) output directory of the convolution operation or 
    # (2) location of the cached convolution from a previous execution. 
    gemm_file_path = f'bin/{MODEL_NAME}/gemm/{layer_name}.bin' 

    weight = module.weight.detach()
    x = x.detach()

    out_n = x.size(dim=0)
    out_c = weight.size(dim=0)
    out_h = math.floor(((x.size(dim=2) + 2 * module.padding[0] - module.dilation[0] * (
        module.kernel_size[0] - 1) - 1) / module.stride[0]) + 1)
    out_w = math.floor(((x.size(dim=3) + 2 * module.padding[1] - module.dilation[1] * (
        module.kernel_size[1] - 1) - 1) / module.stride[1]) + 1)

    weight = weight.flatten(start_dim=1)
    weight = weight.view(weight.size(dim=0), weight.size(dim=1))

    x = nn.Unfold(kernel_size=module.kernel_size, stride=module.stride,
                  padding=module.padding, dilation=module.dilation)(x)

    slices = []
    for i in range(x.size(dim=0)):
        slices.append(x[i])
    x = torch.cat(slices, dim=1)

    h = weight.size(dim=0)
    w = x.size(dim=1)

    weight = pad_matrix(weight)
    x = pad_matrix(x)


    if START_INDEX <= CURRENT_INDEX:
        # Preprocess the matrices given the passed arguments by removing values
        # from the tensors if they're less than or equal to EPSILON
        if ZERO_TYPE in ('both', 'weight'): 
            weight = torch.where(torch.abs(weight) <= EPSILON, 0, weight)
        if ZERO_TYPE in ('both', 'input'): 
            x = torch.where(torch.abs(x) <= EPSILON, 0, x) 

        # We don't need to use cached conv outputs, so calculate the conv.         
        weight_file_path = f'bin/{MODEL_NAME}/weight/{layer_name}.bin'
        save_matrix(weight, weight_file_path) 
        
        x_file_path = f'bin/{MODEL_NAME}/x/{layer_name}.bin'
        save_matrix(x, x_file_path)
    
        output_file_path = f'output/{MODEL_NAME}/result/{layer_name}.txt'
        output_file = open(output_file_path, 'w') 
        
        error_file_path = f'output/{MODEL_NAME}/error/{layer_name}.txt'
        error_file = open(error_file_path, 'w')
        
        print(f'Starting gemm on layer "{layer_name}"')
        subprocess.run(
            [
                "./build/gemm", 
                "--w", 
                weight_file_path,
                "--x", 
                x_file_path, 
                "--o", 
                gemm_file_path
            ], 
            stdout=output_file, 
            stderr=error_file
        )
        print(f'Finished gemm on layer "{layer_name}"') 
    
    x = load_matrix(gemm_file_path) 
    
    x = torch.stack(torch.chunk(x[:h, :w], chunks=out_n, dim=1))
    x = x.view(out_n, out_c, out_h, out_w)

    for name, _ in module.named_parameters():
        if name in ["bias"]:
            bias = module.bias.detach()
            bias = bias.view(1, bias.size(dim=0), 1, 1)
            bias = bias.tile(1, 1, out_h, out_w)

            x = x.add(bias)

    CURRENT_INDEX += 1 
    return x


if __name__ == "__main__":
    
    # python3 run.py <model type> [<index>]
    #   <model type> is a string, can possibly start with a dash "-", and must 
    #     be either "resnet18"/"resnet", "vgg16"/"vgg", "alexnet", or 
    #     "yolov5l"/"yolo".
    #   <index> is an optional int; if specified, the program will start 
    #     execution on the layer specified by it. It is implicitly assumed to 
    #     be 1 if not specified.
    
    if len(sys.argv) == 1: 
        print('Error: must specify model (alexnet, resnet18, vgg16, yolov5l')
        sys.exit(1) 
    
    name = sys.argv[1].lower()
    while name.startswith('-'): name = name[1:] 
    
    if name == 'resnet18' or name == 'resnet':
        model = resnet18() 
        MODEL_NAME = 'resnet18' 
    elif name == 'vgg16' or name == 'vgg': 
        model = vgg16() 
        MODEL_NAME = 'vgg16' 
    elif name == 'alexnet':
        model = alexnet()
        MODEL_NAME = 'alexnet' 
    elif name == 'lenet5' or name == 'lenet': 
        MODEL_NAME = 'lenet5' 
        model = LeNet5() 
    elif name == 'yolov5l' or name == 'yolo': 
        model = yolo() 
        MODEL_NAME = 'yolov5l' 
    elif name == 'googlenet' : 
        model = Google_Net()
        MODEL_NAME = 'Google_Net' 
    else:
        out = 'Unrecognized model name: pass either "-resnet18", "-vgg16", ' \
              '"-alexnet", or "yolov5l" as an argument.'
        print(out) 
        sys.exit(1) 
    
    print(model) 
    
    # If the third argument is an int, it represents the starting index.

    if len(sys.argv) >= 3 and sys.argv[2].isnumeric():
        START_INDEX = int(sys.argv[2]) 
                    
        if START_INDEX < 1 or START_INDEX > len(model): 
           out = f'Error: starting index "{sys.argv[2]}" must be an integer ' \
                  'in range [1, len(model)]' 
           print(out)
           sys.exit(1)

        bin_path = 'bin/' + MODEL_NAME + '/gemm'
        if START_INDEX > 1 and not os.path.exists(bin_path): 
            out = f'Error: if using a different starting index, then bin ' \
                  f'files must exist in the "{bin_path}" directory. Run the ' \
                  f'project at least once on model {MODEL_NAME} up to layer ' \
                  f'{START_INDEX}.'
            print(out)
            sys.exit(1) 

    else: 
        START_INDEX = 1 # i.e., we start at the first layer/beginning of model
    
    # This program needs to proceed through each layer doing PyTorch functions 
    # even though they don't affect the output if START_INDEX != 1. This should
    # be changed in the future, TODO. 
    CURRENT_INDEX = 1
    
    # By default, pulls the images from the "data/" directory.
    data_path = 'dataset'
    
    # There can be other optional non-positional arguments. 
    for arg in sys.argv:
        # Remove any leading '-' 
        while len(arg) > 0 and arg[0] == '-': 
            arg = arg[1:] 

        eq = arg.find('=') 
        if eq != -1: 
            name, value = arg[:eq], arg[eq+1:]
            
            if name == 'dataset':
                if os.path.exists(value):
                    data_path = value
                else:
                    print(f'Error: data path "{data_path}" does not exist.') 
                    sys.exit(1)
            
            elif name == 'zero_type': 
                zero_types = ('none', 'input', 'weight', 'both')
                if value not in zero_types: 
                    print(f'Error: zero_type argument must be in ' \
                          f'{zero_types}, received "{value}"') 
                    sys.exit(1)
                else:
                    ZERO_TYPE = value

            elif name == 'epsilon': 
                try: 
                    EPSILON = float(value)
                except ValueError: 
                    print(f'Error: epsilon argument must be castable to a ' \
                          f'float, received "{value}"')
                    sys.exit(1) 
                         
            else:
                print(f'Error: unrecognized keyword argument "{name}".')
                sys.exit(1)

    # Images are stored within the "data" directory.
    filenames = []
    images    = []
    # folder= glob.glob('dataset/*.*')
    # for data_path in folder:
    
    if os.path.exists(data_path):
        if os.path.isdir(data_path): 
            pathnames = glob(os.path.join("dataset", "*.jpg"))
            for file in sorted(pathnames, key=os.path.basename):
                # Each model has their own unique "preprocess" method.
                image=Image.open(file).convert('RGB')
                tensor = model.preprocess(image)
                name = file[5:] # removes the 'data/' part 
                filenames.append(name) 
                images.append(tensor)
        elif data_path[-4:] == '.jpg':  
            tensor = model.preprocess(Image.open(data_path))
            name = data_path if '/' not in data_path \
                            else data_path[data_path.rindex('/')+1:]
            filenames.append(name)
            images.append(tensor)
        else:
            print('Error: the given image at "{data_path}" must be a ".jpg".')
            sys.exit(1)
    else:
        print('Error: data path "{data_path}" does not exist.') 
        sys.exit(1) 
    
    if len(images) == 0:
        print('Error: at least one .jpg image must be stored within the ' \
             f'"{data_path}" path.')
        sys.exit(1) 
        
    # Output bin files are stored within the bin directory. 
    required_dirs = [
        'bin', 
        'bin/' + MODEL_NAME, 
        'bin/' + MODEL_NAME + '/weight', 
        'bin/' + MODEL_NAME + '/x', 
        'bin/' + MODEL_NAME + '/gemm', 
        'output', 
        'output/' + MODEL_NAME, 
        'output/' + MODEL_NAME + '/result', 
        'output/' + MODEL_NAME + '/error'
    ]
    for required_dir in required_dirs: 
        if not os.path.exists(required_dir): 
            os.mkdir(required_dir) 
    
    # Models are callable: this runs an inference on the images. 
    dataset = torch.stack(images)
    if MODEL_NAME=='lenet5':
        dataset = dataset.to(dtype=torch.float32)
    x, control = model(dataset) 
    
    print(f'\nMSE: {nn.MSELoss()(x, control).item()}\n')
    
    # Added so output values represent confidence/probabilities
    x = F.softmax(x, dim=1)
    
    # Formatting 
    longest_filename = max([len(name) for name in filenames]) + 3 # padding 
    format_str = '%-' + str(longest_filename) + 's: %s'
    sum=0.0
    print("Classifications:")
    for index in range(len(images)):
        path = filenames[index] 
        argmax = torch.argmax(x[index]).item()
        label = label_to_name(argmax)
        confidence = x[index][argmax].item() 
        
        padding = '.' * (longest_filename - len(path))
        confidence_str = '[%.3f] ' % confidence 
        print(path + padding + confidence_str + label) 
        sum= sum+confidence
