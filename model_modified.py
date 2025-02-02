import torch
import torch.nn as nn
from torchvision.models import resnet50
from core_qnn import *
from ReLu import Relu
from core_qnn.quaternion_layers import QuaternionConv
from BatchNormalization import QuaternionBatchNorm2d
from InstanceNormalization import QuaternionInstanceNorm2d
from core_qnn.quaternion_layers import QuaternionLinearAutograd
import numpy


print("Importing model_modified.py")
# Define the FaceNetModel first
class FaceNetModel(nn.Module):
    def __init__(self, pretrained=False):
        super(FaceNetModel, self).__init__()
        self.model = resnet50(pretrained=False)
        self.model.conv1 = QuaternionConv(
            in_channels=4,          # e.g. RGB + depth
            out_channels=64,        # 16 quaternion channels = 64 real channels
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.model.bn1 = QuaternionBatchNorm2d(64)
        self.model.relu = nn.ReLU()
        # Note: review the ordering of layer replacements!
        self.model.layer1 = self._replace_resnet_layer(self.model.layer1, 64, 64)      # Outputs 64*4 = 256 channels
        self.model.layer2 = self._replace_resnet_layer(self.model.layer2, 256, 128)     # Outputs 128*4 = 512 channels
        self.model.layer3 = self._replace_resnet_layer(self.model.layer3, 512, 256)     # Outputs 256*4 = 1024 channels
        self.model.layer4 = self._replace_resnet_layer(self.model.layer4, 1024, 512) 
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model.fc = QuaternionLinearAutograd(2048, 128)
        self.classifier = QuaternionLinearAutograd(128, 500)

    def _replace_resnet_layer(self, layer, in_channels, out_channels):
        expansion = 4  # For ResNet50 Bottleneck blocks
        for i in range(len(layer)):
            block = layer[i]
            # Replace conv1: use a 1x1 convolution to reduce channels
            block.conv1 = QuaternionConv(in_channels, out_channels, kernel_size=1, stride=block.conv1.stride, bias=False)
            block.bn1 = QuaternionBatchNorm2d(out_channels)
            
            # Replace conv2: 3x3 convolution
            block.conv2 = QuaternionConv(out_channels, out_channels, kernel_size=3, stride=block.conv2.stride, padding=1, bias=False)
            block.bn2 = QuaternionBatchNorm2d(out_channels)
            
            # Replace conv3: 1x1 convolution to expand channels
            block.conv3 = QuaternionConv(out_channels, out_channels * expansion, kernel_size=1, stride=1, bias=False)
            block.bn3 = QuaternionBatchNorm2d(out_channels * expansion)
            
            # Replace downsample if it exists so that it outputs out_channels * expansion channels
            if block.downsample:
                block.downsample[0] = QuaternionConv(in_channels, out_channels * expansion, kernel_size=1, stride=block.downsample[0].stride, bias=False)
                block.downsample[1] = QuaternionBatchNorm2d(out_channels * expansion)
            in_channels = out_channels * expansion  # update for subsequent blocks in this layer
        return layer


    def quaternion_l2_norm(self, x):
        real, i, j, k = torch.chunk(x, 4, dim=1)
        norm = torch.sqrt(real**2 + i**2 + j**2 + k**2 + 1e-10)
        return torch.cat([real/norm, i/norm, j/norm, k/norm], dim=1)
    
    def freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze_fc(self):
        for param in self.model.fc.parameters():
            param.requires_grad = False

    def unfreeze_fc(self):
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def freeze_only(self, freeze):
        for name, child in self.model.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    def unfreeze_only(self, unfreeze):
        for name, child in self.model.named_children():
            if name in unfreeze:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.model.conv1(x)
        print("shape of x before batch normalization", x.shape)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        print("shape of x before layers", x.shape)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        return self.quaternion_l2_norm(x)

# Then define helper functions that use FaceNetModel
model_urls = {
    'acc_920': 'https://github.com/khrlimam/facenet/releases/download/acc-0.920/model920-6be7e3e9.pth',
    'acc_921': 'https://github.com/khrlimam/facenet/releases/download/acc-0.92135/model921-af60fb4f.pth'
}

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

def load_state(arch, progress=True):
    return load_state_dict_from_url(model_urls.get(arch), progress=progress)

def model_920(pretrained=True, progress=True):
    model = FaceNetModel()
    if pretrained:
        state = load_state('acc_920', progress)
        model.load_state_dict(state['state_dict'])
    return model

def model_921(pretrained=True, progress=True):
    model = FaceNetModel()
    if pretrained:
        state = load_state('acc_921', progress)
        model.load_state_dict(state['state_dict'])
    return model

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
print("FaceNetModel defined")
