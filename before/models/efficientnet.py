import torch
from torch import nn
import torch.nn.functional as F
import timm
#from torchvision.models import efficientnet_b5

class efficientnet_base(nn.Module):
    def __init__(
        self,
        base_model = 'efficientnet_b4',
        ouput_dim = 1000,
        drop_p = 0.1
        ):
        
        super().__init__()
        
        self.model = timm.create_model(base_model, pretrained=True, num_classes=ouput_dim, drop_rate=drop_p)
        #self.model = efficientnet_b5(pretrained=True)
        
    def forward(self,image):
        y =self.model(image)
        
        return y
    