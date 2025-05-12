import timm 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torchvision.transforms as transforms 
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
import torch.nn.functional as F

class DINO_HEAD(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DINO_HEAD, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fn1 = nn.Linear(in_dim, out_dim)
        self.fn2 = nn.Linear(out_dim, out_dim)  
        self.fn3 = nn.Linear(out_dim, out_dim)
        self.fn4 = nn.Linear(out_dim, out_dim)
        self.fn5 = nn.Linear(out_dim, out_dim)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(out_dim)



    def forward(self, x):
        x = self.fn1(x)
        x = self.activation(x)
        x = self.norm(x)
        x = self.fn2(x)
        x = self.activation(x)
        x = self.norm(x)
        x = self.fn3(x)
        x = self.activation(x)
        x = self.norm(x)
        x = self.fn4(x)
        x = self.activation(x)
        x = self.norm(x)
        x = self.fn5(x)
        return x    


class DINO_MODEL(nn.Module):
    def __init__(self, model_name, img_size, out_dim=1000):
        super(DINO_MODEL, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True, img_size=img_size)
        input_dim = self.model.head.in_features
        self.model.head = nn.Identity()
        for param in self.model.parameters():
            param.requires_grad = False

        self.dino_head = DINO_HEAD(input_dim, out_dim)
    
    def forward(self, x):
        features = self.model.forward_features(x)
        cls_token = features[:, 0]
        x = self.dino_head(cls_token)
        return F.normalize(x, dim=-1, p=2)  

    

        
        

