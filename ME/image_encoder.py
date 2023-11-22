# image_encoder.py
import torch
import torchvision.models as models
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 768)  # Assuming you want a 256-dim feature vector
        
    def forward(self, x):
        return self.model(x)
