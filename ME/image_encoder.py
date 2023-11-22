import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class ImageEncoder(nn.Module):
    def __init__(self, output_features=768):
        super(ImageEncoder, self).__init__()
        # Load pre-trained MobileNetV2
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
        
        # Add custom layers on top of MobileNetV2
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1280, output_features)  # MobileNetV2 last channel is 1280

    def forward(self, x):
        x = self.model(x)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
