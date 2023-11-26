# import torch.nn as nn
# from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# class ImageEncoder(nn.Module):
#     def __init__(self, output_features=768):
#         super(ImageEncoder, self).__init__()
#         # Load pre-trained MobileNetV2 with new weights parameter
#         self.model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).features
        
#         # Add custom layers on top of MobileNetV2
#         self.pooling = nn.AdaptiveAvgPool2d((1, 1))
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(1280, output_features)  # MobileNetV2 last channel is 1280

#     def forward(self, x):
#         x = self.model(x)
#         x = self.pooling(x)
#         x = self.flatten(x)
#         x = self.fc(x)
#         return x


import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ImageEncoder(nn.Module):
    def __init__(self, output_features=768):
        super(ImageEncoder, self).__init__()
        # Load pre-trained ResNet50
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Remove the last fully connected layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        # Add custom layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048, output_features)  # ResNet50 last layer output features are 2048

    def forward(self, x):
        x = self.model(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x