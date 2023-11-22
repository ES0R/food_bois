# loss.py
import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        positive_distance = F.pairwise_distance(anchor, positive, 2)
        negative_distance = F.pairwise_distance(anchor, negative, 2)
        losses = F.relu(positive_distance - negative_distance + self.margin)
        return losses.mean()
