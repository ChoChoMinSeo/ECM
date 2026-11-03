import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
    
class ModeEstimator(nn.Module):
    def __init__(self, in_channels=192, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3,1,1)
        self.conv2 = nn.Conv2d(in_channels, num_classes, 3,1,1)
        self.num_class = num_classes
        
    def forward(self, x, init_prob = None):
        logits = self.conv2(F.relu(self.conv1(x)))
        logits = rearrange(logits , 'b c h w -> (b h w) c')
        soft_prob = F.softmax(logits, dim=-1)
        
        if self.training:
            loss = self.ce_loss(soft_prob, init_prob)
            return soft_prob, loss
        else:
            return soft_prob
    
    def ce_loss(self, pred, target):
        """
        pred: (B, num_classes), softmax 확률
        target: (B, num_classes), one-hot 형태 확률
        """
        
        loss = -(target * torch.log(pred + 1e-8)).sum(dim=1).mean()
        return loss