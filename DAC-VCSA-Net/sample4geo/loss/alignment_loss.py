import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentLoss(nn.Module):
    def __init__(self, loss_function, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.loss_function = loss_function
        self.device = device
        self.use_infonce = False

    def mse_loss(self, pred, target):
        batch_size = pred.size(0)
        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        return 1 - (pred_norm * target_norm).sum() / batch_size

    def forward(self, image_features1, image_features2, logit_scale):
        if not self.use_infonce:
            batch_size, channels, num_parts = image_features1.shape
            feat1 = image_features1.transpose(2, 1).reshape(batch_size, channels * num_parts)
            feat2 = image_features2.transpose(2, 1).reshape(batch_size, channels * num_parts)
            return self.mse_loss(feat1, feat2)

        batch_size = image_features1.shape[0]
        feat1 = image_features1.reshape(batch_size, -1)
        feat2 = image_features2.reshape(batch_size, -1)
        image_features1 = F.normalize(feat1, dim=-1)
        image_features2 = F.normalize(feat2, dim=-1)

        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        logits_per_image2 = logits_per_image1.T
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        return (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels)) / 2
