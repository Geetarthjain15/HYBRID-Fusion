import torch
import torch.nn as nn
import os
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from transformers import ViTForImageClassification

class HybridBrainTumorModel(nn.Module):
    def __init__(self, vit_weights_path=None, num_classes=4):
        super().__init__()
        
        base_cnn = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.cnn_features = base_cnn.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.vit_branch = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        if vit_weights_path and os.path.exists(vit_weights_path):
            self.vit_branch.load_state_dict(torch.load(vit_weights_path, map_location='cpu'))

        self.fusion_head = nn.Sequential(
            nn.Linear(1280 + 768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 1. CNN Branch
        cnn_feat = self.cnn_features(x)
        cnn_pooled = self.avgpool(cnn_feat).view(x.size(0), -1) # (B, 1280)

        # 2. ViT Branch
        vit_out = self.vit_branch.vit(pixel_values=x)
        vit_cls = vit_out.last_hidden_state[:, 0, :] # (B, 768)

        # 3. Concatenation Fusion
        fused = torch.cat([cnn_pooled, vit_cls], dim=1) # (B, 2048)
        
        return self.fusion_head(fused)

