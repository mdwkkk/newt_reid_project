import torch.nn as nn
import torch.nn.functional as F
import timm

class NewtReIDModel(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', embed_dim=512, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        in_features = self.backbone.num_features
        
        # Обновленная голова с Batch Normalization
        self.head = nn.Sequential(
            nn.Linear(in_features, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.head(features)
        return F.normalize(embeddings, p=2, dim=1)