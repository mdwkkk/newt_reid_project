import torch.nn as nn
import timm

class NewtReIDModel(nn.Module):
    # Добавили img_size=224 по умолчанию
    def __init__(self, model_name='vit_base_patch16_224', embed_dim=512, pretrained=True, img_size=256):
        super().__init__()
        
        # Передаем img_size в timm. Он сам сделает бикубическую интерполяцию
        # позиционных эмбеддингов под наш новый размер!
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0,
            img_size=img_size  # <--- ВОТ ОНА, МАГИЯ
        )
        
        in_features = self.backbone.num_features
        
        self.head = nn.Sequential(
            nn.Linear(in_features, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim)
        )
        self._init_head()
        
    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.head(features)
        return embeddings