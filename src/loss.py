import torch
import torch.nn as nn

class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, embeddings, labels):
        # Вычисление матрицы попарных L2-расстояний
        dist_mat = torch.cdist(embeddings, embeddings, p=2)
        N = dist_mat.size(0)
        
        # Маски для позитивных и негативных пар
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
        
        # --- Поиск Hardest Positive ---
        # Клонируем матрицу и игнорируем негативные пары, зануляя их расстояния.
        # Максимум будет искаться только среди позитивных пар.
        dist_mat_pos = dist_mat.clone()
        dist_mat_pos[is_neg] = 0.0 
        dist_ap, _ = torch.max(dist_mat_pos, dim=1, keepdim=True)
        
        # --- Поиск Hardest Negative ---
        # Игнорируем позитивные пары, задавая им бесконечно большое расстояние.
        # Минимум будет искаться строго среди негативных пар.
        dist_mat_neg = dist_mat.clone()
        dist_mat_neg[is_pos] = float('inf') 
        dist_an, _ = torch.min(dist_mat_neg, dim=1, keepdim=True)
        
        # Целевой тензор: хотим, чтобы dist_an > dist_ap + margin
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        return loss