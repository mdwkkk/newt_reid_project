import os

import torch


class VectorDatabase:
    def __init__(self, device="cpu"):
        self.device = device
        self.embeddings = None
        self.labels = []

    @property
    def is_empty(self) -> bool:
        return self.embeddings is None or len(self.labels) == 0

    def count(self) -> int:
        return len(self.labels)

    def add(self, embeddings, labels):
        embeddings = embeddings.to(self.device)
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = torch.cat([self.embeddings, embeddings], dim=0)
        self.labels.extend(labels)

    def remove_label(self, label: str) -> int:
        """Удалить все векторы с данным label. Возвращает число удалённых."""
        if self.is_empty:
            return 0
        keep_indices = [i for i, lb in enumerate(self.labels) if lb != label]
        removed = len(self.labels) - len(keep_indices)
        if removed == 0:
            return 0
        if not keep_indices:
            self.embeddings = None
            self.labels = []
            return removed
        idx_tensor = torch.tensor(keep_indices, device=self.device)
        self.embeddings = self.embeddings.index_select(0, idx_tensor)
        self.labels = [self.labels[i] for i in keep_indices]
        return removed

    def search(self, query_embed, top_k=20):
        if self.is_empty:
            return []
        query_embed = query_embed.to(self.device)
        similarities = torch.matmul(query_embed, self.embeddings.T).squeeze(0)
        k = min(top_k, len(self.labels))
        scores, indices = torch.topk(similarities, k=k)
        results = []
        for score, idx in zip(scores, indices):
            results.append(
                {
                    "label": self.labels[idx.item()],
                    "score": score.item(),
                }
            )
        return results

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if self.is_empty:
            data = {"embeddings": [], "labels": []}
        else:
            data = {
                "embeddings": [self.embeddings[i : i + 1].cpu() for i in range(len(self.labels))],
                "labels": list(self.labels),
            }
        torch.save(data, path)

    @classmethod
    def load_from_pt(cls, pt_path: str, device="cpu") -> "VectorDatabase":
        db = cls(device)
        if not os.path.exists(pt_path):
            return db
        data = torch.load(pt_path, map_location=device)
        for emb, label in zip(data.get("embeddings", []), data.get("labels", [])):
            emb_2d = emb.view(-1).unsqueeze(0).to(device)
            db.add(emb_2d, [label])
        return db
