import torch

class VectorDatabase:
    def __init__(self, device='cpu'):
        self.device = device
        self.embeddings = None
        self.labels = []
        
    def add(self, embeddings, labels):
        embeddings = embeddings.to(self.device)
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = torch.cat([self.embeddings, embeddings], dim=0)
        self.labels.extend(labels)
        
    def search(self, query_embed, top_k=20):
        query_embed = query_embed.to(self.device)
        # Косинусное сходство (скалярное произведение, т.к. векторы L2-нормализованы моделью)
        similarities = torch.matmul(query_embed, self.embeddings.T).squeeze(0)
        
        scores, indices = torch.topk(similarities, k=min(top_k, len(self.labels)))
        
        results = []
        for score, idx in zip(scores, indices):
            results.append({
                "label": self.labels[idx.item()],
                "score": score.item()
            })
        return results