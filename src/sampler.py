import random
from collections import defaultdict
from torch.utils.data.sampler import Sampler

class PKSampler(Sampler):
    def __init__(self, dataset, p_classes, k_instances):
        self.dataset = dataset
        self.p_classes = p_classes
        self.k_instances = k_instances
        
        self.label_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            label = dataset.df.iloc[idx]['label']
            self.label_to_indices[label].append(idx)
            
        self.labels = list(self.label_to_indices.keys())
        
    def __iter__(self):
        batch_indices = []
        labels_copy = self.labels.copy()
        random.shuffle(labels_copy)
        
        for label in labels_copy[:self.p_classes]:
            indices = self.label_to_indices[label]
            replace = len(indices) < self.k_instances
            sampled_indices = random.choices(indices, k=self.k_instances) if replace else random.sample(indices, self.k_instances)
            batch_indices.extend(sampled_indices)
            
        return iter(batch_indices)
    
    def __len__(self):
        return self.p_classes * self.k_instances