from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

class NewtDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        
        # Убедимся, что ID переведены в непрерывные индексы (0, 1, 2...)
        unique_labels = self.df['label'].unique()
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        label = self.df.iloc[idx]['label']
        
        # Загрузка RGB изображения через PIL
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.label_to_idx[label]