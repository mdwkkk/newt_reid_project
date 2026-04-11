import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import NewtDataset
from src.sampler import PKSampler
from src.model import NewtReIDModel
from src.loss import BatchHardTripletLoss
import os

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Аугментации (базовые, так как данные уже выровнены)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = NewtDataset('data/train_unet.csv', transform=transform)
    sampler = PKSampler(dataset, p_classes=16, k_instances=4) # Batch size = 64
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=4)
    
    model = NewtReIDModel().to(device)
    criterion = BatchHardTripletLoss(margin=0.3)
    
    # Разделяем learning rate
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.head.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)
    
    # Меняем scheduler на CosineAnnealing (он лучше работает для Re-ID)
    epochs = 30
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            embeddings = model(images)
            loss = criterion(embeddings, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/best_model.pth')
    print("Training complete. Model saved.")

if __name__ == '__main__':
    train()