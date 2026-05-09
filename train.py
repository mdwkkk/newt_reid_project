import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import NewtDataset
from src.sampler import PKSampler
from src.model import NewtReIDModel
from src.loss import BatchHardTripletLoss
import os
from torchvision import transforms
import mlflow
import mlflow.pytorch


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # === MLFLOW ШАГ 2: ВЫНОСИМ ПАРАМЕТРЫ В ОДИН СЛОВАРЬ ===
    # Это позволит MLflow красиво отображать их в таблице сравнения
    hyperparams = {
        "learning_rate_head": 1e-3,
        "learning_rate_backbone": 1e-5,
        "batch_size": 64,
        "epochs": 50,
        "triplet_margin": 0.3,
        "image_size": 256,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealing",
    }

    # === MLFLOW ШАГ 3: НАСТРАИВАЕМ ПРОЕКТ И ЗАПУСКАЕМ ТРЕКИНГ ===
    mlflow.set_experiment("Newt_ReID_Optimization") # Название проекта
    
    with mlflow.start_run(run_name="ViT_Res256_50_Epochs"): # Имя конкретного запуска
        
        # Логируем параметры в интерфейс MLflow
        mlflow.log_params(hyperparams)

        # Аугментации
        train_transform = transforms.Compose([
            transforms.Resize((hyperparams["image_size"], hyperparams["image_size"])),
            
            # === ИСПРАВЛЕНО ===
            # Шанс 50% применить поворот ровно на 180 градусов
            transforms.RandomApply([
                transforms.RandomRotation((180, 180))
            ], p=0.5),
            
            # Легкое изменение цвета
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            
            # Очень легкий поворот
            transforms.RandomRotation(degrees=10),
            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = NewtDataset('data/train_unwrapped.csv', transform=train_transform)
        sampler = PKSampler(dataset, p_classes=16, k_instances=4) 
        dataloader = DataLoader(dataset, batch_size=hyperparams["batch_size"], sampler=sampler, num_workers=4, drop_last=True)
        
        model = NewtReIDModel(img_size=hyperparams["image_size"]).to(device)
        criterion = BatchHardTripletLoss(margin=hyperparams["triplet_margin"])
        
        optimizer = optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': hyperparams["learning_rate_backbone"]},
            {'params': model.head.parameters(), 'lr': hyperparams["learning_rate_head"]}
        ], weight_decay=1e-4)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyperparams["epochs"])
        
        for epoch in range(hyperparams["epochs"]):
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
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{hyperparams['epochs']}, Loss: {avg_loss:.4f}")
            
            # === MLFLOW ШАГ 4: ЛОГИРУЕМ ГРАФИК ПАДЕНИЯ ОШИБКИ ===
            # MLflow автоматически построит график лосса по эпохам
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            
        # Локальное сохранение (оставляем как было)
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), 'models/best_model.pth')
        print("Training complete. Local model saved.")

        # === MLFLOW ШАГ 5: СОХРАНЯЕМ САМУ МОДЕЛЬ В АРТЕФАКТЫ ===
        mlflow.pytorch.log_model(model, "best_model_artifact")
        print("Model architecture and weights logged to MLflow.")

if __name__ == '__main__':
    train()