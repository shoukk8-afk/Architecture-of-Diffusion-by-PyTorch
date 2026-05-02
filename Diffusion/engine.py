import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


#訓練ループの構築
def diffusion_training(n_epochs, optimizer, model, diffusion, train_loader, test_loader):
    loss_list = []
    loss_val_list = []
    model.train()
    warmup_epochs = 10
    total_epochs = 100

    
    scheduler_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)

    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=(total_epochs - warmup_epochs), eta_min=1e-6)

    scheduler = SequentialLR(
        optimizer, 
        schedulers=[scheduler_warmup, scheduler_cosine], 
        milestones=[warmup_epochs]
    )
    for epoch in range(1, n_epochs + 1):
        total = 0
        loss_train = 0
        loss_test = 0
        for imgs, _ in train_loader:
            loss = diffusion(model, imgs)

            optimizer.zero_grad()
            loss.backward()            
            optimizer.step()
            
            #1回のループの損失を加算して訓練データの1エボックの損失の合計を出す
            loss_train += loss.item()
        with torch.no_grad():
            model.eval()
            for imgs, _ in test_loader:
                loss_val = diffusion(model, imgs)
                loss_test += loss_val.item()

        avg_train_loss = loss_train / len(train_loader)
        avg_test_loss = loss_test / len(test_loader)
        loss_list.append(avg_train_loss)
        loss_val_list.append(avg_test_loss)

        scheduler.step()

        print(f"Epoch: {epoch} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_test_loss:.6f}")

    return loss_list, loss_val_list