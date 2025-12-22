import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import our custom modules
from src.models.unet import UNetGenerator
from src.training.loss import WeightedL1Loss
from src.data.dataloader import S2SDataset

def train_warmstart(config):
    """
    Phase 1: Warm-start training with L1 Loss only (no GAN).
    This establishes a stable baseline before adversarial fine-tuning.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== WARM-START PHASE ===")
    print(f"Training on: {device}")
    print(f"Epochs: {config['epochs']}")
    print(f"Loss: L1 only (no Discriminator)")
    
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("ET-NeuralCast-Warmstart")
        mlflow.start_run()
        mlflow.log_params(config)
    
    # 1. Dataset & Dataloader
    train_dataset = S2SDataset(
        data_dir=config['data_dir'],
        years=config['train_years'],
        lead_weeks=config.get('lead_weeks', 1)
    )
    dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    
    # 2. Model (Generator only)
    G = UNetGenerator(input_channels=5, output_channels=1, target_size=(48, 60)).to(device)
    
    # 3. Optimizer
    opt_G = optim.Adam(G.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    
    # 4. Loss
    criterion_L1 = WeightedL1Loss(weight=5.0).to(device)
    
    # Training Loop
    best_loss = float('inf')
    
    for epoch in range(config['epochs']):
        G.train()
        epoch_loss = 0.0
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        
        for i, (x, y) in enumerate(loop):
            x = x.to(device)
            y = y.to(device)
            
            # Forward pass
            opt_G.zero_grad()
            fake_y = G(x)
            
            # L1 Loss only
            loss = criterion_L1(fake_y, y)
            
            # Backward pass
            loss.backward()
            opt_G.step()
            
            epoch_loss += loss.item()
            
            # Update progress bar
            loop.set_postfix(L1_loss=loss.item())
            
            if MLFLOW_AVAILABLE:
                mlflow.log_metric("batch_L1_loss", loss.item(), step=epoch * len(dataloader) + i)
        
        # Epoch summary
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Avg L1 Loss = {avg_loss:.4f}")
        
        if MLFLOW_AVAILABLE:
            mlflow.log_metric("avg_L1_loss", avg_loss, step=epoch)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"checkpoints/G_warmstart_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': G.state_dict(),
                'optimizer_state_dict': opt_G.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"[+] Saved checkpoint: {checkpoint_path}")
            if MLFLOW_AVAILABLE:
                mlflow.log_artifact(checkpoint_path)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = "checkpoints/G_warmstart_best.pth"
            torch.save(G.state_dict(), best_path)
            print(f"* New best model saved! Loss: {best_loss:.4f}")
            if MLFLOW_AVAILABLE:
                mlflow.log_artifact(best_path)
    
    if MLFLOW_AVAILABLE:
        mlflow.end_run()
    
    print(f"\n=== WARM-START COMPLETE ===")
    print(f"Best L1 Loss: {best_loss:.4f}")

if __name__ == "__main__":
    warmstart_config = {
        'data_dir': 'data/raw',
        'train_years': [2000, 2001],
        'batch_size': 2,
        'lr': 0.0002,
        'epochs': 50,
    }
    
    os.makedirs("checkpoints", exist_ok=True)
    train_warmstart(warmstart_config)
