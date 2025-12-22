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
from src.models.discriminator import PixelDiscriminator
from src.training.loss import GANLoss, WeightedL1Loss
from src.data.dataloader import S2SDataset

def train_gan(config):
    """
    Phase 2: GAN fine-tuning with Discriminator.
    Loads warm-start weights and continues training with adversarial loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== GAN FINE-TUNING PHASE ===")
    print(f"Training on: {device}")
    
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("ET-NeuralCast-GAN")
        mlflow.start_run()
        mlflow.log_params(config)
    
    # Dataset
    train_dataset = S2SDataset(
        data_dir=config['data_dir'],
        years=config['train_years'],
        lead_weeks=config.get('lead_weeks', 1)
    )
    dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    
    # Models
    G = UNetGenerator(input_channels=5, output_channels=1, target_size=(48, 60)).to(device)
    D = PixelDiscriminator(input_channels=6).to(device)  # 5 input + 1 target
    
    # Load warm-start weights
    warmstart_path = "checkpoints/G_warmstart_best.pth"
    if os.path.exists(warmstart_path):
        G.load_state_dict(torch.load(warmstart_path, map_location=device))
        print(f"[+] Loaded warm-start weights from {warmstart_path}")
    else:
        print("⚠ Warning: No warm-start weights found, training from scratch")
    
    # Optimizers
    opt_G = optim.Adam(G.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    
    # Losses
    criterion_GAN = GANLoss(mode='mse').to(device)
    criterion_L1 = WeightedL1Loss(weight=5.0).to(device)
    
    # Training Loop
    for epoch in range(config['epochs']):
        G.train()
        D.train()
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        
        epoch_G_L1 = 0
        epoch_G_GAN = 0
        epoch_D = 0
        
        for i, (x, y) in enumerate(loop):
            x = x.to(device)
            y = y.to(device)
            
            # Train Discriminator
            opt_D.zero_grad()
            fake_y = G(x)
            
            pred_real = D(x, y)
            loss_D_real = criterion_GAN(pred_real, target_is_real=True)
            
            pred_fake = D(x, fake_y.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_is_real=False)
            
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            opt_D.step()
            
            # Train Generator
            opt_G.zero_grad()
            
            pred_fake = D(x, fake_y)
            loss_G_GAN = criterion_GAN(pred_fake, target_is_real=True)
            loss_G_L1 = criterion_L1(fake_y, y)
            
            loss_G = loss_G_GAN + (config['lambda_l1'] * loss_G_L1)
            loss_G.backward()
            opt_G.step()
            
            epoch_G_L1 += loss_G_L1.item()
            epoch_G_GAN += loss_G_GAN.item()
            epoch_D += loss_D.item()
            
            loop.set_postfix(G_L1=loss_G_L1.item(), G_GAN=loss_G_GAN.item(), D=loss_D.item())
            
            if MLFLOW_AVAILABLE:
                step = epoch * len(dataloader) + i
                mlflow.log_metric("G_L1_loss", loss_G_L1.item(), step=step)
                mlflow.log_metric("G_GAN_loss", loss_G_GAN.item(), step=step)
                mlflow.log_metric("D_loss", loss_D.item(), step=step)
        
        # Epoch metrics
        if MLFLOW_AVAILABLE:
            mlflow.log_metric("epoch_G_L1_avg", epoch_G_L1 / len(dataloader), step=epoch)
            mlflow.log_metric("epoch_G_GAN_avg", epoch_G_GAN / len(dataloader), step=epoch)
            mlflow.log_metric("epoch_D_avg", epoch_D / len(dataloader), step=epoch)
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            g_path = f"checkpoints/G_gan_epoch_{epoch+1}.pth"
            d_path = f"checkpoints/D_gan_epoch_{epoch+1}.pth"
            torch.save(G.state_dict(), g_path)
            torch.save(D.state_dict(), d_path)
            print(f"[+] Saved GAN checkpoint: epoch {epoch+1}")
            if MLFLOW_AVAILABLE:
                mlflow.log_artifact(g_path)
                mlflow.log_artifact(d_path)
    
    if MLFLOW_AVAILABLE:
        mlflow.end_run()
    
    print(f"\n=== GAN FINE-TUNING COMPLETE ===")

if __name__ == "__main__":
    gan_config = {
        'data_dir': 'data/raw',
        'train_years': [2000, 2001],
        'batch_size': 2,
        'lr': 0.0001,
        'epochs': 30,
        'lambda_l1': 100
    }
    
    os.makedirs("checkpoints", exist_ok=True)
    train_gan(gan_config)
