import torch
import torch.nn as nn

class PixelDiscriminator(nn.Module):
    """
    PatchGAN Discriminator.
    Input: Concatenation of (Condition X, Target Y/Prediction Y_fake)
    Output: Grid of probabilities (Real/Fake)
    """
    def __init__(self, input_channels, hidden_channels=64):
        super(PixelDiscriminator, self).__init__()
        
        # We expect input_channels to be (C_in + C_out) because we concat X and Y
        
        self.model = nn.Sequential(
            # Layer 1: [B, In, H, W] -> [B, 64, H/2, W/2]
            nn.Conv2d(input_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: [B, 64, H/2, W/2] -> [B, 128, H/4, W/4]
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: [B, 128, H/4, W/4] -> [B, 256, H/8, W/8]
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4 (Output): [B, 256, H/8, W/8] -> [B, 1, H/8, W/8]
            # No Sigmoid here! We use BCEWithLogitsLoss for numerical stability
            nn.Conv2d(hidden_channels * 4, 1, kernel_size=4, stride=1, padding=1) 
        )

    def forward(self, x, y):
        # Resize x (ERA5) to match y (CHIRPS) spatial dimensions before concatenation
        if x.shape[2:] != y.shape[2:]:
            x = torch.nn.functional.interpolate(x, size=y.shape[2:], mode='bilinear', align_corners=False)
            
        # Concatenate condition (x) and target (y) along channel dim
        concat_input = torch.cat([x, y], dim=1)
        return self.model(concat_input)

if __name__ == "__main__":
    print("Testing PatchGAN Discriminator with real-world dimensions...")
    # ERA5 condition: 5 channels, 121x360
    # CHIRPS target: 1 channel, 48x60
    B, C_in, C_out = 1, 5, 1
    H_x, W_x = 121, 360
    H_y, W_y = 48, 60
    
    model = PixelDiscriminator(input_channels=C_in + C_out)
    x = torch.randn(B, C_in, H_x, W_x)
    y = torch.randn(B, C_out, H_y, W_y)
    
    pred = model(x, y)
    print(f"Input X (ERA5): {x.shape}")
    print(f"Input Y (CHIRPS): {y.shape}")
    print(f"Output Patch Grid: {pred.shape}")
    
    print("Test Passed.")

