import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    """
    U-Net Generator for S2S Forecasting.
    Input: Global/Regional Climate Drivers (B, C_in, H, W)
    Output: Precipitation Map (B, C_out, H, W)
    """
    def __init__(self, input_channels, output_channels, hidden_channels=64, target_size=None):
        super(UNetGenerator, self).__init__()
        
        self.target_size = target_size  # (H, W) to resize output to

        # Encoder (Downsampling)
        # Block 1: 64 -> 64 (No Batch Norm in first layer usually, but we use it here for consistency)
        self.enc1 = self.conv_block(input_channels, hidden_channels, bn=False)
        # Block 2: 64 -> 128
        self.enc2 = self.conv_block(hidden_channels, hidden_channels * 2)
        # Block 3: 128 -> 256
        self.enc3 = self.conv_block(hidden_channels * 2, hidden_channels * 4)
        # Block 4: 256 -> 512
        self.enc4 = self.conv_block(hidden_channels * 4, hidden_channels * 8)
        
        # Bottleneck: 512 -> 512
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_channels * 8, hidden_channels * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # Decoder (Upsampling)
        # Block 1: 512 -> 512 (Skip: 512) -> Total Input: 1024
        self.dec1 = self.up_block(hidden_channels * 8, hidden_channels * 8, dropout=0.5)
        # Block 2: 1024 -> 256 (Skip: 256) -> Total Input: 512
        self.dec2 = self.up_block(hidden_channels * 16, hidden_channels * 4, dropout=0.5)
        # Block 3: 512 -> 128 (Skip: 128) -> Total Input: 256
        self.dec3 = self.up_block(hidden_channels * 8, hidden_channels * 2, dropout=0.0)
        # Block 4: 256 -> 64 (Skip: 64) -> Total Input: 128
        self.dec4 = self.up_block(hidden_channels * 4, hidden_channels, dropout=0.0)

        # Output Layer: 128 -> C_out
        self.final = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels * 2, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Normalizes output to [-1, 1]
        )

    def conv_block(self, in_c, out_c, bn=True):
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if bn:
            layers.insert(1, nn.BatchNorm2d(out_c))
        return nn.Sequential(*layers)

    def up_block(self, in_c, out_c, dropout=0.0):
        layers = [
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x) # [B, 64, H/2, W/2]
        e2 = self.enc2(e1) # [B, 128, H/4, W/4]
        e3 = self.enc3(e2) # [B, 256, H/8, W/8]
        e4 = self.enc4(e3) # [B, 512, H/16, W/16]
        
        # Bottleneck
        b = self.bottleneck(e4) # [B, 512, H/32, W/32]
        
        # Decoder (with Skip Connections)
        # Use interpolation to match sizes when skip connections have incompatible dimensions
        d1 = self.dec1(b)
        # Resize e4 to match d1 if needed
        if d1.shape[2:] != e4.shape[2:]:
            e4 = torch.nn.functional.interpolate(e4, size=d1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e4], dim=1)
        
        d2 = self.dec2(d1)
        if d2.shape[2:] != e3.shape[2:]:
            e3 = torch.nn.functional.interpolate(e3, size=d2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e3], dim=1)
        
        d3 = self.dec3(d2)
        if d3.shape[2:] != e2.shape[2:]:
            e2 = torch.nn.functional.interpolate(e2, size=d3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e2], dim=1)
        
        d4 = self.dec4(d3)
        if d4.shape[2:] != e1.shape[2:]:
            e1 = torch.nn.functional.interpolate(e1, size=d4.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e1], dim=1)
        
        out = self.final(d4) # [B, Out, H, W]
        
        # Resize to target size if specified
        if self.target_size is not None:
            out = torch.nn.functional.interpolate(out, size=self.target_size, mode='bilinear', align_corners=False)
        
        return out

if __name__ == "__main__":
    # Test Real-world Shape
    print("Testing U-Net Generator with ERA5 to CHIRPS dimensions...")
    # ERA5 input: 5 channels, 121x360
    # CHIRPS target: 48x60
    B, C_in, H, W = 1, 5, 121, 360
    target_H, target_W = 48, 60
    
    model = UNetGenerator(input_channels=C_in, output_channels=1, target_size=(target_H, target_W))
    x = torch.randn(B, C_in, H, W)
    y = model(x)
    
    print(f"Input (ERA5): {x.shape}")
    print(f"Output (CHIRPS): {y.shape}")
    assert y.shape == (B, 1, target_H, target_W), f"Shape mismatch! Got {y.shape}, expected {(B, 1, target_H, target_W)}"
    print("Test Passed.")

