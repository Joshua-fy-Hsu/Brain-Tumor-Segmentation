import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A standard 3D Residual Block with two convolutional layers and a shortcut connection.
    Supports downsampling via stride.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.downsample = None
        
        # If dimensions change, use a 1x1 convolution for the shortcut to match dimensions.
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_channels)
            )
        
        # Main path
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm3d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm3d(out_channels)

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual  # Add skip connection
        out = self.relu(out)
        return out

class ResUnet3D(nn.Module):
    """
    3D Residual U-Net with Deep Supervision.
    
    Architecture:
    - Encoder: 4 stages of Residual Blocks with downsampling.
    - Bottleneck: A bridge between encoder and decoder.
    - Decoder: 3 upsampling stages with skip connections from the encoder.
    - Deep Supervision: Auxiliary outputs at 1/2 and 1/4 resolution to aid training.
    """
    def __init__(self, in_channels=4, num_classes=4, base_filters=32):
        super(ResUnet3D, self).__init__()

        # --- Encoder ---
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, base_filters, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(base_filters),
            nn.LeakyReLU(inplace=True)
        )
        self.enc2 = ResidualBlock(base_filters, base_filters * 2, stride=2) 
        self.enc3 = ResidualBlock(base_filters * 2, base_filters * 4, stride=2)
        self.enc4 = ResidualBlock(base_filters * 4, base_filters * 8, stride=2)
        
        # --- Bottleneck ---
        self.bottleneck = ResidualBlock(base_filters * 8, base_filters * 16, stride=2)

        # --- Decoder ---
        # Level 4 (Deepest)
        self.up4 = nn.ConvTranspose3d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(base_filters * 16, base_filters * 8)
        
        # Level 3 (with Auxiliary Output 2)
        self.up3 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(base_filters * 8, base_filters * 4)
        self.ds2_cls = nn.Conv3d(base_filters * 4, num_classes, kernel_size=1) # Deep Supervision Head

        # Level 2 (with Auxiliary Output 1)
        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(base_filters * 4, base_filters * 2)
        self.ds1_cls = nn.Conv3d(base_filters * 2, num_classes, kernel_size=1) # Deep Supervision Head

        # Level 1 (Final Output)
        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(base_filters * 2, base_filters)
        self.final_conv = nn.Conv3d(base_filters, num_classes, kernel_size=1)
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming Normal for Conv layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.InstanceNorm3d):
                if m.weight is not None: nn.init.constant_(m.weight, 1)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # --- Encoder Path ---
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x_bot = self.bottleneck(x4)
        
        # --- Decoder Path ---
        # Upsample 4
        d4 = self.up4(x_bot)
        d4 = torch.cat((x4, d4), dim=1) # Skip connection
        d4 = self.dec4(d4)
        
        # Upsample 3
        d3 = self.up3(d4)
        d3 = torch.cat((x3, d3), dim=1) # Skip connection
        d3 = self.dec3(d3)
        out_ds2 = self.ds2_cls(d3) # Deep Supervision 2 (1/4 res)

        # Upsample 2
        d2 = self.up2(d3)
        d2 = torch.cat((x2, d2), dim=1) # Skip connection
        d2 = self.dec2(d2)
        out_ds1 = self.ds1_cls(d2) # Deep Supervision 1 (1/2 res)

        # Upsample 1
        d1 = self.up1(d2)
        d1 = torch.cat((x1, d1), dim=1) # Skip connection
        d1 = self.dec1(d1)
        
        # Final Full Resolution Output
        out_final = self.final_conv(d1)
        
        if self.training:
            # Return tuple of outputs for Deep Supervision loss calculation
            return out_final, out_ds1, out_ds2
        else:
            return out_final