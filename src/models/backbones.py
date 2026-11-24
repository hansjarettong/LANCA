import torch
import torch.nn as nn

class SharedEncoder(nn.Module):
    """
    Standard CNN Encoder for all LANCA and Baseline models.
    Input: (B, C, 96, 96) -> Output: (B, latent_dim)
    """
    def __init__(self, img_channels: int, img_size: int, latent_dim: int):
        super().__init__()
        self.img_size = img_size
        
        # Modern CNN architecture (same as your previous ModernEncoder)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 0), nn.ReLU(True), 
        )
        
        # Calculate flat dimension dynamically
        self.flat_dim = self._get_flat_dim(img_channels, img_size)
        self.fc = nn.Linear(self.flat_dim, latent_dim)

    def _get_flat_dim(self, c, s):
        with torch.no_grad():
            dummy = torch.zeros(1, c, s, s)
            out = self.conv_layers(dummy)
        return out.numel()

    def forward(self, x):
        h = self.conv_layers(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)

class SharedDecoder(nn.Module):
    """
    Standard CNN Decoder for all LANCA and Baseline models.
    Input: (B, latent_dim) -> Output: (B, C, 96, 96) Logits (No Sigmoid!)
    """
    def __init__(self, img_channels: int, img_size: int, latent_dim: int, encoder_flat_dim: int):
        super().__init__()
        self.img_size = img_size
        self.fc = nn.Linear(latent_dim, encoder_flat_dim)
        
        # Assuming the encoder ends with 128 channels and 1x1 spatial dim
        # If image size != 96, this might need adjustment, but for 96 it works.
        self.start_shape = (128, 1, 1) 
        
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, 1, 0), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, output_padding=0), nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),
            # Note: No Sigmoid here. We use BCEWithLogitsLoss for stability.
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), *self.start_shape)
        return self.deconv_layers(h)