import torch 
import torch.nn as nn 
import torch.nn.functional as F   
from attention import SelfAttention

class VAE_ResidualBlock(nn.Module): 
    '''
    Collection of GroupNorm and Conv2D layers, with a Residual connection at the end
    '''
    def __init__(self, in_channels, out_channels): 
        super(VAE_ResidualBlock, self).__init__()    
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)  
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) # Input size does not change
        self.groupnorm_2 = nn.GroupNorm(32, out_channels) 
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) # Again height, width does not change 

        if in_channels == out_channels: 
            self.residual_layer = nn.Identity() 
        else: 
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) 
         
    def forward(self, x:torch.Tensor) -> torch.Tensor: 
        out = self.groupnorm_1(x) 
        out = F.silu(out) 
        out = self.conv_1(out) 
        out = self.groupnorm_2(out) 
        out = F.silu(out) 
        out = self.conv_2(out) 

        # If in_channels = out_channels, then we can directly add out + x 
        # In this case, residual_layer = identity function which does not change x at all 
        # However, if out_channels is different, the residual layer uses conv2d to convert channels from in_channels to out_channels for the input X 
        # Now, initial input x can be added to out since they are guaranteed to have the same dimensions. 
        x = self.residual_layer(x)  
        return x + out 
        
class VAE_AttentionBlock(nn.Module): 
    def __init__(self, channels): 
        super(VAE_AttentionBlock, self).__init__()  
        self.groupnorm = nn.GroupNorm(32, channels) 
        self.attention = SelfAttention(1, channels) 

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        
        n,c,h,w = x.shape 
        out = self.groupnorm(x) 
        out = out.view(n,c,h*w) # Make a flattened vector of pixels for each channel 
        out = out.transpose(-1,-2) # Swap the last and 2nd last dimension 
        # The above makes shape: (n,h*w, c). So each pixel now has c values where c = number of channels/features. 
        # We can also call these features as embeddings for each pixel. 
        # We want to find Self Attention between all the pixels. We treat their corresponding features as their embeddings and use that for self-attention 

        # (n,h*w, c) 
        out = self.attention(out) 
        
        # Convert back to (n,c,h*w) 
        out = out.transpose(-1,-2) 
        # Convert back to (n,c,h,w) 
        out = out.view(n,c,h,w)  

        return out + x # Residual Connection

class VAE_Decoder(nn.Sequential): 
    def __init__(self): 
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0), 
            nn.Conv2d(4, 512, kernel_size=3, padding=1), 
            VAE_ResidualBlock(512,512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512,512), 
            VAE_ResidualBlock(512,512), 
            VAE_ResidualBlock(512,512),  
            VAE_ResidualBlock(512,512), 

            # Up until now size of input is still height/8, width/8. 
            # We scaled channels so far, not the size of the input. Need to upscale size
            # Hence, shape: (Batch_size, 512, height/8, width/8) 

            # (batch_size, 512, height/4, width/4) 
            nn.Upsample(scale_factor=2),  
            nn.Conv2d(512,512, kernel_size=3, padding=1), 

            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 

            # (batch_size, 512, h/2, w/2) 
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(512,512, kernel_size=3, padding=1), 

            VAE_ResidualBlock(512,256), 
            VAE_ResidualBlock(256,256), 
            VAE_ResidualBlock(256, 256), 

            # (batch_size, 128, h, w)   
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 

            VAE_ResidualBlock(256,128), 
            VAE_ResidualBlock(128,128), 
            VAE_ResidualBlock(128,128), 

            nn.GroupNorm(32, 128), 
            nn.SiLU(), 

            # (batch_size, 3, h, w) 
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        ) 
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:  
        # x: (batch_size, 4, h/8, w/8) 

        x /= 0.18215 # To reverse the scaling done by encoder at the end on the latent variable 

        for module in self: 
            x = module(x) 

        # x: (batch_size, 3, h,w) 
        return x  