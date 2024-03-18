import torch 
import torch.nn as nn 
import torch.nn.functional as F  
from decoder import VAE_AttentionBlock, VAE_ResidualBlock 

class VAE_Encoder(nn.Sequential):   
    '''
    This class handles the Encoder part of the VAE, as well as the sampling from the Latent Space in one class.  
    The Encoder works by increasing the number of channels and reducing size of the image
    '''
    def __init__(self): 
        super.__init__( 
            # (batch_size, 128, height, width) 
            nn.Conv2d(3, 128, kernel_size=3, padding=1), 
            VAE_ResidualBlock(128,128), 
            VAE_ResidualBlock(128,128),  

            # (batch_size, 128, height/2, width/2)
            nn.Conv2d(128,128, kernel_size=3, stride=2, padding=0), 
            
            # (batch_size, 256, h/2, w/2) 
            VAE_ResidualBlock(128,256), 
            VAE_ResidualBlock(256,256),  

            # (batch_size, 256, h/4, w/4) 
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),  

            # (batch_size, 512, h/4, w/4) 
            VAE_ResidualBlock(256,512), 
            VAE_ResidualBlock(512,512), 

            # (batch_size, 512, h/8, w/8) 
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), 
            VAE_ResidualBlock(512,512), 
            VAE_ResidualBlock(512,512), 
            VAE_ResidualBlock(512,512), 
            VAE_AttentionBlock(512), 
            VAE_ResidualBlock(512,512), 
            nn.GroupNorm(32, 512), 
            nn.SiLU(), 
            nn.Conv2d(512, 8, kernel_size=3, padding=1), 
            nn.Conv2d(8,8, kernel_size=1, padding=0)
        )  

    def forward(self, x: torch.Tensor, noise:torch.Tensor) -> torch.Tensor: 

        for module in self: 
            if getattr(module, 'stride', None) == (2,2): 
                x = F.pad(x,(0,1,0,1))  # pad = (Left, right, top, bottom)  

            x = module(x)  
        
        # At this point, we have an 8 channel bottleneck. However, this is VAE, not AE. 
        # So we need to estimate the gaussian parameters of the latent space
        # not just compress the input 
            
        # x has shape = (Batch_size, channels = 8, Height/8, width/8)  
        # Chunk makes two chunks (divides input tensor into 2 parts on dimension = 1, which is the number of channels). 
        # So each chunk has 4 channels => (Batch_size, 4,height/8, width/8) 

        mean, log_variance = torch.chunk(x,2, dim=1)  

        # Make sure log_variance is between -30 and 20
        log_variance = torch.clamp(log_variance, -30,20) 

        variance = log_variance.exp() # Converting log_variance into variance 
        std = torch.sqrt(variance) 

        # Sampling from latent space. Z ~ N(mean, std) 
        Z = mean + std * noise  

        # Original Stable Diffusion model scaled output by this constant for some reason 
        Z *= 0.18215

        return Z 

        