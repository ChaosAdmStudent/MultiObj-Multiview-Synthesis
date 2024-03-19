import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from attention import SelfAttention, CrossAttention 

class TimeEmbedding(nn.Module): 
    '''
    Creates embedding of the time positional encoding vector 
    '''
    def __init__(self, t_steps): 
        super(TimeEmbedding, self).__init__() 
        self.linear_1 = nn.Linear(t_steps, 4*t_steps) 
        self.linear_2 = nn.Linear(4*t_steps, 4*t_steps) 
    
    def forward(self, x:torch.Tensor): 
        out = self.linear_1(x) 
        out = F.silu(out) 
        out = self.linear_2(out) 

        # (1,1280)  
        return out 
    
class UNET_ResidualBlock(nn.Module): 
    '''
    We are relating the latent with the time embedding here. 
    '''
    def __init__(self, in_features, out_features, n_time=1280): 
        super(UNET_ResidualBlock, self).__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_features) 
        self.conv_feature = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)   
        self.linear_time = nn.Linear(n_time, out_features)
        self.groupnorm_merged = nn.GroupNorm(32, out_features) 
        self.conv_merged = nn.Conv2d(out_features, out_features, kernel_size=3, padding=1) 

        if in_features == out_features: 
            self.residual_layer = nn.Identity()
        else: 
            self.residual_layer = nn.Conv2d(in_features, out_features, kernel_size=1, padding=0) 

    def forward(self, feature:torch.Tensor, time: torch.Tensor):  
        # feature is the latent : (batch_size, in_features, h,w) 
        # time embedding: (1,1280) 

        res = feature 

        feature = self.groupnorm_feature(feature)  
        feature = F.silu(feature) 
        feature = self.conv_feature(feature)  

        time = F.silu(time) 
        time = self.linear_time(time)  

        # Adding empty height and width for time because it only has batch_size and channels 

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)  # Adding linear time embedding  
        
        merged = self.groupnorm_merged(merged) 
        merged = F.silu(merged) 
        merged = self.conv_merged(merged)  
        
        res = self.residual_layer(res)  

        # Add skip connection 
        return merged + res 

class UNET_AttentionBlock(nn.Module): 
    '''
    Calculates cross attention between latent and context  
    d_context = number of features of context (768 in stable diffusion) 
    '''
    def __init__(self, n_head, n_embed, d_context=768): 
        super(UNET_AttentionBlock, self).__init__() 
        channels =  n_head * n_embed 

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6) 
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0) 

        self.layernorm_1 = nn.LayerNorm(channels) 
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False) 
        self.layernorm_2 = nn.LayerNorm(channels) 
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias = False) 
        self.layernorm_3 = nn.LayerNorm(channels) 
        self.linear_geglu_1 = nn.Linear(channels, 4*channels * 2) 
        self.linear_geglu_2 = nn.Linear(4*channels, channels) 

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0) 
    
    def forward(self, latent:torch.Tensor, context:torch.Tensor): 
        # latent : (batch_size, features, h, w) 
        # context: (batch_size, seq_len=77, dim=768) 

        res_long = latent 

        latent = self.groupnorm(latent) 
        latent = self.conv_input(latent) 

        n,c,h,w = latent.shape 

        latent = latent.view((n,c,h*w)) 
        latent.transpose(1,2) # (n,h*w, c) 

        # Normalization + Self Attention with Skip Connection  

        res_short = latent 
        latent = self.layernorm_1(latent)  
        latent = self.attention_1(latent) # Self Attention 
        latent += res_short 

        # Normalization + Cross Attention with skip connection 
        res_short = latent  
        latent = self.layernorm_2(latent)  
        latent = self.attention_2(latent, context) # Cross Attention 
        latent += res_short 

        # Feed-Forward Layer with GeGLU and Skip Connection 
        res_short = latent 
        latent, gate = self.linear_geglu_1(latent).chunk(2, dim=-1)   
        latent = latent * F.gelu(gate)  # GeGLU activation function (has lot of parameters) 
        latent = self.linear_geglu_2(latent) 
        latent += res_short 

        # Change back dimensions of latent before last long skip connection  
        latent = latent.transpose(-1,-2) # (n,c,h*w) 
        latent = latent.view((n,c,h,w))

        # Final Conv + Normalization + Long Skip Connection  
        latent = self.conv_output(latent) 
        latent = self.layernorm_3(latent) 
        latent += res_long 

        return latent 
    
class SwitchSequential(nn.Sequential): 

    def forward(self, x:torch.Tensor, context: torch.Tensor, time: torch.Tensor): 

        for layer in self: 
            if isinstance(layer, UNET_ResidualBlock): 
                x = layer(x, time)  

            elif isinstance(layer, UNET_AttentionBlock): 
                x = layer(x, context)  

            else: 
                x = layer(x)   

class UpSample(nn.Module):  

    def __init__(self, channels): 
        super(UpSample, self).__init__() 
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1) 
    
    def forward(self, x): 
        # (batch_size, channels, h, w) -> (batch_size, channels, h*2, w*2) 
        out = F.interpolate(x, scale_factor=2, mode='nearest') 
        out = self.conv(out) 

        return out  

class UNET(nn.Module): 
    def __init__(self): 

        super(UNET,self).__init__() 
        
        # Increase number of features, reduce size of input 
        self.encoders = nn.ModuleList([
            # input: (batch_size, 4, h/8, w/8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)), 


            SwitchSequential(
                UNET_ResidualBlock(320,320), 
                UNET_AttentionBlock(8,40)
            ), 

            # output: (batch_size, 320, h/16, w/16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)), 

            # output: (batch_size, 640, h/16, w/16) 
            SwitchSequential(
                UNET_ResidualBlock(320, 640), 
                UNET_AttentionBlock(8, 80) 
            ), 

            SwitchSequential(
                UNET_ResidualBlock(640, 640), 
                UNET_AttentionBlock(8, 80) 
            ),  

            # Output: (batch_size, 640, h/32, w/32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)), 

            SwitchSequential(
                UNET_ResidualBlock(640, 1280), 
                UNET_AttentionBlock(8, 160) 
            ),  

            SwitchSequential(
                UNET_ResidualBlock(1280, 1280), 
                UNET_AttentionBlock(8, 160) 
            ),  

            # out: (batch_size, 1280, h/64, w/64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)), 

            SwitchSequential(UNET_ResidualBlock(1280,1280)), 

            SwitchSequential(UNET_ResidualBlock(1280,1280)) 
        ]) 

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280), 
            UNET_AttentionBlock(8, 160), 
            UNET_ResidualBlock(1280, 1280)  
        )   

        # Bottleneck has output 1280 but decoder takes 2560 because there is a skip connection from last encoder layer
        self.decoders = nn.ModuleList([ 
            # (batch_size, 2560, h/64, w/64) -> (batch_size, 1280, h/64, w/64) 
            SwitchSequential(UNET_ResidualBlock(2560, 1280)), 
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(
                UNET_ResidualBlock(2560, 1280),
                UpSample(1280)
                ),  
            
            SwitchSequential(
                UNET_ResidualBlock(2560, 1280),
                UNET_AttentionBlock(8,160)
                ),   
            
            SwitchSequential(
                UNET_ResidualBlock(2560, 1280),
                UNET_AttentionBlock(8,160)
                ),   

            SwitchSequential(
                UNET_ResidualBlock(1920, 1280),
                UNET_AttentionBlock(8,160), 
                UpSample(1280) 
                ),   

            SwitchSequential(
                UNET_ResidualBlock(1920, 640),
                UNET_AttentionBlock(8,80)
                ),   
            
            SwitchSequential(
                UNET_ResidualBlock(1280, 640),
                UNET_AttentionBlock(8,80)
                ),  
            
            SwitchSequential(
                UNET_ResidualBlock(960, 640),
                UNET_AttentionBlock(8,80), 
                UpSample(640)
                ),  
            
            SwitchSequential(
                UNET_ResidualBlock(960, 320),
                UNET_AttentionBlock(8,40)
                ),  
            
            SwitchSequential(
                UNET_ResidualBlock(640, 320),
                UNET_AttentionBlock(8,40)
                ),  
            
            SwitchSequential(
                UNET_ResidualBlock(640, 320),
                UNET_AttentionBlock(8,40)
                )
        ]) 
    
    def forward(self, x, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim) 
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        return x


class UNET_OutputLayer(nn.Module):  
    def __init__(self, in_channels:int, out_channels:int): 
        super(UNET_OutputLayer, self).__init__() 
        self.groupnorm = nn.GroupNorm(32, in_channels) 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) 

    def forward(self, x):  
        # x: (batch_size, 320, h/8, w/8)   
        out = self.groupnorm(x)  
        out = F.silu(out) 
        out = self.conv(out) 

        return out  


class Diffusion(nn.Module): 
    '''
    This is the high-level implementation of the diffusion's reverse process. 
    It takes 3 inputs: the latent variable (Z), prompt/context embedding and time embedding (for informing time step to predict necessary amount of noise)
    '''

    def __init__(self): 

        super(Diffusion, self).__init__() 
        self.time_embedding = TimeEmbedding(320) 
        self.unet = UNET() 
        self.final = UNET_OutputLayer(320, 4) 

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor: 

        time = self.time_embedding(time)   
        out = self.unet(latent, context, time) 
        out = self.final(out) 

        return out 