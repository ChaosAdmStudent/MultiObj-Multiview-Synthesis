import torch 
import torch.nn as nn 
from torch.nn import functional as F 
from attention import SelfAttention 

class CLIP_Embeddings(nn.Module): 
    def __init__(self, vocab_size:int, n_embed:int, seq_len:int): 
        '''
        Vocab_size: Total number of unique words 
        n_embed: No. of embeddings per word to generate 
        seq_len: Maximum length of tokens in input. This will be used for padding purposes 
        '''

        # Create embeddings for words 
        super(CLIP_Embeddings, self).__init__() 
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embed)
        self.position_embedding = nn.Parameter(torch.zeros((seq_len, n_embed)))  
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:   
        out = self.token_embedding(x) 
        out += self.position_embedding 

        return out 

class CLIP_Layer(nn.Module): 
    def __init__(self, n_head:int, d_embed:int): 
        '''
        n_head: number of heads to use in Self-Attention 
        d_embed: total number of features/embeddings per token. = n_embed 
        '''
        super(CLIP_Layer, self).__init__() 
        self.layernorm_1 = nn.LayerNorm(d_embed) 
        self.attention = SelfAttention(n_head, d_embed) 
        self.layernorm_2 = nn.LayerNorm(d_embed)
        self.linear_1 = nn.Linear(d_embed, 4 * d_embed) 
        self.linear_2 = nn.Linear(4* d_embed, d_embed)     
    
    def forward(self, x:torch.Tensor) -> torch.Tensor: 
        # x: (batch_size, seq_len, d_embed) 

        res = x 
        out = self.layernorm_1(x) 
        out = self.attention(out, causal_mask=True) # We want causal mask for text
        out += res  # Residual skip connection 
        res = out 
        out = self.layernorm_2(out) 
        out = self.linear_1(out)  
        out = out * torch.sigmoid(1.702 * out) # QuickGELU activation function
        out = self.linear_2(out) 
        out += res # Residual Skip connection 

        return out 

class CLIP(nn.Module): 
    '''
    Builds the whole CLIP Encoder architecture with the basic components of CLIP_Embeddings and CLIP_Layer 
    '''
    def __init__(self): 
        super(CLIP, self).__init__()   

        # These parameter values are coming from the original Stable Diffusion implementation 
        self.embedding = CLIP_Embeddings(vocab_size=49408, n_embed=768, seq_len=77)
        self.layers = nn.ModuleList([
            CLIP_Layer(12, 768) for _ in range(12)
        
        ])  

        self.layernorm = nn.LayerNorm(768) 
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor: 
        tokens = tokens.type(torch.long)
        out = self.embedding(tokens) 
        for layer in self.layers: 
            out = layer(out) 
        
        out = self.layernorm(out) 
        return out 

