import torch 
import torch.nn as nn 
import torch.nn.functional as F   
import math 

class SelfAttention(nn.Module): 
    '''
    Does Self Attention on a tensor of shape (N,H*W,d_embed)  
    N = number of batches 
    H*W = number of pixels 
    d_embed = total number of embeddings per pixel (in our implementation, we treat the corresponding value in each channel for a pixel as its embedding)
    ''' 
    
    # C = d_model 
    # d_k = d_model  / num_heads 
    def __init__(self, num_heads:int , d_embed:int , in_proj_bias = True, out_proj_bias=True):
        super(SelfAttention, self).__init__()  

        # Instead of 3 different matrices, we can just make 1 linear layer with 3*d_embed output channel shape 
        # self.Wq =  torch.rand((d_embed, d_embed), requires_grad=True) 
        # self.Wk =  torch.rand((d_embed, d_embed), requires_grad=True) 
        # self.Wv =  torch.rand((d_embed, d_embed), requires_grad=True)   

        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias=in_proj_bias)   
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias) 
        self.n_heads = num_heads  
        self.d_head = d_embed // num_heads 

    def forward(self, x:torch.Tensor, causal_mask = False) -> torch.Tensor: 
        '''
        x is of shape (N,H*W,C) 
        Mask helps to make compute self attention for a pixel only by considering pixels that come before it, not after. 
        '''   
        input_shape = x.shape 
        batch_size, seq_len, d_embed = input_shape

        # We do this for multiheaded self-attention. Splitting d_embed across n_heads 
        interm_shape = (batch_size, seq_len, self.n_heads, self.d_head)  

        out = self.in_proj(x)  
        
        # (Batch_size, Seq_len, 3*d_embed) -> 3 tensors of (Batch_size, Seq_len, d_embed)
        Q,K,V = torch.chunk(out, 3, dim=-1)   
        
        Q = Q.view(interm_shape).transpose(1,2)  
        K = K.view(interm_shape).transpose(1,2) 
        V = V.view(interm_shape).transpose(1,2)    
        
        # K gets converted from (Batch_size, n_heads, seq_len, d_head) -> (Batch_size, n_heads, d_head, seq_len)
        # output of weight = (batch_size, n_heads, seq_len, seq_len)
        weight = Q @ K.transpose(-1,-2)  

        if causal_mask: 
            # We just want to have a mask on elements above the main diagonal 
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)  
            # We fill the upper triangular part with -inf so that softmax makes it 0 when comparison weights are calculated. 
            weight.masked_fill_(mask, value=-torch.inf) 

        # (batch_size, n_heads, seq_len, seq_len) -> (Batch_size, n_heads, seq_len, d_head)
        output = F.softmax(weight/math.sqrt(self.d_head), dim=-1) @ V     
        
        # Converting back the dimension order to (Batch_size, seq_len, n_heads, d_head)
        output = output.transpose(1,2) 
        
        # Concatinating all the heads outputs  
        # (batch_size, H*W, d_embed) 
        output = output.reshape(input_shape)
        
        # (batch_size, H*W, d_embed) 
        output = self.out_proj(output)  

        return output  


class CrossAttention(nn.Module): 
    '''
    Calculates Cross Attention for latent by comparing each query of latent with keys of context. 
    No causal_mask here because we are relating each pixel to each text token in the prompt. There is no concept of "look only at previous tokens" because both sequences are not of same length  
    '''
    
    def __init__(self, n_head:int, d_latent:int, d_context:int, 
                 in_proj_bias = True, out_proj_bias = True): 
        
        super(CrossAttention, self).__init__()  
        self.d_head = d_latent // n_head 
        self.n_head = n_head 

        # Key and Value come from the input or the context vector. Basically, the vector which we are checking contributes how much 
        
        # We could have done the network for K and V like this and then used chunk later, but since it does not match the pretrained stable diffusion model format, I am making a different linear network for K and V 
        # self.kv_proj = nn.Linear(d_context, 2* d_latent, bias=in_proj_bias)  

        self.k_proj = nn.Linear(d_context, d_latent, bias=in_proj_bias) 
        self.v_proj = nn.Linear(d_context, d_latent, bias=in_proj_bias) 
        self.q_proj = nn.Linear(d_latent, d_latent, bias=in_proj_bias)  
        self.out_proj = nn.Linear(d_latent, d_latent, bias=out_proj_bias) 
    
    def forward(self, latent:torch.Tensor, context:torch.Tensor) -> torch.Tensor: 

        # latent: (Batch_size, Seq_len_1, d_latent) 
        # context: (Batch_size, Seq_len_2 = 77, d_context = 768) 

        # Two tensors of shape (batch_size, 77, d_latent) 
        # K,V = self.kv_proj(context).chunk(2, dim=-1) 

        K = self.k_proj(context) 
        V = self.v_proj(context) 
        Q = self.q_proj(latent) 

        batch_size, seq_len1, d_latent = latent.shape   
        batch_size, seq_len2, d_context = context.shape 
        
        interm_shape_latent = (batch_size, seq_len1, self.n_head, self.d_head) 
        interm_shape_context = (batch_size, seq_len2, self.n_head, self.d_head) 

        Q = Q.view(interm_shape_latent).transpose(1,2) # (Batch_size, n_head, seq1, d_head) 
        K = K.view(interm_shape_context).transpose(1,2) # (Batch_size, n_head, seq2, d_head)
        V = V.view(interm_shape_context).transpose(1,2)  # (Batch_size, n_head, seq2, d_head)

        # (batch_size, n_head, seq_len1, seq_len2)  
        scores = F.softmax((Q @ K.transpose(-1,-2)) / math.sqrt(self.d_head), dim=-1)   
        cross_att = scores @ V  # (batch_size, n_head, seq_len1, d_head) 

        cross_att = cross_att.transpose(1,2).contiguous()  
        cross_att = cross_att.reshape((batch_size, seq_len1, d_latent))   

        # Passing this sequence through final Linear layer 
        cross_att = self.out_proj(cross_att) 

        return cross_att 

