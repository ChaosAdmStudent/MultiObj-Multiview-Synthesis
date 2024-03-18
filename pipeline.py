import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
from tqdm import tqdm 
from ddpm import DDPMSampler  

HEIGHT, WIDTH = 512, 512 
LATENT_HEIGHT, LATENT_WIDTH = HEIGHT//8, WIDTH//8 

def generate(prompt:str, unconditional_prompt:str, input_image=None, strength=0.8, 
             do_cfg = True, cfg_scale = 7.5, 
             sampler_name='ddpm', n_inference_steps=50, models={}, seed=None, 
             device = None, idle_device=None, 
             tokenizer=None
             ): 
    '''
    Main function that allows text to img and img-to-img  

    unconditional_prompt: Negative prompt. It will try to stay clear of whatever this string says in the output.  
    input_image: optional if you want to include an img as input 
    strength: how much attention you want to pay to initial starting image, i.e, how less noise do you want to add to the img. 
    do_cfg: Whether to do classifier free guidance 
    cfg_scale: how much attention we want to pay to the prompt (ranges from 1 to 14)  
    models = dictionary of pretrained models 
    device: where to host the model (Some GPU) 
    idle_device: where to transfer the model if we are not using the model anymore. (Usually CPU) 
    inference_steps: number of denoising steps you do. Even if max steps = 1000, we can just denoise 50 times and get a decent image. The more this number, the better quality the output 
    ''' 

    with torch.no_grad(): 
        if not (0 < strength <= 1): 
            raise ValueError("Strength must be between 0 and 1") 
    
        if idle_device: 
            to_idle: lambda x: x.to(idle_device) 
        else: 
            to_idle: lambda x: x 
        
        generator = torch.Generator(device=device) 
        if seed is None: 
            generator.seed() 
        else: 
            generator.manual_seed(seed) 
        
        clip = models['clip']  
        clip = clip.to(device) 

        if do_cfg: 
            # Convert the prompt into tokens using the tokenizer 
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids 

            # Convert tokens into tensor of size (Batch_size=1, Seq_len=77)  
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)  

            # Running tokens through CLIP model to get embedding vector for each token. 
            # (batch_size, seq_len) -> (batch_size=1, seq_len=77, n_embed=768)
            cond_context = clip(cond_tokens)    

            # Doing the same for unconditional prompt 
            uncond_tokens = tokenizer.batch_encode_plus([unconditional_prompt], padding='max_length', max_length=77).input_ids 
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)  
            uncond_context = clip(uncond_tokens)   

            # Concatenate both the context on the batch dimension. So we end up increasing the batch dimension.  
            # (batch_size, seq_len, n_embed) = (2,77,768)  
            # Since there are two batches here, this will make model give 2 outputs, one for conditional prompt and the other for unconditional prompt 
            context = torch.cat([cond_context, uncond_context], dim=0)    

        else: 
            tokens = tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids  
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)  
            context = clip(tokens)   # (1,77,768)   
        
        # Since we are done using CLIP, move it to idle device 
        
        to_idle(clip)    

        if sampler_name=='ddpm': 
            sampler = DDPMSampler(generator) 
            sampler.set_inference_steps(n_inference_steps) 
        
        else: 
            raise ValueError("No other sampler implemented yet!")  
        
        latents_shape = (1,4,LATENT_HEIGHT, LATENT_WIDTH) 

        if input_image: 
            encoder = models['encoder'] 
            encoder.to(device) 

            input_image = input_image.reshape((HEIGHT,WIDTH)) 
            input_image_tensor = np.array(input_image)  

            # (Height, Width, Channels) = (512,512,3)  
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)  

            # Scaling values from [0,255] to [-1,1] 
            input_image_tensor = rescale(input_image_tensor, (0,255), (-1,1)) 

            # (Batch_size, Height, Width, Channels) = (1,512,512,3) 
            input_image_tensor = input_image_tensor.unsqueeze(0)  

            # (Batch_size, Channels, Height, Width) = (1,3,512,512)
            input_image_tensor = input_image_tensor.transpose(1,-1).transpose(-1,-2) 
