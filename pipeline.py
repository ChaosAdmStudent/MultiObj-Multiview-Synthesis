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
    Main function that allows text to img and img-to-img. It's an inference time function, so no training is being done here. 

    unconditional_prompt: Negative prompt. It will try to stay clear of whatever this string says in the output.  
    input_image: optional if you want to include an img as input 
    strength: how much attention you want to pay to  starting image, i.e, how much noise do you want to add to the img.The more this value, the more output will be differnt from input  
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
            to_idle = lambda x: x.to(idle_device) 
        else: 
            to_idle = lambda x: x 
        
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

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device) 

            # Run image through the encoder of the VAE 
            latents = encoder(input_image_tensor, encoder_noise) 

            sampler.set_strength(strength=strength) # Creates time step schedule based on strength 
            latents = sampler.add_noise(latents, sampler.timesteps[0])  

            to_idle(encoder) 
        
        else: 
            # If we do text-to-img, start with random noise ~ N(0,I)
            latents = torch.randn(latents_shape, generator=generator, device=device) 

        
        # If we have 1000 total time steps during training: 0 ... 999 
        # During inference, if we only do 50 steps, that means we will denoise with noise levels at: 1000 980 960 940 .... 0 (skipping 20 time steps at a time) 

        diffusion = models['diffusion'] 
        diffusion.to(device)     
 
        timesteps = tqdm(sampler.timesteps) 

        # Denoising Loop 
        for i, timestep in enumerate(timesteps): 
            # (1,320) 
            time_embedding = get_time_embedding(timestep).to(device) 
            
            # (Batch_size, 4, latents_height = 64, latents_width = 64)  
            model_input = latents  

            if do_cfg: 
                # Repeats on the batch dimension 
                # (batch_size, 4, latent_height, latent_width) -> (2* batch_size, 4,latent height, latent width) 
                # We make 2 copies of the latent, one to be used with conditional prompt, the other to be used with unconditional prompt 
                model_input =  model_input.repeat(2,1,1,1) 

            # Model output is the predicted noise by UNET 
            model_output = diffusion(latents, context, time_embedding) 

            if do_cfg: 
                output_cond, output_uncond = model_output.chunk(2, dim=0) 
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond 

            # Remove noise predicted by the UNET 
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion) 

        decoder = models['decoder'] 
        decoder.to(device) 

        images = decoder(latents) 
        to_idle(decoder) 

        images = rescale(images, (-1,1), (0,255), clamp=True) 
        images = images.permute(0,2,3,1)   
        images = images.to('cpu', torch.uint8).numpy()  
        return images 
    
def rescale(x, old_range, new_range, clamp=False):  
    '''
    Returns rescaled tensor from old_range to new_range 
    '''
    old_min, old_max = old_range 
    new_min, new_max = new_range 

    x -= old_min 
    x *= (new_max - new_min) / (old_max-old_min)  
    x += new_min 

    if clamp: 
        x = x.clamp(new_min, new_max) 
    
    return x  

def get_time_embedding(timestep:int):  
    '''
    Returns positional time encodings for a given timestep. 
    timestep: int position of time 
    '''
    
    # (160, ) 
    sin_freqs = torch.pow(10000, -torch.arange(start=0, end=320, step=2, dtype=torch.float32)/160) 
    cos_freqs = torch.pow(10000, -torch.arange(start=1, end=320, step=2, dtype=torch.float32)/160) 

    # (1,1) * (1,160) = (1,160) 
    # Adding none in a particular position like this is like torch.unsqueeze() 
    sin_freqs = torch.tensor([timestep], dtype=torch.float32)[:, None] * sin_freqs[None]   
    cos_freqs = torch.tensor([timestep], dtype=torch.float32)[:, None] * cos_freqs[None]   

    # (1,320) --> Merges frequencies on the channel dimension, so 160 + 160 = 320 
    time_embedding = torch.cat([torch.sin(sin_freqs), torch.cos(cos_freqs)], dim=-1)  

    return time_embedding 

