import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 

'''
This is the DDPM sampler that has the task of removing noise from a latent, given the output from the unet about the noise present in that latent. The output of this sampler is fed back into unet for continous denoising loop 
''' 

class DDPMSampler:  

    '''
    DDPMSampler creates a linear variance schedule. So it equally distributes the interval (beta_start, beta_end) between 1000 steps. 
    '''
    
    def __init__(self, generator, num_training_steps:int=1000, beta_start:float = 0.00085, beta_end:float = 0.0120): 
        '''
        beta_start: First value of variance (beta) [at t=0]
        beta_end: last value of variance [at t = T]   
        '''  

        # Scaled linear schedule 
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2   
        self.alphas = 1.0 - self.betas  
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0) # Pre-computing cumulative product of alphas which will be used in closed formula for getting noisified img at timestep t 

        self.one = torch.tensor(1.0) 
        self.generator = generator 
        self.num_training_steps = num_training_steps 

        # Create initial timesteps schedule. This will be changed later based on the num_inference_steps 

        self.timesteps = torch.from_numpy(np.arange(0,num_training_steps)[::-1].copy())  
    
    def set_inference_steps(self, num_inference_steps:int = 50): 
        '''
        Update the timestep schedule based on the number of inference steps. 
        ''' 
        step = self.num_training_steps // num_inference_steps 
        self.num_inference_steps = num_inference_steps 
        self.timesteps = torch.from_numpy(np.arange(0, self.num_training_steps, step).round()[::-1].copy().astype(np.int64))  

    def add_noise(self, original_samples: torch.Tensor, timesteps: torch.IntTensor) -> torch.FloatTensor: 

        ''' 
        Returns noisified version of latent by giving x_t where timesteps = list of timestep of noise for each original sample   
        The original_samples is treated as x_0 
        '''     

        alphas_prod = self.alphas_cumprod[timesteps].to(original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(device=original_samples.device) 


        sqrt_alphas_prod = torch.sqrt(alphas_prod)
        sqrt_alphas_prod = sqrt_alphas_prod.flatten() 
        
        # Make sure sqrt_alphas_prod has the same shape as original_samples. This while loop will basically make sure that all the sqrt values are in the batch dimension
        # Thus when you multiply, for one batch (that is one image), a particular pre-computed alphas_prod is multiplied in each channel of the 2D tensor (height,width) 
        # This is the same as sqrt(alphas) * x_0  
        while len(sqrt_alphas_prod.shape) < len(original_samples.shape): 
            sqrt_alphas_prod = sqrt_alphas_prod.unsqueeze(-1) 

        # Each sample's appropriate mean is calculated based on its corresponding timestep for which noise has to be added 
        means = sqrt_alphas_prod * original_samples  

        one_minus_alphas_prod = (1-alphas_prod) * self.one  
        std_devs = torch.sqrt(one_minus_alphas_prod)
        std_devs = std_devs.flatten() 
        while len(std_devs.shape) < len(original_samples.shape): 
            std_devs = std_devs.unsqueeze(-1)   
        
        noises = torch.randn(original_samples.shape, generator=self.generator, dtype=original_samples.dtype, device=original_samples.device) # Z ~ N(0,I)  

        # Sampling from this mean and variance  
        Z = means + std_devs * noises 

        return Z 
     

        


    