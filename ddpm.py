import torch 
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

    def _get_previous_timestep(self, timestep: int): 
        step = self.num_training_steps // self.num_inference_steps  
        return timestep - step  
    
    def _get_variance(self, timestep: int) -> torch.Tensor: 
        t = timestep 
        prev_t = self._get_previous_timestep(t) 

        alpha_prod_t = self.alphas_cumprod[t]  
        alpha_prod_prev_t = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one  
        beta_prod_t = (1-alpha_prod_t) 
        beta_prod_prev_t = (1-alpha_prod_prev_t) 
        current_alpha_t = alpha_prod_t / alpha_prod_prev_t 
        current_beta_t = 1 - current_alpha_t

        # Computed using formula 7 of DDPM paper 
        variance =  (beta_prod_prev_t * current_beta_t) / beta_prod_t 
        variance = torch.clamp(variance, min=1e-20) 

        return variance 

        
    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor): 
        '''
        Removes noise from latent based on model_output. In the diffusion maths, this is the q(x_t-1 | x_t, x_0). 
        Thus, we are getting the previous time step (less noisy version), given the current noisy image and the predicted x_0 as context. As shown by the math, this is a closed formula 
        (Eq 7 in DDPM paper). For x_0 in the formula, we refer to Eq 15 in DDPM paper. 
        '''
        t = timestep   
        prev_t = self._get_previous_timestep(t)  

        alpha_prod_t = self.alphas_cumprod[t]  
        alpha_prod_prev_t = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one  
        beta_prod_t = (1-alpha_prod_t) 
        beta_prod_prev_t = (1-alpha_prod_prev_t)  
        current_alpha_t = alpha_prod_t / alpha_prod_prev_t 
        current_beta_t = 1 - current_alpha_t  

        # Compute predicted original sample using formula 15 of the DDPM paper  
        pred_original_sample = (latents - (beta_prod_t ** 0.5) * model_output) / (alpha_prod_t ** 0.5) 

        # Compute coefficients for pred_original_sample and current sample x_t (Eq 7)  
        pred_original_sample_coeff = (alpha_prod_prev_t ** 0.5 * current_beta_t) / beta_prod_t  
        current_sample_coeff = (current_alpha_t ** 0.5 * beta_prod_prev_t) / beta_prod_t 

        # Compute the predicted previous time step mean  
        pred_previous_mean = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents 

        pred_prev_std_dev = 0 
        if t > 0: 
            pred_prev_std_dev = (self._get_variance(t) ** 0.5)  

        device = model_output.device 
        noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
        
        pred_prev_sample = pred_previous_mean + pred_prev_std_dev * noise 
        
        return pred_prev_sample

    def set_strength(self, strength:float = 1.0): 
        '''
        Alters start step of the timesteps schedule according to the strength. The more this value, the more noise will be added. 
        ''' 

        start_step = self.num_inference_steps - int(self.num_inference_steps * strength) # If strength = 0.8, we skip 20% of the inference steps 
        self.timesteps = self.timesteps[start_step:] 
        self.start_step = start_step 
    
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
     

        


    