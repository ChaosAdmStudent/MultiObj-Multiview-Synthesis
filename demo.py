import sys
sys.dont_write_bytecode = True

import model_loader 
import pipeline 
from PIL import Image 
from transformers import CLIPTokenizer
import torch 

def main(): 

    DEVICE = 'cpu' 

    ALLOW_CUDA = True 
    ALLOW_MPS = False 

    if torch.cuda.is_available() and ALLOW_CUDA: 
        DEVICE = 'cuda' 

    elif torch.backends.mps.is_available() and ALLOW_MPS: 
        DEVICE = 'mps' 

    print(f'Using device: {DEVICE}')  

    tokenizer = CLIPTokenizer('../data/vocab.json', merges_file='../data/merges.txt')
    model_file = '../data/v1-5-pruned-emaonly.ckpt' 
    models = model_loader.preload_models_from_standard_weights(model_file, DEVICE) 

    # Text to Image 

    prompt = 'A black car driving over a bridge on top of a blue river' 
    uncond_prompt = ''   
    do_cfg = True 
    cfg_scale = 8 

    # Image to Image 

    input_image = None 
    image_path = '../images/car.jpg' 
    # input_image = Image.open(image_path) 
    strength = 0.8 

    sampler = 'ddpm' 
    num_inference_steps = 50 
    seed = 53 

    output_image = pipeline.generate(
        prompt=prompt, 
        unconditional_prompt=uncond_prompt, 
        input_image=input_image, 
        strength=strength, 
        do_cfg=do_cfg, 
        cfg_scale=cfg_scale, 
        sampler_name=sampler, 
        n_inference_steps=num_inference_steps, 
        models=models, 
        seed=seed, 
        device=DEVICE, 
        idle_device='cpu', 
        tokenizer=tokenizer
    )  

    print('Done Training! Output Image Shape: ',output_image.shape) 

    img = Image.fromarray(output_image) 
    img.save('../images/txt2img.jpg') 

if __name__ == '__main__': 
    main() 