import importlib
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import torchvision.transforms as transforms
import sys
import gc
import yaml
import glob
import csv
import time




class config:
        def __init__(self):
            pass

class Args:
    """Simple class to hold arguments as attributes"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)



def find_latest_checkpoint(config_folder):
    """Find the most recent checkpoint in the config folder"""
    checkpoint_dir = os.path.join(config_folder, 'checkpoints')
    
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'NECTR_iter_*.pth'))
    
    if not checkpoint_files:
        # Try to find best or final checkpoint
        best_ckpt = os.path.join(checkpoint_dir, 'NECTR_best.pth')
        final_ckpt = os.path.join(checkpoint_dir, 'NECTR_final.pth')
        
        if os.path.exists(best_ckpt):
            return best_ckpt
        elif os.path.exists(final_ckpt):
            return final_ckpt
        else:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    # Extract iteration numbers and find the maximum
    iterations = []
    for ckpt_file in checkpoint_files:
        try:
            iter_num = int(os.path.basename(ckpt_file).split('_')[-1].split('.')[0])
            iterations.append((iter_num, ckpt_file))
        except:
            continue
    
    if iterations:
        iterations.sort(key=lambda x: x[0], reverse=True)
        return iterations[0][1]  # Return path of latest checkpoint
    
    raise FileNotFoundError(f"No valid checkpoint files found in {checkpoint_dir}")


def load_model(config_folder, checkpoint_path=None, window_rad=None, device=None, model_choice='model2'):
    """
    Load model from config folder and checkpoint.
    
    Args:
        config_folder: Path to config folder containing config.yml
        checkpoint_path: Path to checkpoint file. If None, uses latest checkpoint.
        device: torch device. If None, uses cuda if available, else cpu.
        model_choice: Which model implementation to load. Accepts:
            - 'model1' or 'NECTR_models1' to load `NECTR_model.NECTR_models1`
            - 'model2' or 'NECTR_models2' to load `NECTR_model.NECTR_models2` (default)
        Example:
            model = load_model("configs", model_choice='model1')
    
    Returns:
        model: Loaded model in eval mode
        model_args: Model arguments from config
        checkpoint_info: Dictionary with checkpoint metadata
    """
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load configuration
    config_path = os.path.join(config_folder, 'config.yml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_args = Args(config['model_arguments'])
    
    # Find checkpoint if not provided
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(config_folder)
        print(f"Using latest checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Dynamically import the model implementation
    # model_choice accepts values like 'model1' or 'model2', or direct module names 'NECTR_models1'/'NECTR_models2'
    module_map = {
        'model1': 'NECTR_models1',
        'model2': 'NECTR_models2',
        'NECTR_models1': 'NECTR_models1',
        'NECTR_models2': 'NECTR_models2',
        'unet': 'NECTR_models2',
    }
    chosen = module_map.get(model_choice, model_choice)
    try:
        mod = importlib.import_module(f'NECTR_model.{chosen}')
    except Exception as e:
        raise ImportError(f"Could not import model module 'NECTR_model.{chosen}': {e}")

    if not hasattr(mod, 'NECTR_denoiser'):
        raise AttributeError(f"Module 'NECTR_model.{chosen}' does not define 'NECTR_denoiser'")

    NECTR_denoiser_cls = getattr(mod, 'NECTR_denoiser')

    # Initialize model
    print(f"Initializing model '{chosen}' on {device}...")
    model = NECTR_denoiser_cls(model_args, device).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    
    # Set to eval mode
    model.eval()
    
    # Extract checkpoint info
    checkpoint_info = {
        'checkpoint_path': checkpoint_path,
        'iteration': checkpoint.get('iteration', 'unknown'),
        'loss': checkpoint.get('loss', None),
    }
    
    print(f"Model loaded successfully!")
    print(f"  Iteration: {checkpoint_info['iteration']}")
    if checkpoint_info['loss'] is not None:
        print(f"  Loss: {checkpoint_info['loss']:.6f}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    if window_rad is not None:
        old_win_rad = model.window_rad
        model.window_rad = window_rad
        print(f"  Set model window radius to: {window_rad} (from {old_win_rad})")
    else:
        print(f"  Using model window radius: {model.window_rad}")
    
    return model




def img2tensor(img,device):
    img = img.astype(np.float32)
    if img.ndim == 2:
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
    if img.ndim == 3:
        img = img.transpose(2,0,1)
        img = torch.from_numpy(img).unsqueeze(0).to(device)
    return img

def tensor2img(tensor):
    if tensor.shape[1] == 3:
        arr = tensor.cpu().squeeze().detach().numpy().astype(np.float32)
        arr = np.transpose(arr,(1,2,0))
    elif tensor.shape[1] == 1:
        arr = tensor.cpu().squeeze().squeeze().detach().numpy().astype(np.float32)
    del tensor
    gc.collect()
    return arr


def NECTR(noisy_image, model, reference_image, h = 1.0, device='cuda:0'):

    noisy_image_proc = img2tensor(noisy_image, device)  # Convert to tensor and move to device
    reference_image_proc = img2tensor(reference_image, device)  # Convert to tensor and move to device
    with torch.no_grad():
        
        denoised_image, D = model.SYMM_FORWARD(x = noisy_image_proc,
                           reference = reference_image_proc,
                                       sig =h,)
    denoised_image = tensor2img(denoised_image)  # Convert tensor back to numpy array
    # return denoised_image, D
    return denoised_image
