"""
Script: lottery_ticket.py
Description:
    Orchestrates the Lottery Ticket Hypothesis (LTH) pruning and training loop.
    Iteratively prunes weights, rewinds them to initialization (while keeping the mask),
    and retrains to find highly sparse subnetworks that maintain accuracy.

Usage:
    python lottery_ticket.py --data_dir data/casia/processed --batch_size 1024
"""

import torch
import torch.nn.utils.prune as prune
import os
import argparse
import copy
import sys
import subprocess

from train import train
from src.model import HandwritingModel

def get_model(num_classes, arch='transformer'):
    """Factory method to instantiate the model structure."""
    return HandwritingModel(num_classes=num_classes, arch=arch)

def prune_model(model, amount=0.2):
    """
    Applies global unstructured pruning to all Conv1d and Linear layers.
    
    Args:
        model (nn.Module): The model to prune.
        amount (float): Fraction of weights to prune (0.0 - 1.0).
        
    Returns:
        nn.Module: The model with pruning masks applied.
    """
    print(f"[-] Pruning {amount*100}% of weights...")
    
    parameters_to_prune = []
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))
        elif isinstance(module, torch.nn.MultiheadAttention):
            parameters_to_prune.append((module, 'in_proj_weight'))
            parameters_to_prune.append((module.out_proj, 'weight'))

    # Apply global L1 unstructured pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    
    # Calculate sparsity statistics
    total_zeros = 0
    total_params = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.Linear):
            total_zeros += torch.sum(module.weight == 0)
            total_params += module.weight.nelement()
        elif isinstance(module, torch.nn.MultiheadAttention):
            total_zeros += torch.sum(module.in_proj_weight == 0)
            total_params += module.in_proj_weight.nelement()
            total_zeros += torch.sum(module.out_proj.weight == 0)
            total_params += module.out_proj.weight.nelement()
            
    print(f"    Global Sparsity: {100. * total_zeros / total_params:.2f}%")
    return model

def make_pruning_permanent(model):
    """
    Removes the pruning re-parameterization, making the zeroed weights permanent.
    Used for final export.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.Linear):
            if hasattr(module, 'weight_orig'):
                prune.remove(module, 'weight')
        elif isinstance(module, torch.nn.MultiheadAttention):
            if hasattr(module, 'in_proj_weight_orig'):
                prune.remove(module, 'in_proj_weight')
            if hasattr(module.out_proj, 'weight_orig'):
                prune.remove(module.out_proj, 'weight')
    return model

def rewind_weights(model, init_state_dict):
    """
    Rewinds weights to their initial values from 'init_model.pth' while PRESERVING
    the pruning mask. This is the core 'winning ticket' mechanic.
    
    Args:
        model (nn.Module): The pruned model (current state).
        init_state_dict (dict): The state dictionary of the unpruned initialization.
    """
    print("[<] Rewinding weights to init_model.pth...")
    
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.Linear):
                # Scenario 1: Layer is pruned (has weight_orig and weight_mask)
                if hasattr(module, "weight_orig"):
                    if f"{name}.weight" in init_state_dict:
                        module.weight_orig.copy_(init_state_dict[f"{name}.weight"])
                    else:
                        print(f"Warning: {name}.weight not found in init_state_dict.")

                # Scenario 2: Unpruned layer in a module list (if any)
                else:
                     if f"{name}.weight" in init_state_dict:
                         module.weight.copy_(init_state_dict[f"{name}.weight"])

                # Rewind Bias
                if module.bias is not None and f"{name}.bias" in init_state_dict:
                    module.bias.copy_(init_state_dict[f"{name}.bias"])

            elif isinstance(module, torch.nn.MultiheadAttention):
                # in_proj_weight
                key = f"{name}.in_proj_weight"
                if hasattr(module, "in_proj_weight_orig"):
                    if key in init_state_dict:
                        module.in_proj_weight_orig.copy_(init_state_dict[key])
                elif key in init_state_dict:
                    module.in_proj_weight.copy_(init_state_dict[key])
                # in_proj_bias
                if module.in_proj_bias is not None and f"{name}.in_proj_bias" in init_state_dict:
                    module.in_proj_bias.copy_(init_state_dict[f"{name}.in_proj_bias"])
                    
        # Rewind other parameters (e.g. BatchNorm, GRU non-weight params)
        for name, param in model.named_parameters():
            if "weight_orig" not in name and "weight_mask" not in name:
                if name in init_state_dict:
                    param.copy_(init_state_dict[name])

def run_lottery(args):
    """
    Main execution loop for LTH.
    """
    # Prerequisites Checks
    if not os.path.exists("models/init_model.pth"):
        print("Error: 'models/init_model.pth' missing. Train the base model first!")
        return
        
    if not os.path.exists("models/best_model.pth"):
        print("Error: 'models/best_model.pth' missing. Train the base model first!")
        return

    # Hyperparameters
    ROUNDS = 5
    PRUNE_RATE = 0.2
    
    # 1. Determine Input Dimensions
    if os.path.exists(os.path.join(args.data_dir, 'processed', 'class_map.pt')):
        map_path = os.path.join(args.data_dir, 'processed', 'class_map.pt')
    else:
        map_path = os.path.join(args.data_dir, 'class_map.pt')
        
    tag_map = torch.load(map_path)

    # 2. Load the Reference Model (The Teacher/Baseline)
    print("Loading best model to identify winning tickets...")
    best_state = torch.load("models/best_model.pth", map_location='cpu')
    arch = 'transformer' if any(k.startswith('transformer_encoder.') for k in best_state) else 'gru'
    model = get_model(len(tag_map), arch=arch)
    model.load_state_dict(best_state)
    
    # 3. Load Initialization (For Rewinding)
    init_state = torch.load("models/init_model.pth", map_location='cpu')

    # 4. Pruning Loop
    for round_idx in range(ROUNDS):
        print(f"\n=== Pruning Round {round_idx + 1}/{ROUNDS} ===")
        
        # A. Prune
        model = prune_model(model, amount=PRUNE_RATE)
        
        # B. Rewind
        rewind_weights(model, init_state)
        
        # C. Save Intermediate Checkpoint
        # This checkpoint has the sparse mask active and weights set to Init values.
        ckpt_name = f"models/lottery_round_{round_idx+1}.pth"
        torch.save(model.state_dict(), ckpt_name)
        
        # D. Retrain
        # We spawn a subprocess to avoid memory leaks and ensure clean state context.
        print(f"Starting retraining for Round {round_idx+1}...")
        
        cmd = [
            sys.executable, "train.py",
            "--data_dir", args.data_dir,
            "--batch_size", str(args.batch_size),
            "--epochs", "15", # Fixed epoch count for retraining
            "--resume", ckpt_name
        ]
        subprocess.check_call(cmd)
        
        # E. Prepare Next Round
        # Load the *result* of the retraining (which overwrote 'best_model.pth')
        # to use as the base for the next level of pruning.
        print(f"Round {round_idx+1} complete. Loading best model for next round...")
        
        # Ensure model has correct structure before loading sparse dict
        # We re-apply identity masks to all pruned layers
        best_path = "models/best_model.pth"
        if os.path.exists(best_path):
             state_dict = torch.load(best_path, map_location='cpu')
             round_arch = 'transformer' if any(k.startswith('transformer_encoder.') for k in state_dict) else 'gru'
             model = get_model(len(tag_map), arch=round_arch)

             for key in state_dict.keys():
                 if key.endswith('_mask'):
                     # e.g. 'layer.weight_mask' → param='weight', path='layer'
                     # e.g. 'layer.in_proj_weight_mask' → param='in_proj_weight', path='layer'
                     parts = key.rsplit('.', 1)
                     layer_name, mask_name = parts[0], parts[1]
                     param_name = mask_name.replace('_mask', '')
                     module = model
                     for part in layer_name.split('.'):
                         module = getattr(module, part)
                     prune.identity(module, param_name)
             
             model.load_state_dict(state_dict)
        else:
             print("Error: best_model.pth not found after training!")
             break 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Lottery Ticket Hypothesis Pruning")
    parser.add_argument('--data_dir', type=str, default='data/casia/processed', help='Path to processed data')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for retraining')
    args = parser.parse_args()
    
    run_lottery(args)
