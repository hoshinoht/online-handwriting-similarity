"""
Script: generate_canonical_embeddings.py
Description:
    Computes the canonical structural embeddings for each class by averaging the 
    embeddings of correctly classified training samples. These canonical embeddings 
    serve as stable regression targets for the structural loss component in Stage 2 
    of training.

Usage:
    python generate_canonical_embeddings.py --data_dir data/casia --model_path best_model.pth
"""

import torch
from torch.utils.data import DataLoader, ConcatDataset
import os
import argparse
from tqdm import tqdm
import collections
import torch.nn.functional as F

from src.model import HandwritingModel
from src.preprocess import preprocess_stroke
from src.dataset import CASIADataset, CachedDataset

def load_cached_datasets(cache_dir):
    """
    Loads pre-processed .pt datasets from the specified cache directory.

    Args:
        cache_dir (str): Path to the directory containing cached shards.

    Returns:
        CachedDataset or None: The loaded dataset if found, otherwise None.
    """
    train_path = os.path.join(cache_dir, "train.pt")
    
    import glob
    def check_exists(path):
        if os.path.exists(path): return True
        base = path.replace('.pt', '')
        shards = glob.glob(f"{base}_shard*.pt")
        return len(shards) > 0
    
    if check_exists(train_path):
        return CachedDataset(train_path)
    return None

def generate_canonical(args):
    """
    Main execution function to compute and save canonical embeddings.

    Args:
        args (Namespace): Command-line arguments containing paths and configuration.
    """
    # 1. Device Configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # 2. Data Loading
    # Priority: processed/ directory > data_dir root
    print(f"Loading data from {args.data_dir}...")
    dataset = None
    
    processed_dir = os.path.join(args.data_dir, 'processed')
    if os.path.exists(processed_dir):
        dataset = load_cached_datasets(processed_dir)
    elif os.path.exists(args.data_dir):
        dataset = load_cached_datasets(args.data_dir)
         
    if not dataset:
        print("Error: Could not load cached training data. Please run preprocess_dataset.py first.")
        return

    # 3. Load Class Mapping
    map_path = os.path.join(args.data_dir, 'processed', 'class_map.pt')
    if not os.path.exists(map_path):
         map_path = os.path.join(args.data_dir, 'class_map.pt')
    
    if os.path.exists(map_path):
        tag_map = torch.load(map_path)
        num_classes = len(tag_map)
        print(f"Loaded class map with {num_classes} classes.")
    else:
        print("Error: class_map.pt not found.")
        return

    # 4. Model Loading
    model = HandwritingModel(num_classes=num_classes)
    if os.path.exists(args.model_path):
        print(f"Loading model weights from {args.model_path}...")
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Error: Model {args.model_path} not found.")
        return
        
    model.to(device)
    model.eval()
    
    # 5. Compute Embeddings
    print("Computing embeddings for all training samples...")
    # Disable shuffle to sequentialize memory access if possible, though unimportant for embedding gen
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Determine embedding dimension from model architecture
    emb_dim = model.fc_struct.out_features
    
    # Tensors to accumulate sum and count for computing the mean
    class_sums = torch.zeros(num_classes, emb_dim, device=device)
    class_counts = torch.zeros(num_classes, device=device)
    
    with torch.no_grad():
        for features, labels in tqdm(loader, desc="Processing Batches"):
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass: obtain structural output
            char_logits, struct_pred = model(features)
            
            # Robustness Filter:
            # We only use samples that the model classifies correctly.
            # This prevents noise or mislabeled data from corrupting the canonical embedding of a class.
            _, predicted = torch.max(char_logits, 1)
            mask = (predicted == labels)
            
            if mask.sum() > 0:
                valid_labels = labels[mask]
                valid_struct = struct_pred[mask]
                
                # Tensor Scatter Add: Efficiently sum embeddings by class index
                class_sums.index_add_(0, valid_labels, valid_struct)
                class_counts.index_add_(0, valid_labels, torch.ones_like(valid_labels, dtype=torch.float))
            
    # 6. Compute Centroids
    print("Computing centroids...")
    
    # Avoid division by zero for classes that might not have any correctly classified samples
    # (Though ideally, we should have some. If 0, the embedding remains 0 vector)
    class_counts = class_counts.unsqueeze(1).clamp(min=1.0)
    
    canonical_embeddings = class_sums / class_counts
    
    # 7. Save Results
    output_path = "models/canonical_embeddings.pt"
    torch.save(canonical_embeddings.cpu(), output_path)
    print(f"Saved canonical structural embeddings to {output_path}")
    print(f"Shape: {canonical_embeddings.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate canonical structural embeddings from trained model.")
    parser.add_argument('--data_dir', type=str, default='data/casia', help='Root directory of the dataset')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth', help='Path to the trained model checkpoint (.pth)')
    parser.add_argument('--batch_size', type=int, default=256, help='Inference batch size')
    args = parser.parse_args()
    
    generate_canonical(args)
