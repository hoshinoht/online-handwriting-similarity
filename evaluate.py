import torch
from torch.utils.data import DataLoader, ConcatDataset
import os
import argparse
from tqdm import tqdm
import glob

from src.model import HandwritingModel
from src.preprocess import preprocess_stroke
from src.dataset import CASIADataset, CachedDataset

# Directories dependent on the user's data structure
train_directories = [
  "Pot1.0Train",
  "Pot1.2Train",
]

test_directories = [
  "Pot1.0Test",
  "Pot1.2Test",
]

def load_cached_test(cache_dir):
    test_path = os.path.join(cache_dir, "test.pt")
    map_path = os.path.join(cache_dir, "class_map.pt")
    
    # Check for direct files OR valid shards
    def check_exists(path):
        if os.path.exists(path): return True
        base = path.replace('.pt', '')
        shards = glob.glob(f"{base}_shard*.pt")
        return len(shards) > 0
    
    if check_exists(test_path) and os.path.exists(map_path):
        print(f"Found cached test set in {cache_dir}")
        test_ds = CachedDataset(test_path)
        tag_map = torch.load(map_path)
        return test_ds, len(tag_map)
    return None, None

def load_datasets(base_dir):
    """
    Loads training and testing datasets to establish global class mapping.
    """
    train_datasets = []
    test_datasets = []
    
    print("Loading datasets to build class mapping...")
    
    # We MUST load training data to get the complete set of classes,
    # otherwise the model's output layer size won't match the checkpoint.
    for d in train_directories:
        path = os.path.join(base_dir, d)
        if os.path.exists(path):
            ds = CASIADataset(path, transform=preprocess_stroke)
            train_datasets.append(ds)

    for d in test_directories:
        path = os.path.join(base_dir, d)
        if os.path.exists(path):
            ds = CASIADataset(path, transform=preprocess_stroke)
            test_datasets.append(ds)
            
    if not train_datasets and not test_datasets:
        raise ValueError("No datasets found!")
        
    full_train_dataset = ConcatDataset(train_datasets) if train_datasets else None
    full_test_dataset = ConcatDataset(test_datasets) if test_datasets else None
    
    return full_train_dataset, full_test_dataset

def build_tag_mapping(train_dataset, test_dataset):
    """
    Collects all unique tags from both train and test sets to ensure consistent indices.
    """
    all_datasets = []
    if isinstance(train_dataset, ConcatDataset):
        all_datasets.extend(train_dataset.datasets)
    elif train_dataset:
         all_datasets.append(train_dataset)
         
    if isinstance(test_dataset, ConcatDataset):
        all_datasets.extend(test_dataset.datasets)
    elif test_dataset:
        all_datasets.append(test_dataset)
        
    all_tags = set()
    for ds in all_datasets:
         for s in ds.samples:
             all_tags.add(s['tag_code'])
    
    sorted_tags = sorted(list(all_tags))
    tag_to_idx = {tag: i for i, tag in enumerate(sorted_tags)}
    print(f"Global Class Map: {len(sorted_tags)} classes.")
    
    # Apply mapping to all datasets
    for ds in all_datasets:
        ds.tag_to_idx = tag_to_idx
        
    return len(sorted_tags)

def evaluate(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    test_ds = None
    num_classes = 0
    
    # Try cache first
    processed_dir = os.path.join(args.data_dir, 'processed')
    if os.path.exists(processed_dir):
        test_ds, num_classes = load_cached_test(processed_dir)
    
    if not test_ds:
         # Try direct dir as cache
        test_ds, num_classes = load_cached_test(args.data_dir)
        
    if not test_ds:
        print("Using raw .pot files (slow)...")
        train_ds, test_ds = load_datasets(args.data_dir)
        if test_ds is None:
            print("Error: No test data found.")
            return
        # 2. Build Mapping
        num_classes = build_tag_mapping(train_ds, test_ds)
    
    # 3. Model
    model = HandwritingModel(num_classes=num_classes)
    
    # 4. Load Weights
    if os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}...")
        try:
            state_dict = torch.load(args.model_path, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Tip: Ensure the model was saved with 'torch.save(model.state_dict(), ...)'")
            return
    else:
        print(f"Error: Model file {args.model_path} not found.")
        return

    model.to(device)
    model.eval()
    
    # 5. Evaluate
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    correct = 0
    total = 0
    
    print("Starting evaluation...")
    with torch.no_grad():
        for features, labels in tqdm(test_loader):
            features = features.to(device)
            labels = labels.to(device)
            
            char_logits, struct_pred = model(features)
            
            _, predicted = torch.max(char_logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f"Total Samples: {total}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/casia', help='Path to data root')
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pth checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()
    
    evaluate(args)
