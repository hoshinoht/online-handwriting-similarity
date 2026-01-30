import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import numpy as np
import os
import argparse

from src.model import HandwritingModel
from src.loss import CombinedLoss
from src.preprocess import preprocess_stroke
from src.dataset import CASIADataset

# dataset directories for casia
train_directories = [
  "Pot1.0Train",
  "Pot1.2Train",
]

test_directories = [
  "Pot1.0Test",
  "Pot1.2Test",
]

def load_datasets(base_dir):
    """
    Loads training and testing datasets from specific subdirectories.
    """
    train_datasets = []
    test_datasets = []
    
    # Load Training Data
    print("Loading Training Data...")
    for d in train_directories:
        path = os.path.join(base_dir, d)
        if os.path.exists(path):
            print(f"  Loading from {path}...")
            # Note: transform is applied on output.
            ds = CASIADataset(path, transform=preprocess_stroke)
            train_datasets.append(ds)
        else:
            print(f"  Warning: Directory {path} not found.")

    # Load Testing Data
    print("Loading Testing Data...")
    for d in test_directories:
        path = os.path.join(base_dir, d)
        if os.path.exists(path):
            print(f"  Loading from {path}...")
            ds = CASIADataset(path, transform=preprocess_stroke)
            test_datasets.append(ds)
        else:
            print(f"  Warning: Directory {path} not found.")
            
    if not train_datasets and not test_datasets:
        return None, None
        
    full_train_dataset = ConcatDataset(train_datasets) if train_datasets else None
    full_test_dataset = ConcatDataset(test_datasets) if test_datasets else None
    
    return full_train_dataset, full_test_dataset

def train(args):
    # Hyperparameters
    BATCH_SIZE = 16
    LR = 0.001
    EPOCHS = args.epochs
    
    train_dataset = None
    test_dataset = None
    
    if args.data_dir and os.path.exists(args.data_dir):
        # Check if we should use the structured loading
        # If the subdirectories exist in args.data_dir, use specific loading
        # Otherwise, fall back to loading the entire folder as one dataset
        
        # Check if any expected subdirs exist
        has_subdirs = any(os.path.exists(os.path.join(args.data_dir, d)) for d in train_directories + test_directories)
        
        if has_subdirs:
            print(f"Detected structured CASIA dataset in {args.data_dir}")
            train_dataset, test_dataset = load_datasets(args.data_dir)
        else:
            print(f"Loading generic dataset from {args.data_dir}")
            train_dataset = CASIADataset(args.data_dir, transform=preprocess_stroke)
    
    if not train_dataset:
        print("No valid training data found. Using Dummy Dataset.")
        train_dataset = DummyDataset()
        NUM_CLASSES = 10
    else:
        # Determine num classes. 
        # Since we might have multiple datasets, we need a consistent mapping.
        # Currently, each CASIADataset creates its own mapping based on the files it sees.
        # This IS A PROBLEM if we use ConcatDataset. 
        # Different datasets might map the same tag to different indices if the set of tags differs!
        # Ideally, we should scan all files first to build a global tag map.
        # OR: CASIADataset needs to accept a predefined tag map.
        pass 
        
    # FIX: Ensure consistent class mapping across datasets
    # Create a global tag map from all datasets (train + test)
    all_datasets = []
    if isinstance(train_dataset, ConcatDataset):
        all_datasets.extend(train_dataset.datasets)
    elif train_dataset:
         all_datasets.append(train_dataset)
         
    if isinstance(test_dataset, ConcatDataset):
        all_datasets.extend(test_dataset.datasets)
    elif test_dataset:
        all_datasets.append(test_dataset)
        
    if all_datasets and not isinstance(all_datasets[0], DummyDataset):
        # Collect all unique tags
        all_tags = set()
        for ds in all_datasets:
            # Assuming ds has .samples with 'tag_code'
             for s in ds.samples:
                 all_tags.add(s['tag_code'])
        
        sorted_tags = sorted(list(all_tags))
        tag_to_idx = {tag: i for i, tag in enumerate(sorted_tags)}
        NUM_CLASSES = len(sorted_tags)
        print(f"Global Class Map: {NUM_CLASSES} classes found.")
        
        # Update all datasets to use this map
        for ds in all_datasets:
            ds.tag_to_idx = tag_to_idx
    elif isinstance(train_dataset, DummyDataset):
        NUM_CLASSES = 10

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) if test_dataset else None
    
    # Model
    model = HandwritingModel(num_classes=NUM_CLASSES)
    
    # Loss
    criterion = CombinedLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    model.train()
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (features, labels) in enumerate(train_loader):
            # Features: (B, L, 7)
            # Labels: (B,)
            
            char_logits, struct_pred = model(features)
            
            # Dummy struct target for now
            struct_target = torch.randn_like(struct_pred) 
            
            loss, loss_dict = criterion(char_logits, labels, struct_pred, struct_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Train Loss: {avg_loss:.4f}")
        
        # Evaluation
        if test_loader:
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for features, labels in test_loader:
                    char_logits, struct_pred = model(features)
                    # For eval, just check CE loss and accuracy
                    # We need to construct struct_target to compute full loss, or just skip it
                    struct_target = torch.randn_like(struct_pred) 
                    loss, _ = criterion(char_logits, labels, struct_pred, struct_target)
                    test_loss += loss.item()
                    
                    _, predicted = torch.max(char_logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            print(f"Epoch {epoch+1} Test Loss: {test_loss/len(test_loader):.4f} | Acc: {100 * correct / total:.2f}%")
            model.train()

class DummyDataset(Dataset):
    def __init__(self, num_samples=100, num_classes=10):
        self.num_samples = num_samples
        self.num_classes = num_classes
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        points = np.cumsum(np.random.randn(20, 2), axis=0)
        features = preprocess_stroke(points)
        label = np.random.randint(0, self.num_classes)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Updated default to relative 'data/casia' to match user usage pattern slightly better 
    # if they run from project root and have data/casia structure.
    # But user said "locate these datasets in the /data folder" and structure is data/casia/Pot...
    parser.add_argument('--data_dir', type=str, default='data/casia', help='Path to directory containing .pot files or structured subdirs')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    args = parser.parse_args()
    
    train(args)
