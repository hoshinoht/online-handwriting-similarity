import os
import argparse
import torch
import numpy as np
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from src.dataset import CASIADataset
from src.preprocess import preprocess_stroke

# Define standard directories
TRAIN_DIRS = ["Pot1.0Train", "Pot1.2Train"]
TEST_DIRS = ["Pot1.0Test", "Pot1.2Test"]

def process_chunk_wrapper(chunk_data):
    """
    Worker function to preprocess a chunk of samples.
    """
    features_list = []
    labels_list = []
    
    for points, label_idx in chunk_data:
        ft = preprocess_stroke(points)
        features_list.append(torch.tensor(ft, dtype=torch.float32))
        labels_list.append(torch.tensor(label_idx, dtype=torch.long))
        
    return torch.stack(features_list), torch.stack(labels_list)

def save_dataset_sharded_with_pool(pool, raw_samples_list, output_base, num_workers=None, samples_per_shard=200000):
    if not raw_samples_list:
        print(f"No data for {output_base}")
        return
        
    total_samples = len(raw_samples_list)
    print(f"Processing {total_samples} samples for {output_base}...")
    
    num_shards = (total_samples + samples_per_shard - 1) // samples_per_shard
    
    for shard_idx in range(num_shards):
        output_path = f"{output_base}_shard{shard_idx}.pt"
        
        # Check if shard already exists
        if os.path.exists(output_path):
            print(f"  Shard {shard_idx+1}/{num_shards}: {output_path} already exists. Skipping.")
            continue
            
        start = shard_idx * samples_per_shard
        end = min((shard_idx + 1) * samples_per_shard, total_samples)
        
        current_batch = raw_samples_list[start:end]
        
        # Sub-divide current batch for workers
        worker_chunk_size = max(1, len(current_batch) // (num_workers * 2)) 
        
        # Create chunks for workers
        worker_inputs = [
            current_batch[i:i + worker_chunk_size] 
            for i in range(0, len(current_batch), worker_chunk_size)
        ]
        
        print(f"  Shard {shard_idx+1}/{num_shards}: Processing samples {start} to {end}...")
        
        shard_features = []
        shard_labels = []
        
        # Use existing pool
        results = list(tqdm(pool.imap(process_chunk_wrapper, worker_inputs), total=len(worker_inputs), leave=False))
            
        for f, l in results:
            shard_features.append(f)
            shard_labels.append(l)
            
        # Stack shard
        features_tensor = torch.cat(shard_features)
        labels_tensor = torch.cat(shard_labels)
        
        print(f"    Saving shard to {output_path}...")
        torch.save({
            'features': features_tensor,
            'labels': labels_tensor
        }, output_path)
        
        # Explicit delete to free memory
        del features_tensor
        del labels_tensor
        del results
        del shard_features
        del shard_labels
        
    print("Done.")

def main(args):
    base_dir = args.data_dir
    os.makedirs(args.output_dir, exist_ok=True)
    num_workers = args.num_workers if args.num_workers > 0 else cpu_count()
    print(f"Using {num_workers} worker processes.")
    
    # 1. Load everything generically first to get global tag map
    train_datasets = []
    test_datasets = []
    
    print("Scanning for .pot files...")
    
    for d in TRAIN_DIRS:
        p = os.path.join(base_dir, d)
        if os.path.exists(p):
            train_datasets.append(CASIADataset(p, transform=None))
            
    for d in TEST_DIRS:
        p = os.path.join(base_dir, d)
        if os.path.exists(p):
            test_datasets.append(CASIADataset(p, transform=None))
            
    if not train_datasets and not test_datasets:
        print("No datasets found.")
        return

    # 2. Build Global Tag Map
    print("Building global class mapping...")
    all_ds = train_datasets + test_datasets
    all_tags = set()
    for ds in all_ds:
        for s in ds.samples:
            all_tags.add(s['tag_code'])
            
    sorted_tags = sorted(list(all_tags))
    tag_to_idx = {tag: i for i, tag in enumerate(sorted_tags)}
    print(f"Found {len(sorted_tags)} unique classes.")
    
    # Save tag map
    map_path = os.path.join(args.output_dir, "class_map.pt")
    torch.save(tag_to_idx, map_path)
    print(f"Saved class map to {map_path}")
    
    # 3. Collect Raw Samples
    def collect_raw_samples(datasets):
        raw_list = []
        for ds in datasets:
            for s in ds.samples:
                try:
                    label_idx = tag_to_idx[s['tag_code']]
                    raw_list.append((s['points'], label_idx))
                except KeyError:
                    pass # Should not happen given we just built the map
        return raw_list

    print("Collecting raw training samples...")
    train_raw = collect_raw_samples(train_datasets)
    
    print("Collecting raw testing samples...")
    test_raw = collect_raw_samples(test_datasets)
    
    # Initialize Pool ONCE here
    print(f"Initializing multiprocessing pool with {num_workers} workers...")
    with Pool(processes=num_workers) as pool:
        # Save base + suffix
        save_dataset_sharded_with_pool(pool, train_raw, os.path.join(args.output_dir, "train"), num_workers)
        save_dataset_sharded_with_pool(pool, test_raw, os.path.join(args.output_dir, "test"), num_workers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/casia', help='Input directory containing Pot folders')
    parser.add_argument('--output_dir', type=str, default='data/processed', help='Output directory for .pt files')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers. 0 = all cores')
    args = parser.parse_args()
    main(args)
