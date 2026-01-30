import struct
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import glob
from .preprocess import preprocess_stroke

class CachedDataset(Dataset):
    def __init__(self, pt_file_path):
        """
        Loads pre-processed tensors from a .pt file or multiple sharded files.
        Expects dict with 'features' and 'labels'.
        If pt_file_path is a directory or base prefix, we look for shards.
        """
        self.features = []
        self.labels = []
        
        # Check if direct file or shard pattern
        if os.path.exists(pt_file_path) and os.path.isfile(pt_file_path):
            # Single file case
            print(f"Loading cached dataset from {pt_file_path}...")
            data = torch.load(pt_file_path)
            self.features = data['features']
            self.labels = data['labels']
        else:
            # Check for shards: prefix_shard0.pt, prefix_shard1.pt...
            # The 'pt_file_path' coming in might be "data/processed/train.pt"
            # But actual files might be "data/processed/train_shard0.pt"
            
            # Assuming pt_file_path is like ".../train" or ".../train.pt"
            base_path = pt_file_path.replace('.pt', '')
            
            # Find all matching shards
            shard_files = sorted(glob.glob(f"{base_path}_shard*.pt"))
            
            if not shard_files:
                raise FileNotFoundError(f"No files found for {pt_file_path} or shards {base_path}_shard*.pt")
            
            print(f"Loading {len(shard_files)} shards from {base_path}...")
            
            all_features = []
            all_labels = []
            
            for sh in shard_files:
                print(f"  Loading {sh}...")
                data = torch.load(sh)
                all_features.append(data['features'])
                all_labels.append(data['labels'])
                
            self.features = torch.cat(all_features)
            self.labels = torch.cat(all_labels)
            
        print(f"Loaded {len(self.labels)} samples.")
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def parse_pot_file(filepath):
    """
    Parses a CASIA .pot file.
    Returns a list of samples: {'tag': int, 'points': np.ndarray}
    """
    samples = []
    with open(filepath, 'rb') as f:
        while True:
            # Read split header size (2 bytes)
            # According to common implementations, header is somewhat variable, 
            # but usually starts with sample size (2 bytes)
            # Actually, standard pot format:
            # Short: Sample Size (total bytes for this sample)
            # Short: Tag Code (GBK)
            # Short: Stroke Count
            # Then strokes...
            
            chunk = f.read(2)
            if not chunk:
                break
            
            sample_size = struct.unpack('<H', chunk)[0]
            
            if sample_size == 0:
                # Padding or error
                continue
                
            # Read remainder of sample
            # Sample size includes the size field itself? 
            # Usually yes. Let's read sample_size - 2 bytes.
            if sample_size <= 2:
                continue
                
            data = f.read(sample_size - 2)
            if len(data) < sample_size - 2:
                break
                
            # Parse header in data
            # tag_code (2 bytes), stroke_count (2 bytes)
            tag_code = struct.unpack('<H', data[0:2])[0]
            
            # Use GBK to decode tag_code if needed, but we might just keep the int
            # tag_char = data[0:2].decode('gbk', errors='ignore')
            
            stroke_count = struct.unpack('<H', data[2:4])[0]
            
            # Point data starts at offset 4? 
            # Actually, previous implementations show:
            # 2 bytes for tag code
            # 2 bytes for stroke count
            # Then strokes.
            # Each stroke ends with (-1, 0)? Or (-1, -1)?
            # Wait, Pot format usually interleaves strokes or just lists points.
            # Let's check common spec.
            # Points are 4 bytes: x (2 bytes short), y (2 bytes short).
            # End of stroke: (-1, 0)
            # End of sample: (-1, -1)
            
            # Let's read points.
            points_data = data[4:]
            
            points = []
            
            # Iterate through points
            # integer shorts
            num_shorts = len(points_data) // 2
            shorts = struct.unpack('<' + 'h' * num_shorts, points_data)
            
            current_stroke = []
            all_points = []
            
            i = 0
            while i < len(shorts) - 1:
                x = shorts[i]
                y = shorts[i+1]
                i += 2
                
                if x == -1 and y == 0:
                    # End of stroke
                    if current_stroke:
                        all_points.extend(current_stroke)
                        current_stroke = []
                    continue
                elif x == -1 and y == -1:
                    # End of sample (should be at end of data usually)
                    break
                else:
                    current_stroke.append([x, y])
            
            if current_stroke:
                all_points.extend(current_stroke)
                
            if len(all_points) > 0:
                samples.append({
                    'tag_code': tag_code,
                    'points': np.array(all_points, dtype=np.float32)
                })
                
    return samples

class CASIADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        :param root_dir: Directory containing .pot files
        :param transform: Function to apply to points (e.g. preprocess_stroke)
        """
        self.files = glob.glob(os.path.join(root_dir, '*.pot')) + glob.glob(os.path.join(root_dir, '**', '*.pot'), recursive=True)
        self.samples = []
        self.transform = transform
        
        if not self.files:
            print(f"No .pot files found in {root_dir}")
        else:
            print(f"Found {len(self.files)} .pot files. Loading...")
            for f in self.files:
                self.samples.extend(parse_pot_file(f))
            print(f"Loaded {len(self.samples)} samples.")
            
        # Create a mapping from tag_code to class index
        all_tags = sorted(list(set(s['tag_code'] for s in self.samples)))
        self.tag_to_idx = {tag: i for i, tag in enumerate(all_tags)}
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        points = sample['points']
        tag = sample['tag_code']
        label = self.tag_to_idx[tag]
        
        if self.transform:
            points = self.transform(points)
            
        # Return points and label
        # points is (L, 7) or similar
        # label is int
        
        return torch.tensor(points, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
