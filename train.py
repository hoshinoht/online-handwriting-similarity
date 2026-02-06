"""
Script: train.py
Description:
    Main training script for the Online Handwriting Similarity project.
    Handles data loading, model initialization, training loops, validation, 
    and checkpoint saving. Supports resumed training (Stage 2/3) and 
    Structural Loss via canonical embeddings.

Usage:
    python train.py --data_dir data/casia --batch_size 256 --epochs 15
"""

from src.augment import augment_stroke
from src.dataset import CASIADataset, CachedDataset
from src.preprocess import preprocess_stroke
from src.loss import CombinedLoss
from src.model import HandwritingModel
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import numpy as np
import os
import argparse
import glob
from tqdm import tqdm
import random


def mixup_data(x, y, alpha=0.2, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam, struct_pred, struct_target):
    # This is tricky with combined loss.
    # struct_target also needs mixing?
    # For now, let's just mix classification targets.
    # We will call criterion twice? No.
    # Let's manually interpolate the loss.
    # But criterion returns dict.

    # Actually, simpler to just modify the criterion or handling loop.
    return None  # Placeholder, will handle in loop.


# Enable TensorFloat-32 for RTX 4000 series
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# AMP Gradient Scaler Import
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic algorithms can slow down, so we stick to seed setting + benchmark
    # torch.use_deterministic_algorithms(True)


# --- Configuration Constants ---
TRAIN_DIRS = ["Pot1.0Train", "Pot1.2Train"]
TEST_DIRS = ["Pot1.0Test", "Pot1.2Test"]


def load_cached_datasets(cache_dir, augment_train=False):
    """
    Attempts to load pre-processed .pt dataset shards from a directory.

    Args:
        cache_dir (str): Directory searching for 'train.pt', 'test.pt', or shards.
        augment_train (bool): Whether to apply online augmentation to training data.

    Returns:
        tuple: (train_dataset, test_dataset, num_classes) or (None, None, None)
    """
    train_path = os.path.join(cache_dir, "train.pt")
    test_path = os.path.join(cache_dir, "test.pt")
    map_path = os.path.join(cache_dir, "class_map.pt")

    def check_exists(path):
        if os.path.exists(path):
            return True
        # Check for sharded files (e.g. train_shard0.pt)
        base = path.replace('.pt', '')
        shards = glob.glob(f"{base}_shard*.pt")
        return len(shards) > 0

    if check_exists(train_path) and check_exists(test_path) and os.path.exists(map_path):
        print(f"Found cached datasets in {cache_dir}")
        train_ds = CachedDataset(train_path, augment=augment_train)
        test_ds = CachedDataset(test_path, augment=False)  # Never augment test
        tag_map = torch.load(map_path)
        return train_ds, test_ds, len(tag_map)
    else:
        return None, None, None


def load_datasets(base_dir):
    """
    Loads raw .pot datasets from specific CASIA subdirectories.
    Used as a fallback if cached .pt files are not found.

    Args:
        base_dir (str): Root directory containing 'Pot1.0Train', etc.

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    train_datasets = []
    test_datasets = []

    print("Loading Training Data...")
    for d in TRAIN_DIRS:
        path = os.path.join(base_dir, d)
        if os.path.exists(path):
            print(f"  Loading from {path}...")
            train_datasets.append(CASIADataset(
                path, transform=preprocess_stroke))
        else:
            print(f"  Warning: Directory {path} not found.")

    print("Loading Testing Data...")
    for d in TEST_DIRS:
        path = os.path.join(base_dir, d)
        if os.path.exists(path):
            print(f"  Loading from {path}...")
            test_datasets.append(CASIADataset(
                path, transform=preprocess_stroke))
        else:
            print(f"  Warning: Directory {path} not found.")

    if not train_datasets and not test_datasets:
        return None, None

    full_train = ConcatDataset(train_datasets) if train_datasets else None
    full_test = ConcatDataset(test_datasets) if test_datasets else None

    return full_train, full_test


def train(args):
    """
    Main training loop.
    """
    if args.seed is not None:
        set_seed(args.seed)

    BATCH_SIZE = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs

    # 1. Device Setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    print(f"Learning Rate: {LR}")

    # 2. Data Loading
    train_dataset = None
    test_dataset = None
    NUM_CLASSES = 0

    if args.data_dir and os.path.exists(args.data_dir):
        # A. Try Loading Cache
        processed_dir = os.path.join(args.data_dir, 'processed')
        if os.path.exists(processed_dir):
            train_dataset, test_dataset, NUM_CLASSES = load_cached_datasets(
                processed_dir, augment_train=True)

        # Fallback: Check for data/processed (sibling to casia)
        if not train_dataset:
            sibling_processed = os.path.join(os.path.dirname(
                args.data_dir.rstrip('/\\')), 'processed')
            if os.path.exists(sibling_processed):
                print(f"Checking detected sibling cache: {sibling_processed}")
                train_dataset, test_dataset, NUM_CLASSES = load_cached_datasets(
                    sibling_processed, augment_train=True)

            # Hardcoded common fallback
            if not train_dataset and os.path.exists('data/processed'):
                train_dataset, test_dataset, NUM_CLASSES = load_cached_datasets(
                    'data/processed', augment_train=True)

        if not train_dataset:
            train_dataset, test_dataset, NUM_CLASSES = load_cached_datasets(
                args.data_dir, augment_train=True)

        if train_dataset:
            print(f"Using cached data. Num Classes: {NUM_CLASSES}")
        else:
            # B. Fallback to Raw Loading
            has_subdirs = any(os.path.exists(os.path.join(args.data_dir, d))
                              for d in TRAIN_DIRS + TEST_DIRS)

            has_subdirs = any(os.path.exists(os.path.join(args.data_dir, d))
                              for d in TRAIN_DIRS + TEST_DIRS)

            if has_subdirs:
                print(f"Detected structured CASIA dataset in {args.data_dir}")
                train_dataset, test_dataset = load_datasets(args.data_dir)
                # Apply augmentation to training part only (we need to modify load_datasets or wrapper)
                # Ideally, load_datasets should allow passing augment flag.
                # For now, let's assume load_datasets uses CASIADataset which defaults to False.
                # We can iterate and set .augment = True for train datasets.
                if train_dataset:
                    if isinstance(train_dataset, ConcatDataset):
                        for ds in train_dataset.datasets:
                            if hasattr(ds, 'augment'):
                                ds.augment = True
                    elif hasattr(train_dataset, 'augment'):
                        train_dataset.augment = True

            else:
                print(f"Loading generic dataset from {args.data_dir}")
                train_dataset = CASIADataset(
                    args.data_dir, transform=preprocess_stroke, augment=True)

    # 3. Handle Missing Data (Strict)
    if not train_dataset:
        raise ValueError(
            f"CRITICAL: No training data found in {args.data_dir}. Halting.")

    if not NUM_CLASSES:
        # Need to calc num classes if not cached
        pass

    # DummyDataset class was removed, so we don't use it.

    # 4. Global Class Mapping (for Raw Datasets)
    if not NUM_CLASSES and train_dataset:
        all_datasets = []
        if isinstance(train_dataset, ConcatDataset):
            all_datasets.extend(train_dataset.datasets)
        elif train_dataset:
            all_datasets.append(train_dataset)
        if isinstance(test_dataset, ConcatDataset):
            all_datasets.extend(test_dataset.datasets)
        elif test_dataset:
            all_datasets.append(test_dataset)

        if all_datasets:
            all_tags = set()
            for ds in all_datasets:
                if hasattr(ds, 'samples'):
                    for s in ds.samples:
                        all_tags.add(s['tag_code'])

            if all_tags:
                sorted_tags = sorted(list(all_tags))
                tag_to_idx = {tag: i for i, tag in enumerate(sorted_tags)}
                NUM_CLASSES = len(sorted_tags)
                print(f"Global Class Map: {NUM_CLASSES} classes found.")
                for ds in all_datasets:
                    if hasattr(ds, 'tag_to_idx'):
                        ds.tag_to_idx = tag_to_idx

    # 5. DataLoaders
    # Pin memory helps data transfer, but can be tricky on MPS in older versions.
    use_pin_memory = (device.type != 'mps')
    num_workers = min(4, os.cpu_count() or 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=(num_workers > 0)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    ) if test_dataset else None

    # 6. Model & Optimization
    model = HandwritingModel(num_classes=NUM_CLASSES,
                             hidden_dim=args.hidden_dim, dropout=0.3)
    criterion = CombinedLoss(lambda_struct=args.lambda_struct)

    # Optim: AdamW
    # Scheduler: CosineAnnealingWarmRestarts
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    model.to(device)
    model.train()

    # AMP Scaler
    use_amp = (device.type == 'cuda')
    scaler = GradScaler(enabled=use_amp)

    # Scheduler: Reduce LR on plateau (Safer than restarts for convergence)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    # Save initialization for Lottery Ticket Rewinding
    if not args.resume:  # Only save init if starting fresh
        torch.save(model.state_dict(), "models/init_model.pth")
        print("Saved 'models/init_model.pth' for Lottery Ticket rewinding.")

    best_acc = 0.0

    if args.resume:
        resume_path = args.resume if isinstance(
            args.resume, str) else "models/best_model.pth"
        if os.path.exists(resume_path):
            print(f"Resuming training from {resume_path}...")
            state_dict = torch.load(resume_path, map_location=device)

            # Pruning Detection: Apply identity masks if loading a sparse model
            for key in state_dict.keys():
                if 'weight_mask' in key:
                    layer_name = key.replace('.weight_mask', '')
                    module = model
                    for part in layer_name.split('.'):
                        module = getattr(module, part)

                    import torch.nn.utils.prune as prune
                    prune.identity(module, 'weight')
                    print(
                        f"  [Resume] Applied pruning structure to {layer_name}")

            model.load_state_dict(state_dict)
        else:
            print(
                f"Warning: Resume file {resume_path} not found. Starting from scratch.")

    # 8. Canonical Embeddings (Stage 2 Support)
    # Check for Canonical Embeddings
    canonical_emb_path = "models/canonical_embeddings.pt"
    canonical_emb = None
    if os.path.exists(canonical_emb_path):
        print(
            f"Found {canonical_emb_path}. Using Canonical Embeddings for Structural Loss.")
        canonical_emb = torch.load(canonical_emb_path, map_location=device)
        if canonical_emb.shape[0] != NUM_CLASSES:
            print(
                f"Warning: Canvas mismatch ({canonical_emb.shape[0]} vs {NUM_CLASSES}). Fallback to random.")
            canonical_emb = None
    else:
        print("Note: 'models/canonical_embeddings.pt' not found. Using random noise for Structural Loss (Stage 1).")

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    # 9. Training Loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch_idx, (features, labels) in enumerate(pbar):
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Debug: Check labels before loss
            if torch.max(labels) >= NUM_CLASSES or torch.min(labels) < 0:
                print(f"\n[DEBUG] Label mismatch detected!")
                print(f"  NUM_CLASSES: {NUM_CLASSES}")
                print(f"  Max Label: {torch.max(labels).item()}")
                print(f"  Min Label: {torch.min(labels).item()}")
                invalid = labels[(labels >= NUM_CLASSES) | (labels < 0)]
                print(f"  Invalid Labels Sample: {invalid[:10].tolist()}")
                raise ValueError(
                    "CRITICAL: Training halted due to invalid labels.")

            # AMP / Standard Context
            if device.type == 'cuda':
                ctx = autocast('cuda', enabled=True)
            else:
                ctx = torch.autocast(device_type=device.type, enabled=False) if hasattr(
                    torch, 'autocast') else torch.no_grad()

            if use_amp:
                with autocast('cuda'):
                    # MixUp Logic
                    # Turn off MixUp in last 5 epochs
                    use_mixup = (epoch < EPOCHS - 5)
                    if use_mixup:
                        inputs, targets_a, targets_b, lam = mixup_data(
                            features, labels, alpha=0.2, device=device)
                        char_logits, struct_pred = model(inputs)

                        # Handle Structural Target (also mix?)
                        # Standard MixUp only mixes labels (CrossEntropy).
                        # For Structural (Cosine), we can mix the target embeddings if we have them.
                        struct_target_a = canonical_emb[targets_a] if canonical_emb is not None else torch.randn_like(
                            struct_pred)
                        struct_target_b = canonical_emb[targets_b] if canonical_emb is not None else torch.randn_like(
                            struct_pred)

                        # Compute Loss for both
                        loss_a, _ = criterion(
                            char_logits, targets_a, struct_pred, struct_target_a)
                        loss_b, _ = criterion(
                            char_logits, targets_b, struct_pred, struct_target_b)
                        loss = lam * loss_a + (1 - lam) * loss_b
                    else:
                        char_logits, struct_pred = model(features)
                        if canonical_emb is not None:
                            struct_target = canonical_emb[labels]
                        else:
                            struct_target = torch.randn_like(struct_pred)
                        loss, loss_dict = criterion(
                            char_logits, labels, struct_pred, struct_target)

                scaler.scale(loss).backward()
                # Gradient Clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
            else:
                # CPU/MPS Path (Simplified without MixUp for now or same logic)
                # Let's apply MixUp here too for consistency if needed, but primarily User uses 4070Ti (AMP)
                char_logits, struct_pred = model(features)

                if canonical_emb is not None:
                    struct_target = canonical_emb[labels]
                else:
                    struct_target = torch.randn_like(struct_pred)

                loss, loss_dict = criterion(
                    char_logits, labels, struct_pred, struct_target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Train Loss: {avg_loss:.4f}")
        history['train_loss'].append(avg_loss)

        # 10. Evaluation
        if test_loader:
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for features, labels in test_loader:
                    features = features.to(device)
                    labels = labels.to(device)

                    char_logits, struct_pred = model(features)

                    if canonical_emb is not None:
                        struct_target = canonical_emb[labels]
                    else:
                        struct_target = torch.randn_like(struct_pred)

                    loss, _ = criterion(char_logits, labels,
                                        struct_pred, struct_target)
                    test_loss += loss.item()

                    _, predicted = torch.max(char_logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            avg_test_loss = test_loss/len(test_loader)
            current_acc = 100 * correct / total
            print(
                f"Epoch {epoch+1} Test Loss: {avg_test_loss:.4f} | Acc: {current_acc:.2f}%")

            print(
                f"Epoch {epoch+1} Test Loss: {avg_test_loss:.4f} | Acc: {current_acc:.2f}%")

            # scheduler.step() for ReduceLROnPlateau needs a metric
            scheduler.step(avg_test_loss)
            history['val_loss'].append(avg_test_loss)
            history['val_acc'].append(current_acc)

            if current_acc > best_acc:
                best_acc = current_acc
                torch.save(model.state_dict(), "models/best_model.pth")
                print(
                    f"  [+] Saved new best model with accuracy: {current_acc:.2f}%")

            model.train()

    torch.save(model.state_dict(), "models/final_model.pth")
    print("Training finished. Saved 'models/final_model.pth'.")

    # 11. Plotting
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        if history['val_loss']:
            plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        if history['val_acc']:
            plt.subplot(1, 2, 2)
            plt.plot(history['val_acc'], label='Val Accuracy')
            plt.title('Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()

        plt.savefig('training_history.png')
        print("Saved training history plot to 'training_history.png'")
    except ImportError:
        print("matplotlib not found, skipping plotting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Handwriting Similarity Model")
    parser.add_argument('--data_dir', type=str,
                        default='data/casia', help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int,
                        default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='Learning Rate')
    parser.add_argument('--lambda_struct', type=float,
                        default=0.2, help='Structural Loss Weight')
    parser.add_argument('--hidden_dim', type=int,
                        default=256, help='Hidden Dimension Size')
    parser.add_argument('--seed', type=int, default=42, help='Random Seed')
    parser.add_argument('--resume', nargs='?',
                        const='models/best_model.pth', help='Resume from checkpoint')
    args = parser.parse_args()

    train(args)
