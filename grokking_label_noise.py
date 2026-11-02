import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from scipy.optimize import curve_fit
import time
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ModularAdditionDataset:
    def __init__(self, p=97, train_fraction=0.4, noise_level=0.0):
        """
        Create dataset with COMPLETELY RANDOM split (no seed).
        Each instantiation creates a different random train/test partition.
        """
        self.p = p
        self.noise_level = noise_level

        # Generate all possible pairs
        all_pairs = [(i, j) for i in range(p) for j in range(p)]
        
        # COMPLETELY RANDOM shuffle (no seed)
        np.random.shuffle(all_pairs)

        # Split into train and test
        n_train = int(len(all_pairs) * train_fraction)
        self.train_pairs = all_pairs[:n_train]
        self.test_pairs = all_pairs[n_train:]

        # Create tensors
        self.X_train = torch.tensor(self.train_pairs, dtype=torch.long)
        self.y_train = torch.tensor([(x + y) % p for x, y in self.train_pairs], dtype=torch.long)

        self.X_test = torch.tensor(self.test_pairs, dtype=torch.long)
        self.y_test = torch.tensor([(x + y) % p for x, y in self.test_pairs], dtype=torch.long)

        # Store original clean labels
        self.y_train_clean = self.y_train.clone()

        # Generate noisy labels ONCE and store them for consistent evaluation
        self.y_train_noisy = self.y_train_clean.clone()
        if self.noise_level > 0:
            n_noisy = int(len(self.y_train_noisy) * self.noise_level)
            if n_noisy > 0:
                noisy_indices = torch.randperm(len(self.y_train_noisy))[:n_noisy]
                self.y_train_noisy[noisy_indices] = torch.randint(0, self.p, (n_noisy,))
    
    def get_split_info(self):
        """Return information about the current split for reporting"""
        # Get first 5 train and test examples as a fingerprint
        train_sample = self.train_pairs[:5] if len(self.train_pairs) >= 5 else self.train_pairs
        test_sample = self.test_pairs[:5] if len(self.test_pairs) >= 5 else self.test_pairs
        
        return {
            'n_train': len(self.train_pairs),
            'n_test': len(self.test_pairs),
            'train_sample': str(train_sample),
            'test_sample': str(test_sample),
            'train_pairs': self.train_pairs,  # Full list for saving
            'test_pairs': self.test_pairs      # Full list for saving
        }

    def get_noisy_train_batch(self, batch_size):
        """Get a batch of training data with consistent noise"""
        indices = torch.randperm(len(self.X_train))[:batch_size]
        X_batch = self.X_train[indices]
        y_batch = self.y_train_noisy[indices]
        return X_batch.to(device), y_batch.to(device)

    def get_full_train_set(self):
        """Get full training set with consistent noise"""
        return self.X_train.to(device), self.y_train_noisy.to(device)

    def get_test_set(self):
        """Get clean test set (no noise)"""
        return self.X_test.to(device), self.y_test.to(device)

class GrokkingMLP(nn.Module):
    def __init__(self, p=97, hidden_size=512, embedding_dim=128):
        """
        MLP with RANDOM initialization (no seed control).
        Each instantiation has different random weights.
        """
        super().__init__()
        self.p = p

        # Embedding for inputs
        self.embedding = nn.Embedding(p, embedding_dim)

        # MLP layers
        self.fc1 = nn.Linear(2 * embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, p)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.0)

        # RANDOM initialization (PyTorch default random state)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x1_emb = self.embedding(x[:, 0])
        x2_emb = self.embedding(x[:, 1])
        x = torch.cat([x1_emb, x2_emb], dim=1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)

        return x

def calculate_grokking_index(train_accs, test_accs, epoch_interval=1):
    """Calculate grokking index as area between accuracy curves."""
    epochs = np.arange(len(train_accs)) * epoch_interval

    start_idx = -1
    for i, acc in enumerate(train_accs):
        if acc >= 0.95:
            start_idx = i
            break

    if start_idx == -1:
        return 0.0

    end_idx = len(test_accs)
    for i in range(start_idx, len(test_accs)):
        if test_accs[i] >= train_accs[i] - 0.05:
            end_idx = i + 1
            break
        if epochs[i] >= 50000:
            end_idx = i
            break

    if end_idx <= start_idx:
        return 0.0

    train_accs_slice = np.array(train_accs[start_idx:end_idx])
    test_accs_slice = np.array(test_accs[start_idx:end_idx])
    epochs_slice = epochs[start_idx:end_idx]

    if len(epochs_slice) < 2:
        return 0.0

    diff = train_accs_slice - test_accs_slice
    grokking_index = np.trapezoid(diff, epochs_slice)

    return grokking_index

def calculate_loss_index(train_losses, test_losses, train_accs, test_accs, epoch_interval=1):
    """Calculate loss index as area between loss curves."""
    epochs = np.arange(len(train_losses)) * epoch_interval

    start_idx = -1
    for i, acc in enumerate(train_accs):
        if acc >= 0.95:
            start_idx = i
            break

    if start_idx == -1:
        return 0.0

    end_idx = len(test_accs)
    for i in range(start_idx, len(test_accs)):
        if test_accs[i] >= train_accs[i] - 0.05:
            end_idx = i + 1
            break
        if epochs[i] >= 50000:
            end_idx = i
            break

    if end_idx <= start_idx:
        return 0.0

    train_losses_slice = np.array(train_losses[start_idx:end_idx])
    test_losses_slice = np.array(test_losses[start_idx:end_idx])
    epochs_slice = epochs[start_idx:end_idx]

    if len(epochs_slice) < 2:
        return 0.0

    diff = test_losses_slice - train_losses_slice
    loss_index = np.trapezoid(diff, epochs_slice)

    return loss_index

def train_model(noise_level, max_epochs=50000, batch_size=256, 
                weight_decay=1.0, learning_rate=5e-3, 
                config_name='default', verbose=True,
                use_new_split=False):
    """
    Train a single model with given noise level and hyperparameters.
    
    Parameters:
    -----------
    use_new_split : bool
        If True, creates a NEW random train-test split for this run.
        If False, uses whatever split the dataset generates (still random, but reused within this call).
    
    Returns:
    --------
    train_losses, test_losses, train_accs, test_accs, split_info
    """

    # Create dataset with COMPLETELY RANDOM split
    # Each call creates a different random partition
    dataset = ModularAdditionDataset(
        p=97, 
        train_fraction=0.4, 
        noise_level=noise_level
    )
    
    # Get split information for reporting
    split_info = dataset.get_split_info()

    # Create model with COMPLETELY RANDOM initialization
    # Each call creates different random weights
    model = GrokkingMLP(p=97, hidden_size=512, embedding_dim=128).to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Learning rate schedule
    def lr_lambda(epoch):
        warmup_epochs = 100
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 0.99995 ** (epoch - warmup_epochs)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    # Tracking metrics
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    # Get test set (clean)
    X_test, y_test = dataset.get_test_set()

    # Get full training set
    X_train_full, y_train_noisy = dataset.get_full_train_set()

    # Training loop
    desc = f"{config_name} | Noise {noise_level:.1%}"
    pbar = tqdm(range(max_epochs), desc=desc, disable=not verbose)

    for epoch in pbar:
        # Training step
        model.train()
        X_batch, y_batch = dataset.get_noisy_train_batch(min(batch_size, len(dataset.X_train)))

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            # Training set evaluation
            train_outputs = model(X_train_full)
            train_loss = criterion(train_outputs, y_train_noisy).item()
            train_acc = (train_outputs.argmax(dim=1) == y_train_noisy).float().mean().item()

            # Test set evaluation
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test).item()
            test_acc = (test_outputs.argmax(dim=1) == y_test).float().mean().item()

            # Record metrics
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            # Update progress bar
            if verbose and epoch % 100 == 0:
                pbar.set_postfix({
                    'ep': epoch,
                    'train': f'{train_acc:.3f}',
                    'test': f'{test_acc:.3f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })

            # Early stopping
            if train_acc >= 0.80 and test_acc >= train_acc - 0.05:
                if verbose:
                    print(f"\n  → Early stopping at epoch {epoch}")
                break

    return train_losses, test_losses, train_accs, test_accs, split_info

def exponential_func(x, a, b):
    """Exponential function for curve fitting"""
    return a * np.exp(b * x)

def save_split_details(split_info, split_idx, base_path='splits'):
    """Save complete train and test splits to files for documentation"""
    
    # Create splits directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    train_pairs = split_info['train_pairs']
    test_pairs = split_info['test_pairs']
    p = 97  # modulo value
    
    # Save training set
    train_file = f"{base_path}/split_{split_idx}_train.txt"
    with open(train_file, 'w') as f:
        f.write(f"Split {split_idx} - Training Set\n")
        f.write(f"Total examples: {len(train_pairs)}\n")
        f.write("="*60 + "\n\n")
        for i, (x, y) in enumerate(train_pairs):
            result = (x + y) % p
            f.write(f"{i+1:5d}. ({x:2d}, {y:2d}) -> {result:2d}\n")
    
    # Save test set
    test_file = f"{base_path}/split_{split_idx}_test.txt"
    with open(test_file, 'w') as f:
        f.write(f"Split {split_idx} - Test Set\n")
        f.write(f"Total examples: {len(test_pairs)}\n")
        f.write("="*60 + "\n\n")
        for i, (x, y) in enumerate(test_pairs):
            result = (x + y) % p
            f.write(f"{i+1:5d}. ({x:2d}, {y:2d}) -> {result:2d}\n")
    
    print(f"  ✓ Split {split_idx} saved to {train_file} and {test_file}")
    
    return train_file, test_file

def run_split_validation_experiment():
    """Run Experiment 2: Split Validation"""
    print("\n" + "="*70)
    print("EXPERIMENT 2: SPLIT VALIDATION")
    print("Testing robustness across COMPLETELY RANDOM train-test splits")
    print("="*70)
    
    noise_levels_split = [0.0, 0.05, 0.10, 0.15]
    n_splits = 3
    n_runs_per_split = 5
    
    split_validation_results = []
    
    print(f"Noise levels: {[f'{n:.0%}' for n in noise_levels_split]}")
    print(f"Number of independent splits: {n_splits}")
    print(f"Runs per split: {n_runs_per_split}")
    print(f"Total runs: {len(noise_levels_split) * n_splits * n_runs_per_split}")
    print(f"\nNote: Each 'split' uses 5 runs with the SAME random partition,")
    print(f"      but different random initializations.")
    print(f"      Different 'splits' use DIFFERENT random partitions.")
    print(f"      Complete splits will be saved to the 'splits/' directory.")
    
    for split_idx in range(n_splits):
        print(f"\n{'='*60}")
        print(f"SPLIT GROUP {split_idx + 1}/{n_splits}")
        print(f"Creating NEW random train-test partition for this group")
        print(f"{'='*60}")
        
        # Create ONE dataset for this split group
        base_dataset = ModularAdditionDataset(p=97, train_fraction=0.4, noise_level=0.0)
        base_split_info = base_dataset.get_split_info()
        
        # SAVE THE COMPLETE SPLIT TO FILES
        train_file, test_file = save_split_details(base_split_info, split_idx + 1)
        
        # Print first 10 examples as preview
        print(f"\nPreview of Split {split_idx + 1}:")
        print(f"  First 10 train examples: {base_split_info['train_pairs'][:10]}")
        print(f"  First 10 test examples: {base_split_info['test_pairs'][:10]}")
        print(f"  Full split saved to: {train_file} and {test_file}\n")
        
        for noise_level in noise_levels_split:
            print(f"Noise Level: {noise_level:.1%}")
            
            # Track metrics for this split-noise combination
            split_noise_gis = []
            split_noise_lis = []
            split_noise_epochs = []
            
            for run_idx in range(n_runs_per_split):
                print(f"  Run {run_idx + 1}/{n_runs_per_split} (split {split_idx+1}, new init)", end=" ")
                
                train_losses, test_losses, train_accs, test_accs, split_info = train_model(
                    noise_level=noise_level,
                    max_epochs=50000,
                    batch_size=256,
                    weight_decay=1.0,
                    learning_rate=5e-3,
                    config_name=f'Split{split_idx+1}',
                    verbose=False,
                    use_new_split=False
                )
                
                grokking_index = calculate_grokking_index(train_accs, test_accs)
                loss_index = calculate_loss_index(train_losses, test_losses, train_accs, test_accs)
                max_acc_gap = max(np.array(train_accs) - np.array(test_accs)) if train_accs else 0
                
                split_noise_gis.append(grokking_index)
                split_noise_lis.append(loss_index)
                split_noise_epochs.append(len(train_accs))
                
                result = {
                    'experiment': 'split_validation',
                    'split_group': split_idx,
                    'split_train_file': train_file,
                    'split_test_file': test_file,
                    'noise_level': noise_level,
                    'run': run_idx,
                    'grokking_index': grokking_index,
                    'loss_index': loss_index,
                    'max_acc_gap': max_acc_gap,
                    'total_epochs': len(train_accs),
                    'final_train_acc': train_accs[-1] if train_accs else 0,
                    'final_test_acc': test_accs[-1] if test_accs else 0,
                    'final_train_loss': train_losses[-1] if train_losses else float('inf'),
                    'final_test_loss': test_losses[-1] if test_losses else float('inf'),
                    'n_train': split_info['n_train'],
                    'n_test': split_info['n_test'],
                    'train_sample': split_info['train_sample'],
                    'test_sample': split_info['test_sample']
                }
                
                split_validation_results.append(result)
                print(f"→ GI: {grokking_index:.2f}, LI: {loss_index:.2f}, Epochs: {len(train_accs)}")
            
            # Print averages
            avg_gi = np.mean(split_noise_gis)
            avg_li = np.mean(split_noise_lis)
            avg_epochs = np.mean(split_noise_epochs)
            std_gi = np.std(split_noise_gis)
            std_li = np.std(split_noise_lis)
            std_epochs = np.std(split_noise_epochs)
            
            print(f"  → Split {split_idx+1}, Noise {noise_level:.1%} AVERAGES:")
            print(f"     GI: {avg_gi:.2f} ± {std_gi:.2f}")
            print(f"     LI: {avg_li:.2f} ± {std_li:.2f}")
            print(f"     Epochs: {avg_epochs:.1f} ± {std_epochs:.1f}")
    
    # Save results
    split_df = pd.DataFrame(split_validation_results)
    split_df.to_csv('split_validation_results.csv', index=False)
    
    # Compute and save averages
    split_noise_averages = []
    for split_idx in range(n_splits):
        for noise in noise_levels_split:
            subset = split_df[(split_df['split_group'] == split_idx) & (split_df['noise_level'] == noise)]
            avg_result = {
                'split_group': split_idx,
                'noise_level': noise,
                'avg_grokking_index': subset['grokking_index'].mean(),
                'std_grokking_index': subset['grokking_index'].std(),
                'avg_loss_index': subset['loss_index'].mean(),
                'std_loss_index': subset['loss_index'].std(),
                'avg_epochs': subset['total_epochs'].mean(),
                'std_epochs': subset['total_epochs'].std(),
                'avg_max_acc_gap': subset['max_acc_gap'].mean(),
                'std_max_acc_gap': subset['max_acc_gap'].std()
            }
            split_noise_averages.append(avg_result)
    
    split_avg_df = pd.DataFrame(split_noise_averages)
    split_avg_df.to_csv('split_validation_averages.csv', index=False)
    
    # Variance decomposition
    variance_decomp_results = []
    for noise in noise_levels_split:
        noise_data = split_df[split_df['noise_level'] == noise]
        overall_mean_gi = noise_data['grokking_index'].mean()
        overall_std_gi = noise_data['grokking_index'].std()
        split_means_gi = noise_data.groupby('split_group')['grokking_index'].mean()
        between_split_std_gi = split_means_gi.std()
        split_stds_gi = noise_data.groupby('split_group')['grokking_index'].std()
        within_split_std_gi = split_stds_gi.mean()
        ratio_gi = between_split_std_gi / within_split_std_gi if within_split_std_gi > 0 else 0
        
        variance_decomp_results.append({
            'noise_level': noise,
            'overall_mean_gi': overall_mean_gi,
            'overall_std_gi': overall_std_gi,
            'between_split_std_gi': between_split_std_gi,
            'within_split_std_gi': within_split_std_gi,
            'ratio_gi': ratio_gi
        })
    
    variance_df = pd.DataFrame(variance_decomp_results)
    variance_df.to_csv('split_variance_decomposition.csv', index=False)
    
    print(f"\n{'='*70}")
    print("✓ Split validation results saved to 'split_validation_results.csv'")
    print("✓ Split validation averages saved to 'split_validation_averages.csv'")
    print("✓ Variance decomposition saved to 'split_variance_decomposition.csv'")
    print(f"{'='*70}\n")

def run_hyperparameter_ablation_experiment():
    """Run Experiment 3: Hyperparameter Ablation"""
    print("\n" + "="*70)
    print("EXPERIMENT 3: HYPERPARAMETER ABLATION")
    print("Each run uses COMPLETELY RANDOM initialization")
    print("="*70)
    
    noise_levels_ablation = [0.0, 0.05, 0.10, 0.15]
    n_runs_ablation = 5
    
    configs = [
        {'weight_decay': 1.0, 'lr': 5e-3, 'name': 'Original'},
        {'weight_decay': 0.5, 'lr': 5e-3, 'name': 'WD_0.5'},
        {'weight_decay': 2.0, 'lr': 5e-3, 'name': 'WD_2.0'},
        {'weight_decay': 1.0, 'lr': 1e-3, 'name': 'LR_1e-3'},
        {'weight_decay': 1.0, 'lr': 1e-2, 'name': 'LR_1e-2'},
    ]
    
    ablation_results = []
    
    print(f"Configurations: {len(configs)}")
    print(f"Noise levels: {[f'{n:.0%}' for n in noise_levels_ablation]}")
    print(f"Runs per config: {n_runs_ablation}")
    print(f"Total runs: {len(configs) * len(noise_levels_ablation) * n_runs_ablation}")
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"CONFIG: {config['name']}")
        print(f"  Weight Decay: {config['weight_decay']}")
        print(f"  Learning Rate: {config['lr']}")
        print(f"{'='*60}")
        
        for noise_level in noise_levels_ablation:
            print(f"\nNoise Level: {noise_level:.1%}")
            
            config_noise_gis = []
            config_noise_lis = []
            config_noise_epochs = []
            
            for run_idx in range(n_runs_ablation):
                print(f"  Run {run_idx + 1}/{n_runs_ablation} (new random init)", end=" ")
                
                train_losses, test_losses, train_accs, test_accs, split_info = train_model(
                    noise_level=noise_level,
                    max_epochs=50000,
                    batch_size=256,
                    weight_decay=config['weight_decay'],
                    learning_rate=config['lr'],
                    config_name=config['name'],
                    verbose=False,
                    use_new_split=True
                )
                
                grokking_index = calculate_grokking_index(train_accs, test_accs)
                loss_index = calculate_loss_index(train_losses, test_losses, train_accs, test_accs)
                max_acc_gap = max(np.array(train_accs) - np.array(test_accs)) if train_accs else 0
                
                config_noise_gis.append(grokking_index)
                config_noise_lis.append(loss_index)
                config_noise_epochs.append(len(train_accs))
                
                result = {
                    'experiment': 'hyperparameter_ablation',
                    'config_name': config['name'],
                    'weight_decay': config['weight_decay'],
                    'learning_rate': config['lr'],
                    'noise_level': noise_level,
                    'run': run_idx,
                    'grokking_index': grokking_index,
                    'loss_index': loss_index,
                    'max_acc_gap': max_acc_gap,
                    'total_epochs': len(train_accs),
                    'final_train_acc': train_accs[-1] if train_accs else 0,
                    'final_test_acc': test_accs[-1] if test_accs else 0,
                    'final_train_loss': train_losses[-1] if train_losses else float('inf'),
                    'final_test_loss': test_losses[-1] if test_losses else float('inf'),
                    'n_train': split_info['n_train'],
                    'n_test': split_info['n_test'],
                    'train_sample': split_info['train_sample'],
                    'test_sample': split_info['test_sample']
                }
                
                ablation_results.append(result)
                print(f"→ GI: {grokking_index:.2f}, LI: {loss_index:.2f}, Epochs: {len(train_accs)}")
            
            # Print averages
            avg_gi = np.mean(config_noise_gis)
            avg_li = np.mean(config_noise_lis)
            avg_epochs = np.mean(config_noise_epochs)
            std_gi = np.std(config_noise_gis)
            std_li = np.std(config_noise_lis)
            std_epochs = np.std(config_noise_epochs)
            
            print(f"  → Config {config['name']}, Noise {noise_level:.1%} AVERAGES:")
            print(f"     GI: {avg_gi:.2f} ± {std_gi:.2f}")
            print(f"     LI: {avg_li:.2f} ± {std_li:.2f}")
            print(f"     Epochs: {avg_epochs:.1f} ± {std_epochs:.1f}")
    
    # Save results
    ablation_df = pd.DataFrame(ablation_results)
    ablation_df.to_csv('hyperparameter_ablation_results.csv', index=False)
    
    # Compute and save averages
    config_noise_averages = []
    for config in configs:
        for noise in noise_levels_ablation:
            subset = ablation_df[(ablation_df['config_name'] == config['name']) & (ablation_df['noise_level'] == noise)]
            avg_result = {
                'config_name': config['name'],
                'weight_decay': config['weight_decay'],
                'learning_rate': config['lr'],
                'noise_level': noise,
                'avg_grokking_index': subset['grokking_index'].mean(),
                'std_grokking_index': subset['grokking_index'].std(),
                'avg_loss_index': subset['loss_index'].mean(),
                'std_loss_index': subset['loss_index'].std(),
                'avg_epochs': subset['total_epochs'].mean(),
                'std_epochs': subset['total_epochs'].std(),
                'avg_max_acc_gap': subset['max_acc_gap'].mean(),
                'std_max_acc_gap': subset['max_acc_gap'].std()
            }
            config_noise_averages.append(avg_result)
    
    config_avg_df = pd.DataFrame(config_noise_averages)
    config_avg_df.to_csv('hyperparameter_ablation_averages.csv', index=False)
    
    # Exponential fitting
    exp_fit_results = []
    for config_name in ablation_df['config_name'].unique():
        config_data = ablation_df[ablation_df['config_name'] == config_name]
        means = config_data.groupby('noise_level')['grokking_index'].mean()
        
        try:
            popt, pcov = curve_fit(exponential_func, means.index * 100, means.values, p0=[100, 0.2], maxfev=10000)
            perr = np.sqrt(np.diag(pcov))
            residuals = means.values - exponential_func(means.index * 100, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((means.values - np.mean(means.values))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            exp_fit_results.append({
                'config_name': config_name,
                'weight_decay': config_data['weight_decay'].iloc[0],
                'learning_rate': config_data['learning_rate'].iloc[0],
                'a': popt[0],
                'b': popt[1],
                'b_stderr': perr[1],
                'b_ci_lower': popt[1] - 1.96*perr[1],
                'b_ci_upper': popt[1] + 1.96*perr[1],
                'r_squared': r_squared
            })
        except Exception as e:
            print(f"\n{config_name}: Could not fit exponential - {e}")
    
    exp_fit_df = pd.DataFrame(exp_fit_results)
    exp_fit_df.to_csv('exponential_fits_by_config.csv', index=False)
    
    print(f"\n{'='*70}")
    print("✓ Hyperparameter ablation results saved to 'hyperparameter_ablation_results.csv'")
    print("✓ Hyperparameter ablation averages saved to 'hyperparameter_ablation_averages.csv'")
    print("✓ Exponential fits saved to 'exponential_fits_by_config.csv'")
    print(f"{'='*70}\n")

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    
    # ==================== EXPERIMENT SELECTION ====================
    print("\n" + "="*70)
    print("GROKKING WITH LABEL NOISE - ROBUSTNESS VALIDATION")
    print("FULLY RANDOM MODE: No seeds, completely stochastic")
    print("="*70)
    print("\nSelect experiment to run:")
    print("1. Main Experiment (0-15% noise, 0.5% increments, 5 runs)")
    print("2. Split Validation → Hyperparameter Ablation (Sequential)")
    print("3. Hyperparameter Ablation only")
    print("4. Run ALL experiments")
    
    experiment_choice = input("\nEnter choice (1-4): ").strip()
    
    # ==================== EXPERIMENT 1: MAIN EXPERIMENT ====================
    
    if experiment_choice in ['1', '4']:
        print("\n" + "="*70)
        print("EXPERIMENT 1: MAIN EXPERIMENT")
        print("Each run uses COMPLETELY RANDOM initialization and split")
        print("="*70)
        
        noise_levels = np.arange(0.0, 0.16, 0.005)
        n_runs = 5
        all_results = []
        
        print(f"Noise levels: 0% to 15% in 0.5% increments ({len(noise_levels)} levels)")
        print(f"Runs per noise level: {n_runs}")
        print(f"Total runs: {len(noise_levels) * n_runs}")
        
        for noise_level in noise_levels:
            print(f"\n--- Noise Level: {noise_level:.1%} ---")
            
            for run_idx in range(n_runs):
                print(f"Run {run_idx + 1}/{n_runs} (new random split)", end=" ")
                
                train_losses, test_losses, train_accs, test_accs, split_info = train_model(
                    noise_level=noise_level,
                    max_epochs=50000,
                    batch_size=256,
                    weight_decay=1.0,
                    learning_rate=5e-3,
                    config_name='Main',
                    verbose=False,
                    use_new_split=True
                )
                
                grokking_index = calculate_grokking_index(train_accs, test_accs)
                loss_index = calculate_loss_index(train_losses, test_losses, train_accs, test_accs)
                max_acc_gap = max(np.array(train_accs) - np.array(test_accs)) if train_accs else 0
                
                result = {
                    'experiment': 'main',
                    'noise_level': noise_level,
                    'run': run_idx + 1,
                    'grokking_index': grokking_index,
                    'loss_index': loss_index,
                    'max_acc_gap': max_acc_gap,
                    'total_epochs': len(train_accs),
                    'final_train_acc': train_accs[-1] if train_accs else 0,
                    'final_test_acc': test_accs[-1] if test_accs else 0,
                    'final_train_loss': train_losses[-1] if train_losses else float('inf'),
                    'final_test_loss': test_losses[-1] if test_losses else float('inf'),
                    'n_train': split_info['n_train'],
                    'n_test': split_info['n_test'],
                    'train_sample': split_info['train_sample'],
                    'test_sample': split_info['test_sample']
                }
                
                all_results.append(result)
                print(f"→ GI: {grokking_index:.2f}, LI: {loss_index:.2f}, Epochs: {len(train_accs)}")
        
        df_main = pd.DataFrame(all_results)
        df_main.to_csv('main_experiment_results.csv', index=False)
        print("\n✓ Main experiment results saved to 'main_experiment_results.csv'")
    
    # ==================== EXPERIMENT 2 → 3: SEQUENTIAL EXECUTION ====================
    
    if experiment_choice in ['2', '4']:
        # Run Split Validation
        run_split_validation_experiment()
        
        # Automatically run Hyperparameter Ablation
        print("\n" + "="*70)
        print("🚀 AUTOMATICALLY STARTING EXPERIMENT 3")
        print("="*70)
        time.sleep(2)  # Brief pause for visibility
        
        run_hyperparameter_ablation_experiment()
    
    # ==================== EXPERIMENT 3 ONLY ====================
    
    if experiment_choice == '3':
        run_hyperparameter_ablation_experiment()
    
    # ==================== COMPLETION MESSAGE ====================
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*70)
    print("\nGenerated files:")
    if experiment_choice in ['1', '4']:
        print("  - main_experiment_results.csv")
    if experiment_choice in ['2', '4']:
        print("  - split_validation_results.csv")
        print("  - split_validation_averages.csv")
        print("  - split_variance_decomposition.csv")
        print("  - splits/split_1_train.txt")
        print("  - splits/split_1_test.txt")
        print("  - splits/split_2_train.txt")
        print("  - splits/split_2_test.txt")
        print("  - splits/split_3_train.txt")
        print("  - splits/split_3_test.txt")
        print("  - hyperparameter_ablation_results.csv")
        print("  - hyperparameter_ablation_averages.csv")
        print("  - exponential_fits_by_config.csv")
    if experiment_choice == '3':
        print("  - hyperparameter_ablation_results.csv")
        print("  - hyperparameter_ablation_averages.csv")
        print("  - exponential_fits_by_config.csv")
