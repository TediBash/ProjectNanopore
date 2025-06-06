import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from collections import Counter
import seaborn as sns

def analyze_dataset(npz_dir):
    """Analyze the dataset statistics"""
    stats = {
        'num_files': 0,
        'total_samples': 0,
        'nucleotide_counts': Counter(),
        'sequence_lengths': [],
        'signal_stats': {
            'min': float('inf'),
            'max': float('-inf'),
            'means': [],
            'stds': []
        }
    }
    
    npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
    stats['num_files'] = len(npz_files)
    
    for npz_file in npz_files:
        data = np.load(os.path.join(npz_dir, npz_file))
        num_samples = len(data['x'])
        stats['total_samples'] += num_samples
        
        for i in range(num_samples):
            x = data['x'][i]
            y = data['y'][i]
            
            # Signal statistics
            stats['signal_stats']['min'] = min(stats['signal_stats']['min'], x.min())
            stats['signal_stats']['max'] = max(stats['signal_stats']['max'], x.max())
            stats['signal_stats']['means'].append(x.mean())
            stats['signal_stats']['stds'].append(x.std())
            
            # Sequence statistics
            sequence = [char for char in y if char != '']
            stats['sequence_lengths'].append(len(sequence))
            stats['nucleotide_counts'].update(sequence)
    
    # Calculate final statistics
    stats['signal_stats']['global_mean'] = np.mean(stats['signal_stats']['means'])
    stats['signal_stats']['global_std'] = np.mean(stats['signal_stats']['stds'])
    stats['avg_sequence_length'] = np.mean(stats['sequence_lengths'])
    stats['nucleotide_distribution'] = {
        nuc: count / sum(stats['nucleotide_counts'].values()) 
        for nuc, count in stats['nucleotide_counts'].items()
    }
    
    return stats

def plot_dataset_statistics(stats, save_dir=None):
    """Visualize dataset statistics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Nucleotide distribution
    nucleotides = list(stats['nucleotide_distribution'].keys())
    counts = list(stats['nucleotide_distribution'].values())
    axes[0, 0].bar(nucleotides, counts)
    axes[0, 0].set_title('Nucleotide Distribution')
    axes[0, 0].set_xlabel('Nucleotide')
    axes[0, 0].set_ylabel('Proportion')
    
    # Sequence length distribution
    axes[0, 1].hist(stats['sequence_lengths'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Sequence Length Distribution')
    axes[0, 1].set_xlabel('Sequence Length')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].axvline(stats['avg_sequence_length'], color='red', linestyle='--', 
                       label=f'Mean: {stats["avg_sequence_length"]:.1f}')
    axes[0, 1].legend()
    
    # Signal mean distribution
    axes[1, 0].hist(stats['signal_stats']['means'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Signal Mean Distribution')
    axes[1, 0].set_xlabel('Mean Signal Value')
    axes[1, 0].set_ylabel('Count')
    
    # Signal std distribution
    axes[1, 1].hist(stats['signal_stats']['stds'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Signal Standard Deviation Distribution')
    axes[1, 1].set_xlabel('Signal Std Dev')
    axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'dataset_statistics.png'), dpi=300, bbox_inches='tight')
    plt.show()

def preprocess_signal(x, normalize=True, smooth=False, window_size=5):
    """Preprocess electrical signal"""
    x = x.copy()
    
    # Remove outliers using IQR method
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    x = np.clip(x, lower_bound, upper_bound)
    
    # Smooth signal if requested
    if smooth:
        from scipy.signal import savgol_filter
        x = savgol_filter(x, window_size, 3)
    
    # Normalize
    if normalize:
        x = (x - x.mean()) / (x.std() + 1e-8)
    
    return x

def augment_signal(x, noise_factor=0.1, shift_factor=0.1):
    """Augment signal for training"""
    x_aug = x.copy()
    
    # Add Gaussian noise
    if noise_factor > 0:
        noise = np.random.normal(0, noise_factor * x.std(), x.shape)
        x_aug += noise
    
    # Random shift
    if shift_factor > 0:
        shift = np.random.uniform(-shift_factor, shift_factor) * x.std()
        x_aug += shift
    
    return x_aug

def create_train_val_split(npz_dir, output_dir, train_ratio=0.8, random_seed=42):
    """Create train/validation split and save to separate directories"""
    np.random.seed(random_seed)
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get all npz files
    npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
    np.random.shuffle(npz_files)
    
    # Split files
    n_train = int(len(npz_files) * train_ratio)
    train_files = npz_files[:n_train]
    val_files = npz_files[n_train:]
    
    # Copy files to respective directories
    import shutil
    for f in train_files:
        shutil.copy(os.path.join(npz_dir, f), os.path.join(train_dir, f))
    
    for f in val_files:
        shutil.copy(os.path.join(npz_dir, f), os.path.join(val_dir, f))
    
    print(f"Split complete: {len(train_files)} train files, {len(val_files)} val files")
    
    return train_dir, val_dir

def visualize_signal_segments(npz_file, num_samples=5, segment_size=500):
    """Visualize segments of signals with their nucleotide labels"""
    data = np.load(npz_file)
    num_samples = min(num_samples, len(data['x']))
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 3*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    colors = {'A': 'red', 'C': 'blue', 'G': 'green', 'T': 'orange', '': 'gray'}
    
    for i in range(num_samples):
        x = data['x'][i][:segment_size]
        y = data['y'][i][:segment_size]
        
        # Plot signal
        axes[i].plot(x, 'k-', linewidth=0.5, alpha=0.7)
        
        # Color background by nucleotide
        for j, nuc in enumerate(y):
            if nuc != '':
                axes[i].axvspan(j, j+1, alpha=0.3, color=colors[nuc])
        
        axes[i].set_xlim(0, segment_size)
        axes[i].set_title(f'Sample {i+1}')
        axes[i].set_ylabel('Signal')
        if i == num_samples - 1:
            axes[i].set_xlabel('Position')
        
        # Add legend to first plot
        if i == 0:
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=colors[nuc], alpha=0.3, label=nuc) 
                             for nuc in ['A', 'C', 'G', 'T']]
            axes[i].legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()

def calculate_signal_nucleotide_correlation(npz_dir, num_samples=1000):
    """Calculate correlation between signal features and nucleotides"""
    encoding_dict = {'A': 1, 'C': 2, 'G': 3, 'T': 4, '': 0}
    
    signal_features = []
    nucleotide_labels = []
    
    npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
    samples_processed = 0
    
    for npz_file in npz_files:
        if samples_processed >= num_samples:
            break
            
        data = np.load(os.path.join(npz_dir, npz_file))
        
        for i in range(len(data['x'])):
            if samples_processed >= num_samples:
                break
                
            x = data['x'][i]
            y = data['y'][i]
            
            # Extract features for each position
            for j in range(len(x)):
                if y[j] != '':  # Skip padding
                    # Extract local features
                    window_size = 10
                    start = max(0, j - window_size//2)
                    end = min(len(x), j + window_size//2)
                    
                    local_signal = x[start:end]
                    features = [
                        local_signal.mean(),
                        local_signal.std(),
                        local_signal.max(),
                        local_signal.min(),
                        x[j]  # Current value
                    ]
                    
                    signal_features.append(features)
                    nucleotide_labels.append(encoding_dict[y[j]])
            
            samples_processed += 1
    
    # Convert to numpy arrays
    signal_features = np.array(signal_features)
    nucleotide_labels = np.array(nucleotide_labels)
    
    # Calculate correlation matrix
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    signal_features_scaled = scaler.fit_transform(signal_features)
    
    # Create correlation heatmap
    feature_names = ['Mean', 'Std', 'Max', 'Min', 'Current']
    nucleotide_names = ['A', 'C', 'G', 'T']
    
    correlation_matrix = np.zeros((len(feature_names), len(nucleotide_names)))
    
    for i, nuc_id in enumerate([1, 2, 3, 4]):
        mask = nucleotide_labels == nuc_id
        if mask.sum() > 0:
            for j in range(len(feature_names)):
                correlation_matrix[j, i] = np.corrcoef(
                    signal_features_scaled[mask, j], 
                    nucleotide_labels[mask]
                )[0, 1]
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, 
                xticklabels=nucleotide_names,
                yticklabels=feature_names,
                annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, vmin=-1, vmax=1)
    plt.title('Signal Features vs Nucleotide Correlation')
    plt.tight_layout()
    plt.show()
    
    return correlation_matrix

def export_predictions_to_fasta(predictions, output_file):
    """Export predictions to FASTA format"""
    decoding_dict = {0: '', 1: 'A', 2: 'C', 3: 'G', 4: 'T'}
    
    with open(output_file, 'w') as f:
        for i, pred in enumerate(predictions):
            # Decode sequence
            sequence = ''.join([decoding_dict[p] for p in pred if p != 0])
            
            # Write FASTA format
            f.write(f'>Sequence_{i+1}\n')
            # Write sequence in chunks of 80 characters
            for j in range(0, len(sequence), 80):
                f.write(sequence[j:j+80] + '\n')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Data analysis utilities')
    parser.add_argument('--npz_dir', type=str, required=True, help='Directory containing npz files')
    parser.add_argument('--action', type=str, choices=['analyze', 'visualize', 'split', 'correlate'], 
                       default='analyze', help='Action to perform')
    parser.add_argument('--output_dir', type=str, default='./data_analysis', help='Output directory')
    parser.add_argument('--npz_file', type=str, help='Specific npz file for visualization')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.action == 'analyze':
        print("Analyzing dataset...")
        npz_dir =  os.path.join("datasets", "nuovi_npy")
        stats = analyze_dataset(npz_dir)
        
        print("\nDataset Statistics:")
        print(f"Number of files: {stats['num_files']}")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Average sequence length: {stats['avg_sequence_length']:.1f}")
        print(f"Signal range: [{stats['signal_stats']['min']:.2f}, {stats['signal_stats']['max']:.2f}]")
        print(f"Signal mean: {stats['signal_stats']['global_mean']:.2f} ± {stats['signal_stats']['global_std']:.2f}")
        print("\nNucleotide distribution:")
        for nuc, prop in stats['nucleotide_distribution'].items():
            print(f"  {nuc}: {prop:.3f}")
        
        # Save statistics
        import json
        with open(os.path.join(args.output_dir, 'dataset_stats.json'), 'w') as f:
            json.dump({k: v for k, v in stats.items() if k not in ['signal_stats']}, f, indent=2)
        
        # Plot statistics
        plot_dataset_statistics(stats, args.output_dir)
    
    elif args.action == 'visualize':
        if args.npz_file:
            visualize_signal_segments(args.npz_file)
        else:
            print("Please provide --npz_file for visualization")
    
    elif args.action == 'split':
        create_train_val_split(args.npz_dir, args.output_dir)
    
    elif args.action == 'correlate':
        print("Calculating signal-nucleotide correlations...")
        correlation_matrix = calculate_signal_nucleotide_correlation(args.npz_dir)
        np.save(os.path.join(args.output_dir, 'correlation_matrix.npy'), correlation_matrix)