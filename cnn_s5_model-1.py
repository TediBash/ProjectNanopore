import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import argparse
from tqdm import tqdm
import json
from pathlib import Path
from s5 import S5Block


# S5 Layer implementation
class S5Layer(nn.Module):
    def __init__(self, d_model, d_state=64, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Initialize parameters
        self.A = nn.Parameter(torch.randn(d_state, d_state))
        self.B = nn.Parameter(torch.randn(d_state, d_model))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model, d_model))
        
        # Learnable timestep
        self.log_dt = nn.Parameter(torch.rand(1) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min))
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        dt = torch.exp(self.log_dt)
        
        # Discretize continuous parameters
        A_discrete = torch.matrix_exp(self.A * dt)
        B_discrete = torch.linalg.solve(self.A, (A_discrete - torch.eye(self.d_state, device=x.device)) @ self.B)
        
        # Initialize state
        h = torch.zeros(batch_size, self.d_state, device=x.device)
        outputs = []
        
        # Process sequence
        for t in range(seq_len):
            h = A_discrete @ h.T + B_discrete @ x[:, t].T
            h = h.T
            y = h @ self.C.T + x[:, t] @ self.D.T
            outputs.append(y)
        
        output = torch.stack(outputs, dim=1)
        return self.norm(output + x)  # Residual connection

# CNN Feature Extractor
class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, out_channels=128, kernel_sizes=[7, 5, 3]):
        super().__init__()
        self.convs = nn.ModuleList()
        channels = [in_channels] + [out_channels] * len(kernel_sizes)
        
        for i in range(len(kernel_sizes)):
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(channels[i], channels[i+1], kernel_sizes[i], padding=kernel_sizes[i]//2),
                    nn.BatchNorm1d(channels[i+1]),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
            )
    
    def forward(self, x):
        # x shape: (batch, seq_len)
        x = x.unsqueeze(1)  # Add channel dimension: (batch, 1, seq_len)
        
        for conv in self.convs:
            x = conv(x)
        
        return x.transpose(1, 2)  # (batch, seq_len, channels)

# CRF Layer
class CRF(nn.Module):
    def __init__(self, num_tags, batch_first=True):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        
        # Transition parameters
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
    
    def forward(self, emissions, tags, mask=None):
        """Compute negative log likelihood"""
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)
        
        # emissions: (batch_size, seq_len, num_tags)
        # tags: (batch_size, seq_len)
        # mask: (batch_size, seq_len)
        
        score = self._compute_score(emissions, tags, mask)
        partition = self._compute_partition(emissions, mask)
        return partition - score
    
    def decode(self, emissions, mask=None):
        """Viterbi decoding"""
        if mask is None:
            mask = torch.ones(emissions.size()[:2], dtype=torch.bool, device=emissions.device)
        
        return self._viterbi_decode(emissions, mask)
    
    def _compute_score(self, emissions, tags, mask):
        batch_size, seq_len = tags.shape
        score = torch.zeros(batch_size, device=emissions.device)
        
        # Start transition score
        score += self.start_transitions[tags[:, 0]]
        
        for i in range(seq_len):
            if i > 0:
                # Transition score
                score += self.transitions[tags[:, i-1], tags[:, i]] * mask[:, i].float()
            # Emission score
            score += emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1) * mask[:, i].float()
        
        # End transition score
        last_tag_indices = mask.sum(1) - 1
        last_tags = tags.gather(1, last_tag_indices.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]
        
        return score
    
    def _compute_partition(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape
        
        # Initialize alphas
        alphas = self.start_transitions + emissions[:, 0]
        
        for i in range(1, seq_len):
            alphas_t = []
            for tag in range(num_tags):
                emit_score = emissions[:, i, tag].unsqueeze(1)
                trans_score = self.transitions[:, tag]
                next_tag_var = alphas + trans_score + emit_score
                alphas_t.append(torch.logsumexp(next_tag_var, dim=1))
            alphas = torch.stack(alphas_t, dim=1) * mask[:, i].unsqueeze(1) + alphas * (~mask[:, i]).unsqueeze(1)
        
        # Add end transitions
        alphas = alphas + self.end_transitions
        return torch.logsumexp(alphas, dim=1)
    
    def _viterbi_decode(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape
        
        # Initialize
        path_scores = self.start_transitions + emissions[:, 0]
        path_indices = []
        
        for i in range(1, seq_len):
            scores_with_trans = path_scores.unsqueeze(2) + self.transitions.unsqueeze(0)
            scores_with_emit = scores_with_trans + emissions[:, i].unsqueeze(1)
            
            path_scores, indices = torch.max(scores_with_emit, dim=1)
            path_scores = path_scores * mask[:, i].unsqueeze(1) + path_scores * (~mask[:, i]).unsqueeze(1)
            path_indices.append(indices)
        
        # Add end transitions
        path_scores = path_scores + self.end_transitions
        
        # Backtrack
        best_paths = []
        _, last_tags = torch.max(path_scores, dim=1)
        best_paths.append(last_tags)
        
        for i in range(seq_len - 2, -1, -1):
            last_tags = path_indices[i].gather(1, last_tags.unsqueeze(1)).squeeze(1)
            best_paths.append(last_tags)
        
        best_paths.reverse()
        return torch.stack(best_paths, dim=1)

# Complete Model with CRF
class CNNS5Model(nn.Module):
    def __init__(self, num_classes=5, cnn_channels=128, s5_dim=256, num_s5_layers=4, use_crf=True):
        super().__init__()
        
        # CNN feature extractor
        self.cnn = CNNFeatureExtractor(out_channels=cnn_channels)
        
        # Project CNN output to S5 dimension
        self.proj_in = nn.Linear(cnn_channels, s5_dim)
        
        s5_blocks = []
        for _ in range(num_s5_layers):
            # S5Block(in_dim, out_dim, use_dropout)
            s5_blocks.append(S5Block(s5_dim, s5_dim, False))
        self.s5_layers = nn.Sequential(*s5_blocks)
        
        # Output projection
        self.classifier = nn.Linear(s5_dim, num_classes)
        
        # CRF layer
        self.use_crf = use_crf
        if use_crf:
            self.crf = CRF(num_classes)
        
    def forward(self, x, tags=None, mask=None):
        # CNN feature extraction
        x = self.cnn(x)
        
        # Project to S5 dimension
        x = self.proj_in(x)
        
        # Apply S5 layers
        for s5_layer in self.s5_layers:
            x = s5_layer(x)
        
        # Get emissions
        emissions = self.classifier(x)
        
        if self.use_crf:
            if tags is not None:
                # Training: return negative log likelihood
                return -self.crf(emissions, tags, mask)
            else:
                # Inference: return decoded tags
                return self.crf.decode(emissions, mask)
        else:
            return emissions

# Dataset
class NucleotideDataset(Dataset):
    def __init__(self, npz_dir, split='train', train_ratio=0.8):
        self.npz_dir = npz_dir
        self.encoding_dict = {'A': 1, 'C': 2, 'G': 3, 'T': 4, '': 0}
        npz_dir =  os.path.join("datasets", "nuovi_npy")
        # Load all npz files
        self.samples = []
        npz_files = sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz')])
        
        for npz_file in npz_files:
            data = np.load(os.path.join(npz_dir, npz_file))
            num_samples = len(data['x'])
            
            for i in range(num_samples):
                x = data['x'][i].astype(np.float32)
                y = data['y'][i]
                
                # Encode y
                y_encoded = np.array([self.encoding_dict.get(char, 0) for char in y], dtype=np.int64)
                
                self.samples.append((x, y_encoded))
        
        # Split train/val
        n_train = int(len(self.samples) * train_ratio)
        if split == 'train':
            self.samples = self.samples[:n_train]
        else:
            self.samples = self.samples[n_train:]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)

# Training functions
def calculate_accuracy(predictions, targets, ignore_index=0):
    """Calculate accuracy ignoring padding tokens"""
    mask = targets != ignore_index
    correct = (predictions == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_acc = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        if model.use_crf:
            # Create mask for CRF (non-padding positions)
            mask = y != 0
            # Forward pass with CRF
            loss = model(x, y, mask).mean()
        else:
            # Forward pass without CRF
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            if model.use_crf:
                predictions = model(x, mask=mask)
            else:
                predictions = torch.argmax(model(x), dim=-1)
            acc = calculate_accuracy(predictions, y)
        
        total_loss += loss.item()
        total_acc += acc.item()
        total_samples += 1
        
        pbar.set_postfix({'loss': loss.item(), 'acc': acc.item()})
    
    return total_loss / total_samples, total_acc / total_samples

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            if model.use_crf:
                mask = y != 0
                loss = model(x, y, mask).mean()
                predictions = model(x, mask=mask)
            else:
                logits = model(x)
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                predictions = torch.argmax(logits, dim=-1)
            
            acc = calculate_accuracy(predictions, y)
            
            total_loss += loss.item()
            total_acc += acc.item()
            total_samples += 1
            
            pbar.set_postfix({'loss': loss.item(), 'acc': acc.item()})
    
    return total_loss / total_samples, total_acc / total_samples

def main():
    parser = argparse.ArgumentParser(description='Train CNN-S5 model for nucleotide classification')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--cnn_channels', type=int, default=128, help='CNN output channels')
    parser.add_argument('--s5_dim', type=int, default=256, help='S5 dimension')
    parser.add_argument('--num_s5_layers', type=int, default=4, help='Number of S5 layers')
    parser.add_argument('--npz_dir', type=str, default='/datasets/nuovi_npz', help='Directory containing npz files')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--use_crf', action='store_true', help='Use CRF layer instead of linear classifier')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = NucleotideDataset(args.npz_dir, split='train')
    val_dataset = NucleotideDataset(args.npz_dir, split='val')
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = CNNS5Model(
        num_classes=5,
        cnn_channels=args.cnn_channels,
        s5_dim=args.s5_dim,
        num_s5_layers=args.num_s5_layers,
        use_crf=args.use_crf
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': args
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"Saved best model with val_acc: {val_acc:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'args': args
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Save final history
    np.save(os.path.join(args.output_dir, 'history.npy'), history)
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
