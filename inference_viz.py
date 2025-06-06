import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from CNNS5Model import CNNS5Model, NucleotideDataset
import seaborn as sns

def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    
    model = CNNS5Model(
        num_classes=5,
        cnn_channels=args.cnn_channels,
        s5_dim=args.s5_dim,
        num_s5_layers=args.num_s5_layers
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, args

def predict_sample(model, x, device='cuda'):
    """Predict nucleotides for a single sample"""
    model.eval()
    decoding_dict = {0: '', 1: 'A', 2: 'C', 3: 'G', 4: 'T'}
    
    with torch.no_grad():
        x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(device)
        logits = model(x_tensor)
        predictions = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
    
    # Decode predictions
    predicted_sequence = ''.join([decoding_dict[p] for p in predictions if p != 0])
    
    return predictions, predicted_sequence

def visualize_prediction(x, y_true, y_pred, save_path=None):
    """Visualize signal with ground truth and predictions"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Plot signal
    axes[0].plot(x, linewidth=0.5)
    axes[0].set_title('Electrical Signal')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Current')
    axes[0].grid(True, alpha=0.3)
    
    # Plot ground truth
    colors = {'': 'white', 'A': 'red', 'C': 'blue', 'G': 'green', 'T': 'orange'}
    decoding_dict = {0: '', 1: 'A', 2: 'C', 3: 'G', 4: 'T'}
    
    # Ground truth
    gt_colors = [colors[decoding_dict[y]] for y in y_true]
    axes[1].scatter(range(len(y_true)), np.ones(len(y_true)), c=gt_colors, s=5, marker='|')
    axes[1].set_title('Ground Truth Nucleotides')
    axes[1].set_ylim(0.5, 1.5)
    axes[1].set_yticks([])
    
    # Predictions
    pred_colors = [colors[decoding_dict[y]] for y in y_pred]
    axes[2].scatter(range(len(y_pred)), np.ones(len(y_pred)), c=pred_colors, s=5, marker='|')
    axes[2].set_title('Predicted Nucleotides')
    axes[2].set_ylim(0.5, 1.5)
    axes[2].set_yticks([])
    axes[2].set_xlabel('Position')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[nuc], label=nuc if nuc else 'Padding') 
                      for nuc in ['A', 'C', 'G', 'T', '']]
    axes[2].legend(handles=legend_elements, loc='upper right', ncol=5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def calculate_confusion_matrix(model, dataloader, device='cuda'):
    """Calculate confusion matrix for the model"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            predictions = torch.argmax(logits, dim=-1)
            
            # Flatten and filter out padding
            mask = y != 0
            all_preds.extend(predictions[mask].cpu().numpy())
            all_targets.extend(y[mask].cpu().numpy())
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_targets, all_preds, labels=[1, 2, 3, 4])
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['A', 'C', 'G', 'T'],
                yticklabels=['A', 'C', 'G', 'T'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return cm

def analyze_sequence_accuracy(y_true, y_pred):
    """Analyze accuracy at sequence level"""
    # Remove padding
    mask = y_true != 0
    y_true_seq = y_true[mask]
    y_pred_seq = y_pred[mask]
    
    # Calculate metrics
    correct = (y_true_seq == y_pred_seq).sum()
    total = len(y_true_seq)
    accuracy = correct / total
    
    # Per-nucleotide accuracy
    nucleotide_acc = {}
    for nuc_id, nuc_name in [(1, 'A'), (2, 'C'), (3, 'G'), (4, 'T')]:
        mask = y_true_seq == nuc_id
        if mask.sum() > 0:
            nucleotide_acc[nuc_name] = (y_pred_seq[mask] == nuc_id).sum() / mask.sum()
    
    return {
        'overall_accuracy': float(accuracy),
        'correct_predictions': int(correct),
        'total_nucleotides': int(total),
        'per_nucleotide_accuracy': nucleotide_acc
    }

def main():
    parser = argparse.ArgumentParser(description='Inference and visualization for CNN-S5 model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--npz_file', type=str, help='Specific npz file for inference')
    parser.add_argument('--sample_idx', type=int, default=0, help='Sample index within npz file')
    parser.add_argument('--output_dir', type=str, default='./inference_outputs', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model, train_args = load_model(args.checkpoint, args.device)
    
    if args.npz_file:
        # Single sample inference
        data = np.load(args.npz_file)
        x = data['x'][args.sample_idx].astype(np.float32)
        y_str = data['y'][args.sample_idx]
        
        # Encode ground truth
        encoding_dict = {'A': 1, 'C': 2, 'G': 3, 'T': 4, '': 0}
        y_true = np.array([encoding_dict.get(char, 0) for char in y_str], dtype=np.int64)
        
        # Predict
        print("Running inference...")
        y_pred, predicted_sequence = predict_sample(model, x, args.device)
        
        # Analyze accuracy
        metrics = analyze_sequence_accuracy(y_true, y_pred)
        print(f"\nSequence Metrics:")
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"Correct Predictions: {metrics['correct_predictions']}/{metrics['total_nucleotides']}")
        print(f"\nPer-nucleotide Accuracy:")
        for nuc, acc in metrics['per_nucleotide_accuracy'].items():
            print(f"  {nuc}: {acc:.4f}")
        
        # Extract true sequence (without padding)
        true_sequence = ''.join([char for char in y_str if char != ''])
        print(f"\nTrue Sequence Length: {len(true_sequence)}")
        print(f"Predicted Sequence Length: {len(predicted_sequence)}")
        
        # Save sequences
        with open(os.path.join(args.output_dir, 'sequences.txt'), 'w') as f:
            f.write(f"True Sequence:\n{true_sequence}\n\n")
            f.write(f"Predicted Sequence:\n{predicted_sequence}\n\n")
            f.write(f"Metrics:\n{json.dumps(metrics, indent=2)}")
        
        # Visualize
        visualize_prediction(x, y_true, y_pred, 
                           save_path=os.path.join(args.output_dir, 'prediction_visualization.png'))
    
    else:
        # Full dataset evaluation
        print("Evaluating on validation set...")
        val_dataset = NucleotideDataset(train_args.npz_dir, split='val')
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Calculate confusion matrix
        cm = calculate_confusion_matrix(model, val_loader, args.device)
        
        # Save confusion matrix
        np.save(os.path.join(args.output_dir, 'confusion_matrix.npy'), cm)
        plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()
