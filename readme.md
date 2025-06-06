# CNN-S5 Nucleotide Classification from Electrical Signals

This project implements a hybrid CNN-S5 (State Space Model) architecture for classifying nucleotides (A, C, G, T) from electrical signal data.

## Architecture Overview

The model consists of:
1. **CNN Feature Extractor**: Extracts local features from the electrical signal
2. **S5 Layers**: Stack of State Space Model layers for capturing long-range dependencies
3. **Linear Classifier**: Maps S5 outputs to nucleotide predictions

## Project Structure

```
├── CNNS5Model.py          # Main model implementation and training script
├── inference_viz.py       # Inference and visualization utilities
├── data_utils.py          # Data analysis and preprocessing tools
├── datasets/
│   └── nuovi_npz/        # Directory containing .npz data files
└── outputs/              # Training outputs and checkpoints
```

## Installation

```bash
pip install torch torchvision numpy matplotlib tqdm scikit-learn seaborn scipy
```

## Data Format

Each `.npz` file contains:
- `x`: Electrical signal array of shape (num_samples, 2000) with float values
- `y`: Ground truth nucleotide sequences of shape (num_samples, 2000) with characters ['A', 'C', 'G', 'T', '']

The empty string '' represents padding. The actual nucleotide sequence consists only of A, C, G, T characters.

## Usage

### 1. Data Analysis

Analyze your dataset statistics:

```bash
python data_utils.py --npz_dir /datasets/nuovi_npz --action analyze --output_dir ./data_analysis
```

Visualize signal segments:

```bash
python data_utils.py --npz_dir /datasets/nuovi_npz --action visualize --npz_file /path/to/specific.npz
```

### 2. Training

Basic training:

```bash
python CNNS5Model.py --npz_dir /datasets/nuovi_npz --epochs 50 --batch_size 32
```

Full training with custom parameters:

```bash
python CNNS5Model.py \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.001 \
    --cnn_channels 128 \
    --s5_dim 256 \
    --num_s5_layers 4 \
    --npz_dir /datasets/nuovi_npz \
    --output_dir ./outputs \
    --device cuda
```

### 3. Inference

Single sample inference:

```bash
python inference_viz.py \
    --checkpoint ./outputs/best_model.pth \
    --npz_file /path/to/test.npz \
    --sample_idx 0 \
    --output_dir ./inference_outputs
```

Full validation set evaluation:

```bash
python inference_viz.py \
    --checkpoint ./outputs/best_model.pth \
    --output_dir ./inference_outputs
```

## Model Parameters

- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 32)
- `--lr`: Learning rate (default: 1e-3)
- `--cnn_channels`: Number of CNN output channels (default: 128)
- `--s5_dim`: S5 model dimension (default: 256)
- `--num_s5_layers`: Number of stacked S5 layers (default: 4)

## Output Files

### Training Outputs
- `best_model.pth`: Best model checkpoint based on validation accuracy
- `checkpoint_epoch_X.pth`: Periodic checkpoints during training
- `history.npy`: Training history (loss and accuracy curves)
- `config.json`: Training configuration

### Inference Outputs
- `sequences.txt`: True vs predicted sequences
- `prediction_visualization.png`: Visualization of signal with predictions
- `confusion_matrix.png`: Confusion matrix for nucleotide classification
- `confusion_matrix.npy`: Raw confusion matrix data

## Model Architecture Details

### CNN Feature Extractor
- Multiple 1D convolutional layers with varying kernel sizes [7, 5, 3]
- Batch normalization and ReLU activation
- Dropout for regularization

### S5 State Space Model
- Learnable state matrices (A, B, C, D)
- Adaptive timestep parameter
- Residual connections and layer normalization
- Efficient sequence processing with state-space formulation

### Training Strategy
- Adam optimizer with learning rate scheduling
- Cross-entropy loss with padding token ignored
- Early stopping based on validation accuracy
- Data augmentation options available

## Performance Metrics

The model evaluates:
- Overall sequence accuracy
- Per-nucleotide accuracy (A, C, G, T)
- Confusion matrix
- Position-wise accuracy visualization

## Tips for Better Performance

1. **Data Preprocessing**: Use `preprocess_signal()` in data_utils.py for signal normalization
2. **Data Augmentation**: Apply noise and shift augmentation during training
3. **Hyperparameter Tuning**: Experiment with S5 dimensions and number of layers
4. **Learning Rate Schedule**: The model uses ReduceLROnPlateau for adaptive learning
5. **Batch Size**: Larger batch sizes (64-128) often improve stability

## Troubleshooting

- **Out of Memory**: Reduce batch_size or s5_dim
- **Poor Convergence**: Try different learning rates or more S5 layers
- **Overfitting**: Increase dropout or use data augmentation

## Citation

If you use this code, please cite:
```
CNN-S5 Model for Nucleotide Classification from Electrical Signals
[Your citation here]
```