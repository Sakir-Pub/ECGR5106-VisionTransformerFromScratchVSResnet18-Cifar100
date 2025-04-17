import os
import time
import argparse
import csv
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from dataset_loader import get_cifar100_data_loaders
from model import create_model, count_parameters

# Try to import torchinfo for model summary
try:
    from torchinfo import summary
    HAS_TORCHINFO = True
except ImportError:
    HAS_TORCHINFO = False
    print("torchinfo not found. Install with 'pip install torchinfo' for detailed model summary.")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a ViT or ResNet on CIFAR-100')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='vit', choices=['vit', 'resnet18'],
                        help='Model architecture (vit or resnet18)')
    parser.add_argument('--patch_size', type=int, default=4,
                        help='Patch size for ViT (4 or 8)')
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='Embedding dimension for ViT (256 or 512)')
    parser.add_argument('--depth', type=int, default=4,
                        help='Number of transformer layers (4 or 8)')
    parser.add_argument('--heads', type=int, default=2,
                        help='Number of attention heads (2 or 4)')
    parser.add_argument('--mlp_ratio', type=float, default=2.0,
                        help='MLP ratio (2.0 or 4.0)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training and validation')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--data_aug', action='store_true',
                        help='Use data augmentation')
    
    # Other options
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs')
    parser.add_argument('--eval_freq', type=int, default=1,
                        help='Evaluation frequency (epochs)')
    
    return parser.parse_args()


def train(model, train_loader, optimizer, criterion, device, epoch):
    """Train the model for one epoch."""
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    # Start time for epoch
    start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update metrics
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Print progress
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"Epoch: {epoch} | Batch: {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {train_loss/(batch_idx+1):.3f} | "
                  f"Acc: {100.*correct/total:.2f}% ({correct}/{total})")
    
    # Calculate epoch metrics
    train_loss = train_loss / len(train_loader)
    train_acc = 100. * correct / total
    
    # Calculate epoch time
    epoch_time = time.time() - start_time
    
    return train_loss, train_acc, epoch_time


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Update metrics
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # Calculate validation metrics
    val_loss = val_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def count_flops(model, input_size=(1, 3, 32, 32), device='cuda'):
    """Estimate FLOPs using torchinfo."""
    if not HAS_TORCHINFO:
        print("torchinfo not available, skipping FLOPs calculation")
        return None
    
    try:
        stats = summary(model, input_size=input_size, device=device, verbose=0)
        return stats.total_mult_adds
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        return None


def create_log_file(args, model_params):
    """Create a CSV log file and return the file handle."""
    # Create directories if they don't exist
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a filename based on model configuration
    if args.model == 'vit':
        filename = f"vit_p{args.patch_size}_e{args.embed_dim}_d{args.depth}_h{args.heads}_{timestamp}.csv"
    else:
        filename = f"resnet18_{timestamp}.csv"
    
    filepath = os.path.join(args.log_dir, filename)
    
    # Open the file and create a CSV writer
    file = open(filepath, 'w', newline='')
    writer = csv.writer(file)
    
    # Write header
    header = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 
              'learning_rate', 'epoch_time']
    writer.writerow(header)
    
    # Write model configuration
    writer.writerow(['Model Configuration'])
    writer.writerow(['Model Type', args.model])
    if args.model == 'vit':
        writer.writerow(['Patch Size', args.patch_size])
        writer.writerow(['Embedding Dimension', args.embed_dim])
        writer.writerow(['Depth', args.depth])
        writer.writerow(['Heads', args.heads])
        writer.writerow(['MLP Ratio', args.mlp_ratio])
    writer.writerow(['Parameters', model_params])
    writer.writerow(['Batch Size', args.batch_size])
    writer.writerow(['Learning Rate', args.lr])
    writer.writerow([])  # Empty row before data
    writer.writerow(header)  # Repeat header before actual data
    
    return file, writer


def save_checkpoint(model, optimizer, epoch, args, val_acc, model_params, flops):
    """Save model checkpoint."""
    os.makedirs(args.save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.model == 'vit':
        filename = f"vit_p{args.patch_size}_e{args.embed_dim}_d{args.depth}_h{args.heads}_ep{epoch}_acc{val_acc:.2f}.pth"
    else:
        filename = f"resnet18_ep{epoch}_acc{val_acc:.2f}.pth"
    
    filepath = os.path.join(args.save_dir, filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'configuration': {
            'model_type': args.model,
            'parameters': model_params,
            'flops': flops
        }
    }
    
    if args.model == 'vit':
        checkpoint['configuration'].update({
            'patch_size': args.patch_size,
            'embed_dim': args.embed_dim,
            'depth': args.depth,
            'heads': args.heads,
            'mlp_ratio': args.mlp_ratio
        })
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def plot_training_curves(train_losses, train_accs, val_losses, val_accs, args):
    """Plot training and validation curves."""
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    # Save the plot
    os.makedirs('plots', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.model == 'vit':
        filename = f"vit_p{args.patch_size}_e{args.embed_dim}_d{args.depth}_h{args.heads}_{timestamp}.png"
    else:
        filename = f"resnet18_{timestamp}.png"
    
    filepath = os.path.join('plots', filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Training curves saved to {filepath}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_cifar100_data_loaders(
        batch_size=args.batch_size,
        augment=args.data_aug
    )
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    model_kwargs = {}
    if args.model == 'vit':
        model_kwargs = {
            'patch_size': args.patch_size,
            'embed_dim': args.embed_dim,
            'depth': args.depth,
            'num_heads': args.heads,
            'mlp_ratio': args.mlp_ratio,
            'drop_rate': args.dropout
        }
    
    model = create_model(args.model, **model_kwargs)
    model = model.to(device)
    
    # Count parameters and FLOPs
    model_params = count_parameters(model)
    print(f"Model parameters: {model_params:,}")
    
    flops = count_flops(model, input_size=(1, 3, 32, 32), device=device)
    if flops:
        print(f"Model FLOPs: {flops:,}")
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Create log file
    log_file, log_writer = create_log_file(args, model_params)
    
    # Track metrics
    best_val_acc = 0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc, epoch_time = train(
            model, train_loader, optimizer, criterion, device, epoch
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        if epoch % args.eval_freq == 0:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            print(f"Epoch {epoch} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}% | "
                  f"Time: {epoch_time:.2f}s")
            
            # Save checkpoint if best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(model, optimizer, epoch, args, val_acc, model_params, flops)
        else:
            # If not validating, use previous values
            val_loss = val_losses[-1] if val_losses else float('inf')
            val_acc = val_accs[-1] if val_accs else 0
            
            print(f"Epoch {epoch} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}% | "
                  f"Time: {epoch_time:.2f}s")
        
        # Log metrics
        log_writer.writerow([
            epoch, train_loss, train_acc, val_loss, val_acc, args.lr, epoch_time
        ])
        log_file.flush()  # Ensure metrics are written immediately
    
    # Final evaluation on test set
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    
    # Save final model
    save_checkpoint(model, optimizer, args.epochs, args, test_acc, model_params, flops)
    
    # Plot training curves
    plot_training_curves(train_losses, train_accs, val_losses, val_accs, args)
    
    # Close log file
    log_file.close()
    
    # Summary of results
    print("\n" + "="*50)
    print("Training Summary:")
    print(f"Model: {args.model}")
    if args.model == 'vit':
        print(f"Configuration: patch_size={args.patch_size}, embed_dim={args.embed_dim}, "
              f"depth={args.depth}, heads={args.heads}, mlp_ratio={args.mlp_ratio}")
    print(f"Parameters: {model_params:,}")
    if flops:
        print(f"FLOPs: {flops:,}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print("="*50)


if __name__ == "__main__":
    main()