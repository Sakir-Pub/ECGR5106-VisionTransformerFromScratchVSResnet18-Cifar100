import argparse
import os
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset_loader import CIFAR100DataLoader
from model import create_model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer for updating weights
        device: Device to train on (CPU or CUDA)
        
    Returns:
        tuple: (average_loss, accuracy, time_taken)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Start timing
    start_time = time.time()
    
    # Progress bar
    pbar = tqdm(train_loader, desc="Training")
    
    for inputs, targets in pbar:
        # Move data to device
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs).logits
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    # End timing
    end_time = time.time()
    time_taken = end_time - start_time
    
    # Calculate metrics
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, time_taken


def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on (CPU or CUDA)
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for inputs, targets in pbar:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs).logits
            loss = criterion(outputs, targets)
            
            # Update statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
    
    # Calculate metrics
    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train_model(model_name, num_epochs, batch_size=32, learning_rate=2e-5, 
                data_dir='./data', output_dir='./output', seed=42):
    """
    Train a model with the specified parameters.
    
    Args:
        model_name (str): Name of the model to train
        num_epochs (int): Number of epochs to train for
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for the optimizer
        data_dir (str): Directory for data
        output_dir (str): Directory for saving outputs
        seed (int): Random seed for reproducibility
        
    Returns:
        dict: Training metrics
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    data_loader = CIFAR100DataLoader(data_dir=data_dir, batch_size=batch_size, seed=seed)
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()
    
    # Create model
    print(f"Creating model: {model_name}")
    model_wrapper = create_model(model_name, num_classes=100)
    model = model_wrapper.get_model()
    model.to(device)
    
    # Print model information
    print(model_wrapper.get_model_info())
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs...")
    
    # For tracking metrics
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'time_per_epoch': [],
        'model_name': model_name,
        'total_parameters': model_wrapper.get_total_parameters(),
        'trainable_parameters': model_wrapper.get_trainable_parameters(),
        'best_val_acc': 0.0,
        'best_epoch': 0
    }
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # Train for one epoch
        train_loss, train_acc, time_taken = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Time: {time_taken:.2f}s")
        
        # Update metrics
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        metrics['time_per_epoch'].append(time_taken)
        
        # Save best model
        if val_acc > metrics['best_val_acc']:
            metrics['best_val_acc'] = val_acc
            metrics['best_epoch'] = epoch
            
            # Save model
            checkpoint_path = os.path.join(model_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"Saved best model checkpoint to {checkpoint_path}")
    
    # Save the final model
    final_checkpoint_path = os.path.join(model_dir, 'final_model.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
    }, final_checkpoint_path)
    print(f"Saved final model checkpoint to {final_checkpoint_path}")
    
    # Save metrics
    metrics_file = os.path.join(model_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Training complete. Metrics saved to {metrics_file}")
    
    # Quick test evaluation
    print("\nEvaluating on test set...")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    # Update metrics with test results
    metrics['test_loss'] = test_loss
    metrics['test_acc'] = test_acc
    
    # Save updated metrics
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics


def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(description='Train Swin Transformer models on CIFAR-100')
    
    parser.add_argument('--model', type=str, default='swin_tiny_pretrained',
                        choices=['swin_tiny_pretrained', 'swin_small_pretrained', 'swin_tiny_scratch'],
                        help='Model to train')
    
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train for')
    
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory for data')
    
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Directory for saving outputs')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Run training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{args.model}_{timestamp}"
    
    train_model(
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()