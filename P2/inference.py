import argparse
import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from pathlib import Path

from dataset_loader import CIFAR100DataLoader
from model import create_model


def load_model(model_name, model_path, device):
    """
    Load a saved model.
    
    Args:
        model_name (str): Name of the model architecture
        model_path (str): Path to the saved model checkpoint
        device (torch.device): Device to load the model on
        
    Returns:
        nn.Module: The loaded model
    """
    # Create model architecture
    model_wrapper = create_model(model_name, num_classes=100)
    model = model_wrapper.get_model()
    
    # Load saved weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to device
    model.to(device)
    
    return model, model_wrapper


def evaluate_model(model, test_loader, device):
    """
    Evaluate a model on the test set.
    
    Args:
        model (nn.Module): The model to evaluate
        test_loader (DataLoader): DataLoader for the test data
        device (torch.device): Device to evaluate on
        
    Returns:
        tuple: (accuracy, loss, time_taken)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    correct = 0
    total = 0
    running_loss = 0.0
    
    # Class-wise correct predictions
    class_correct = [0] * 100
    class_total = [0] * 100
    
    # Start timing
    start_time = time.time()
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs).logits
            loss = criterion(outputs, targets)
            
            # Update statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Class-wise accuracy
            correct_tensor = predicted.eq(targets)
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_correct[label] += correct_tensor[i].item()
                class_total[label] += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
    
    # End timing
    end_time = time.time()
    time_taken = end_time - start_time
    
    # Calculate metrics
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(test_loader)
    
    # Calculate class-wise accuracy
    class_accuracy = [(100. * c / t) if t > 0 else 0 for c, t in zip(class_correct, class_total)]
    
    return accuracy, avg_loss, time_taken, class_accuracy


def load_metrics(output_dir, model_name):
    """
    Load training metrics from file.
    
    Args:
        output_dir (str): Directory containing model outputs
        model_name (str): Name of the model
        
    Returns:
        dict: Training metrics
    """
    metrics_file = os.path.join(output_dir, model_name, 'metrics.json')
    
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}")
        return None
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def compare_models(models_data, output_dir):
    """
    Compare multiple models and save the results.
    
    Args:
        models_data (list): List of dictionaries containing model data
        output_dir (str): Directory to save comparison results
    """
    # Create comparison table
    comparison_data = []
    
    for model_data in models_data:
        model_name = model_data['model_name']
        test_acc = model_data.get('test_acc', 0)
        avg_time = sum(model_data.get('time_per_epoch', [0])) / max(len(model_data.get('time_per_epoch', [1])), 1)
        total_params = model_data.get('total_parameters', 0)
        trainable_params = model_data.get('trainable_parameters', 0)
        trainable_percent = 100 * trainable_params / total_params if total_params > 0 else 0
        
        comparison_data.append({
            'Model': model_name,
            'Test Accuracy (%)': round(test_acc, 2),
            'Avg Time per Epoch (s)': round(avg_time, 2),
            'Total Parameters': f"{total_params:,}",
            'Trainable Parameters': f"{trainable_params:,} ({trainable_percent:.2f}%)"
        })
    
    # Create DataFrame and save as CSV
    df = pd.DataFrame(comparison_data)
    csv_path = os.path.join(output_dir, 'model_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved comparison results to {csv_path}")
    
    # Print table
    print("\nModel Comparison:")
    print(tabulate(df, headers='keys', tablefmt='pretty'))
    
    # Create performance plot
    plt.figure(figsize=(12, 6))
    
    # Plot test accuracy
    plt.subplot(1, 2, 1)
    model_names = [data['Model'] for data in comparison_data]
    accuracies = [data['Test Accuracy (%)'] for data in comparison_data]
    plt.bar(model_names, accuracies)
    plt.title('Test Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Plot average time per epoch
    plt.subplot(1, 2, 2)
    times = [data['Avg Time per Epoch (s)'] for data in comparison_data]
    plt.bar(model_names, times)
    plt.title('Average Time per Epoch')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(plot_path)
    print(f"Saved comparison plot to {plot_path}")


def main():
    """Main entry point for the inference script."""
    parser = argparse.ArgumentParser(description='Evaluate Swin Transformer models on CIFAR-100')
    
    parser.add_argument('--models', type=str, nargs='+',
                        default=['swin_tiny_pretrained', 'swin_small_pretrained', 'swin_tiny_scratch'],
                        help='Models to evaluate')
    
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory for data')
    
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Directory containing model outputs')
    
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    data_loader = CIFAR100DataLoader(data_dir=args.data_dir, batch_size=args.batch_size)
    _, _, test_loader = data_loader.get_data_loaders()
    
    # Collect all model metrics
    all_model_data = []
    
    # Evaluate each model
    for model_name in args.models:
        print(f"\nEvaluating model: {model_name}")
        
        # Find model directory
        model_dir = os.path.join(args.output_dir, model_name)
        if not os.path.exists(model_dir):
            print(f"Model directory not found: {model_dir}")
            # Look for subdirectories that start with the model name
            matching_dirs = [d for d in os.listdir(args.output_dir) 
                           if os.path.isdir(os.path.join(args.output_dir, d)) and d.startswith(model_name)]
            
            if matching_dirs:
                # Use the most recent directory
                model_dir = os.path.join(args.output_dir, sorted(matching_dirs)[-1])
                print(f"Using most recent matching directory: {model_dir}")
            else:
                print(f"No matching directories found for {model_name}")
                continue
        
        # Find best model checkpoint
        model_path = os.path.join(model_dir, 'best_model.pth')
        if not os.path.exists(model_path):
            print(f"Model checkpoint not found: {model_path}")
            continue
        
        # Load the model
        try:
            model, model_wrapper = load_model(model_name, model_path, device)
            print(model_wrapper.get_model_info())
        except Exception as e:
            print(f"Error loading model: {e}")
            continue
        
        # Evaluate the model
        test_acc, test_loss, eval_time, class_accuracy = evaluate_model(model, test_loader, device)
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Evaluation Time: {eval_time:.2f}s")
        
        # Load training metrics
        metrics = load_metrics(args.output_dir, model_name)
        if metrics:
            # Update with test results if not already present
            if 'test_acc' not in metrics:
                metrics['test_acc'] = test_acc
                metrics['test_loss'] = test_loss
                
                # Save updated metrics
                metrics_file = os.path.join(model_dir, 'metrics.json')
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=4)
        else:
            # Create metrics dictionary if not loaded
            metrics = {
                'model_name': model_name,
                'test_acc': test_acc,
                'test_loss': test_loss,
                'total_parameters': model_wrapper.get_total_parameters(),
                'trainable_parameters': model_wrapper.get_trainable_parameters(),
                'time_per_epoch': [eval_time]
            }
        
        # Add to collection
        all_model_data.append(metrics)
    
    # Compare models
    if all_model_data:
        compare_models(all_model_data, args.output_dir)
    else:
        print("No models were successfully evaluated.")


if __name__ == "__main__":
    main()