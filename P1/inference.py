import os
import argparse
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_loader import get_cifar100_data_loaders, get_class_names
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
    parser = argparse.ArgumentParser(description='Inference for ViT or ResNet on CIFAR-100')
    
    # Model checkpoint or configuration
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='vit', choices=['vit', 'resnet18'],
                        help='Model architecture if no checkpoint provided')
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
    
    # Inference options
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for inference')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize model predictions')
    parser.add_argument('--compare_models', action='store_true',
                        help='Compare multiple models from results directory')
    
    return parser.parse_args()


def load_model(args, device):
    """Load a model from checkpoint or create a new one."""
    if args.checkpoint:
        # Load checkpoint
        print(f"Loading model from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Get configuration from checkpoint
        config = checkpoint.get('configuration', {})
        model_type = config.get('model_type', args.model)
        
        # Create model based on configuration
        if model_type == 'vit':
            model_kwargs = {
                'patch_size': config.get('patch_size', args.patch_size),
                'embed_dim': config.get('embed_dim', args.embed_dim),
                'depth': config.get('depth', args.depth),
                'num_heads': config.get('heads', args.heads),
                'mlp_ratio': config.get('mlp_ratio', args.mlp_ratio)
            }
            model = create_model('vit', **model_kwargs)
        else:
            model = create_model('resnet18')
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Return configuration details
        config_details = {
            'model_type': model_type,
            'params': count_parameters(model)
        }
        if model_type == 'vit':
            config_details.update(model_kwargs)
        
    else:
        # Create new model based on arguments
        print(f"Creating new {args.model} model")
        if args.model == 'vit':
            model_kwargs = {
                'patch_size': args.patch_size,
                'embed_dim': args.embed_dim,
                'depth': args.depth,
                'num_heads': args.heads,
                'mlp_ratio': args.mlp_ratio
            }
            model = create_model('vit', **model_kwargs)
            config_details = {
                'model_type': 'vit',
                'params': count_parameters(model),
                **model_kwargs
            }
        else:
            model = create_model('resnet18')
            config_details = {
                'model_type': 'resnet18',
                'params': count_parameters(model)
            }
    
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    return model, config_details


def evaluate_model(model, data_loader, device, class_names=None):
    """Evaluate model on a dataset."""
    print("Evaluating model...")
    
    # Initialize metrics
    correct = 0
    total = 0
    loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    # Lists to store predictions and ground truth
    all_preds = []
    all_targets = []
    all_probs = []
    
    # Track inference time
    start_time = time.time()
    
    # Evaluate model
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            batch_loss = criterion(outputs, targets)
            loss += batch_loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Store predictions and targets
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())
    
    # Calculate metrics
    accuracy = 100 * correct / total
    avg_loss = loss / len(data_loader)
    inference_time = time.time() - start_time
    
    # Print results
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Inference Time: {inference_time:.2f} seconds")
    print(f"Time per sample: {inference_time/total*1000:.2f} ms")
    
    # Create classification report
    if class_names:
        report = classification_report(
            all_targets, all_preds, 
            target_names=class_names[:100], 
            output_dict=True
        )
    else:
        report = classification_report(
            all_targets, all_preds,
            output_dict=True
        )
    
    # Prepare results
    results = {
        'accuracy': accuracy,
        'loss': avg_loss,
        'inference_time': inference_time,
        'time_per_sample': inference_time/total,
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs,
        'classification_report': report
    }
    
    return results


def visualize_predictions(inputs, targets, predictions, class_names, num_samples=25):
    """Visualize model predictions."""
    # Denormalize images for visualization
    mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
    std = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)
    
    # Select a subset of images
    indices = np.random.choice(len(inputs), min(num_samples, len(inputs)), replace=False)
    
    # Create figure
    num_cols = 5
    num_rows = (len(indices) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3*num_rows))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        img = inputs[idx].cpu()
        img = img * std + mean  # Denormalize
        img = img.permute(1, 2, 0).numpy()  # Change from CxHxW to HxWxC
        img = np.clip(img, 0, 1)
        
        pred = predictions[idx]
        target = targets[idx]
        
        # Display image
        axes[i].imshow(img)
        
        # Set title color based on correctness
        title_color = 'green' if pred == target else 'red'
        
        # Add title with prediction and ground truth
        pred_name = class_names[pred] if pred < len(class_names) else f"Class {pred}"
        target_name = class_names[target] if target < len(class_names) else f"Class {target}"
        
        axes[i].set_title(f"Pred: {pred_name}\nTrue: {target_name}", 
                          color=title_color, fontsize=9)
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def calculate_flops(model, input_shape=(1, 3, 32, 32), device='cuda'):
    """Calculate FLOPs for a model using torchinfo."""
    if not HAS_TORCHINFO:
        print("torchinfo not available. Install with pip install torchinfo")
        return None
    
    try:
        stats = summary(model, input_size=input_shape, verbose=0, device=device)
        return stats.total_mult_adds
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        return None


def save_results(results, config, args):
    """Save evaluation results to file."""
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Create filename based on configuration
    if config['model_type'] == 'vit':
        filename = f"vit_p{config.get('patch_size')}_e{config.get('embed_dim')}_d{config.get('depth')}_h{config.get('num_heads', config.get('heads'))}.json"
    else:
        filename = "resnet18.json"
    
    filepath = os.path.join(args.results_dir, filename)
    
    # Prepare data for saving
    save_data = {
        'config': config,
        'results': {
            'accuracy': results['accuracy'],
            'loss': results['loss'],
            'inference_time': results['inference_time'],
            'time_per_sample': results['time_per_sample'],
        },
        'classification_report': results['classification_report']
    }
    
    # Save to JSON file
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=4)
    
    print(f"Results saved to {filepath}")
    
    return filepath


def compare_models(results_dir):
    """Compare multiple model results from the results directory."""
    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} does not exist")
        return
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"No result files found in {results_dir}")
        return
    
    # Load all results
    all_results = []
    for file in json_files:
        with open(os.path.join(results_dir, file), 'r') as f:
            data = json.load(f)
            
            # Extract key information
            model_info = {
                'model_type': data['config']['model_type']
            }
            
            if model_info['model_type'] == 'vit':
                model_info.update({
                    'patch_size': data['config'].get('patch_size', 'N/A'),
                    'embed_dim': data['config'].get('embed_dim', 'N/A'),
                    'depth': data['config'].get('depth', 'N/A'),
                    'heads': data['config'].get('num_heads', data['config'].get('heads', 'N/A')),
                    'mlp_ratio': data['config'].get('mlp_ratio', 'N/A'),
                })
            
            model_info.update({
                'accuracy': data['results']['accuracy'],
                'params': data['config'].get('params', 'N/A'),
                'flops': data['config'].get('flops', 'N/A'),
                'time_per_sample': data['results']['time_per_sample'] * 1000  # Convert to ms
            })
            
            all_results.append(model_info)
    
    # Convert to DataFrame for easy analysis
    df = pd.DataFrame(all_results)
    
    # Print comparison table
    print("\n=== Model Comparison ===")
    
    if 'vit' in df['model_type'].values and 'resnet18' in df['model_type'].values:
        # If we have both ViT and ResNet models, sort by model type first
        df = df.sort_values(['model_type', 'accuracy'], ascending=[True, False])
    else:
        # Otherwise, just sort by accuracy
        df = df.sort_values('accuracy', ascending=False)
    
    # Format the dataframe for display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print(df.to_string(index=False))
    
    # Create a comparison plot
    plt.figure(figsize=(12, 6))
    
    # Plot accuracy vs parameters
    plt.subplot(1, 2, 1)
    vit_mask = df['model_type'] == 'vit'
    resnet_mask = df['model_type'] == 'resnet18'
    
    if any(vit_mask):
        plt.scatter(df.loc[vit_mask, 'params'], df.loc[vit_mask, 'accuracy'], 
                    label='ViT', marker='o', s=100)
    
    if any(resnet_mask):
        plt.scatter(df.loc[resnet_mask, 'params'], df.loc[resnet_mask, 'accuracy'], 
                    label='ResNet-18', marker='x', s=100)
    
    plt.xlabel('Parameters')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Parameters')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot accuracy vs inference time
    plt.subplot(1, 2, 2)
    if any(vit_mask):
        plt.scatter(df.loc[vit_mask, 'time_per_sample'], df.loc[vit_mask, 'accuracy'], 
                    label='ViT', marker='o', s=100)
    
    if any(resnet_mask):
        plt.scatter(df.loc[resnet_mask, 'time_per_sample'], df.loc[resnet_mask, 'accuracy'], 
                    label='ResNet-18', marker='x', s=100)
    
    plt.xlabel('Time per Sample (ms)')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Inference Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_comparison.png'))
    print(f"Comparison plot saved to {os.path.join(results_dir, 'model_comparison.png')}")
    
    # Return the comparison DataFrame
    return df


def main():
    """Main function for inference."""
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Compare models if requested
    if args.compare_models:
        compare_models(args.results_dir)
        return
    
    # Load model
    model, config = load_model(args, device)
    
    # Print model information
    print("\nModel Information:")
    print(f"Model Type: {config['model_type']}")
    print(f"Parameters: {config['params']:,}")
    
    if config['model_type'] == 'vit':
        print(f"Patch Size: {config['patch_size']}")
        print(f"Embedding Dimension: {config['embed_dim']}")
        print(f"Depth: {config['depth']}")
        print(f"Attention Heads: {config['num_heads'] if 'num_heads' in config else config['heads']}")
        print(f"MLP Ratio: {config['mlp_ratio']}")
    
    # Calculate FLOPs
    flops = calculate_flops(model, device=device)
    if flops:
        print(f"FLOPs: {flops:,}")
        config['flops'] = flops
    
    # Load test data
    _, _, test_loader = get_cifar100_data_loaders(batch_size=args.batch_size)
    
    try:
        class_names = get_class_names()
        print(f"Loaded {len(class_names)} class names")
    except Exception as e:
        print(f"Could not load class names: {e}")
        class_names = [f"Class {i}" for i in range(100)]
    
    # Evaluate model
    results = evaluate_model(model, test_loader, device, class_names)
    
    # Save results
    save_results(results, config, args)
    
    # Visualize predictions if requested
    if args.visualize:
        print("Visualizing predictions...")
        # Get a batch of data
        inputs, targets = next(iter(test_loader))
        
        # Get predictions
        with torch.no_grad():
            outputs = model(inputs.to(device))
            _, predictions = torch.max(outputs, 1)
        
        # Create visualization
        fig = visualize_predictions(
            inputs, targets, predictions.cpu().numpy(), 
            class_names, num_samples=25
        )
        
        # Save visualization
        os.makedirs(args.results_dir, exist_ok=True)
        if config['model_type'] == 'vit':
            viz_path = os.path.join(
                args.results_dir, 
                f"vit_p{config['patch_size']}_e{config['embed_dim']}_d{config['depth']}_h{config['num_heads'] if 'num_heads' in config else config['heads']}_viz.png"
            )
        else:
            viz_path = os.path.join(args.results_dir, "resnet18_viz.png")
        
        fig.savefig(viz_path)
        print(f"Visualization saved to {viz_path}")


if __name__ == "__main__":
    main()