import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def get_cifar100_data_loaders(batch_size=64, num_workers=4, data_dir='./data', 
                              val_split=0.1, augment=True, download=True):
    """
    Create data loaders for CIFAR-100 dataset with optional data augmentation.
    
    Args:
        batch_size (int): Batch size for the data loaders
        num_workers (int): Number of workers for the data loaders
        data_dir (str): Directory to store the dataset
        val_split (float): Proportion of training data to use for validation
        augment (bool): Whether to use data augmentation for training
        download (bool): Whether to download the dataset if not available
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Define the normalization parameters for CIFAR-100
    # These are the mean and std for each channel in the dataset
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    
    # Define transformations for training data
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    # Define transformations for test data (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Load the CIFAR-100 training dataset
    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        transform=train_transform,
        download=download
    )
    
    # Split training data into training and validation sets
    if val_split > 0:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Apply validation transform to validation dataset
        val_dataset.dataset.transform = test_transform
    else:
        val_dataset = None
    
    # Load the CIFAR-100 test dataset
    test_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        transform=test_transform,
        download=download
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        val_loader = None
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def get_class_names():
    """
    Get the class names for CIFAR-100.
    
    Returns:
        list: List of class names
    """
    # Create a temporary dataset to get the class names
    temp_dataset = datasets.CIFAR100(root='./data', train=False, download=True)
    return temp_dataset.classes if hasattr(temp_dataset, 'classes') else [f"Class {i}" for i in range(100)]

def visualize_samples(data_loader, num_samples=5, classes=None, cols=5):
    """
    Visualize random samples from a data loader.
    
    Args:
        data_loader: PyTorch DataLoader
        num_samples (int): Number of samples to visualize
        classes (list): List of class names, if None get from CIFAR-100
        cols (int): Number of columns in the plot
    """
    if classes is None:
        classes = get_class_names()
    
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    
    # Denormalize images
    mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
    std = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)
    images = images * std + mean
    
    # Convert to numpy and transpose from (C, H, W) to (H, W, C)
    images = images.numpy().transpose((0, 2, 3, 1))
    images = np.clip(images, 0, 1)
    
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()
    
    for i in range(num_samples):
        axes[i].imshow(images[i])
        axes[i].set_title(classes[labels[i]])
        axes[i].axis('off')
    
    # Hide any unused axes
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def get_dataset_stats():
    """
    Return basic statistics about the CIFAR-100 dataset.
    
    Returns:
        dict: Dictionary containing dataset statistics
    """
    return {
        'name': 'CIFAR-100',
        'num_classes': 100,
        'image_size': (32, 32),
        'channels': 3,
        'train_samples': 50000,
        'test_samples': 10000,
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761)
    }

if __name__ == "__main__":
    # Example usage
    train_loader, val_loader, test_loader = get_cifar100_data_loaders()
    
    # Print basic dataset information
    stats = get_dataset_stats()
    print(f"Dataset: {stats['name']}")
    print(f"Number of classes: {stats['num_classes']}")
    print(f"Image size: {stats['image_size']}")
    print(f"Training samples: {stats['train_samples']}")
    print(f"Test samples: {stats['test_samples']}")
    
    # Calculate size of data loaders
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Get class names
    try:
        class_names = get_class_names()
        print(f"First 10 classes: {class_names[:10]}")
        
        # Visualize training samples
        print("Visualizing training samples:")
        visualize_samples(train_loader, num_samples=10, classes=class_names)
    except Exception as e:
        print(f"Error displaying class names or samples: {e}")
        print("Continuing without visualization...")