import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os

class CIFAR100DataLoader:
    """
    DataLoader class for loading and preprocessing the CIFAR-100 dataset
    for Swin Transformer models.
    """
    
    def __init__(self, data_dir='./data', batch_size=32, num_workers=4, 
                 val_split=0.1, seed=42):
        """
        Initialize the CIFAR-100 data loader.
        
        Args:
            data_dir (str): Directory where dataset will be stored
            batch_size (int): Batch size for training and testing
            num_workers (int): Number of workers for data loading
            val_split (float): Fraction of training data to use for validation
            seed (int): Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed
        
        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        
        # Define transformations
        self.train_transform = self._get_train_transforms()
        self.test_transform = self._get_test_transforms()
        
    def _get_train_transforms(self):
        """
        Get transformations for training data.
        Includes resizing to 224x224 (Swin input size) and data augmentation.
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to Swin input dimensions
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                                 std=[0.2675, 0.2565, 0.2761])  # CIFAR-100 normalization
        ])
    
    def _get_test_transforms(self):
        """
        Get transformations for test/validation data.
        Only includes resizing and normalization.
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to Swin input dimensions
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                                 std=[0.2675, 0.2565, 0.2761])  # CIFAR-100 normalization
        ])
    
    def get_data_loaders(self):
        """
        Create and return train, validation and test data loaders.
        
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        # Load the training dataset
        train_dataset = datasets.CIFAR100(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.train_transform
        )
        
        # Split into training and validation sets
        val_size = int(len(train_dataset) * self.val_split)
        train_size = len(train_dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            train_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed)
        )
        
        # For validation, we want to use test transforms (no augmentation)
        val_dataset.dataset.transform = self.test_transform
        
        # Load the test dataset
        test_dataset = datasets.CIFAR100(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.test_transform
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    @staticmethod
    def get_cifar100_classes():
        """
        Get the list of class names for CIFAR-100.
        
        Returns:
            list: List of class names
        """
        return [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 
            'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 
            'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 
            'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 
            'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
        ]


# Simple usage example
if __name__ == "__main__":
    data_loader = CIFAR100DataLoader()
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()
    
    # Print some statistics
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of testing batches: {len(test_loader)}")
    
    # Get a batch and print its shape
    images, labels = next(iter(train_loader))
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    
    # Get class names
    class_names = data_loader.get_cifar100_classes()
    print(f"Total number of classes: {len(class_names)}")
    print(f"Sample classes: {class_names[:5]}")