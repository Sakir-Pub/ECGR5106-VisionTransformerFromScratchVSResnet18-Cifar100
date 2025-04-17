import torch
import torch.nn as nn
from transformers import SwinForImageClassification, SwinConfig


class SwinTransformerModel:
    """
    A wrapper class for Swin Transformer models, supporting both fine-tuning of pretrained
    models and training from scratch.
    """
    
    def __init__(self, model_type, num_classes=100, pretrained=True, freeze_backbone=True):
        """
        Initialize a Swin Transformer model.
        
        Args:
            model_type (str): Type of Swin model ('tiny' or 'small')
            num_classes (int): Number of output classes (100 for CIFAR-100)
            pretrained (bool): Whether to load pretrained weights
            freeze_backbone (bool): Whether to freeze the backbone layers
        """
        self.model_type = model_type
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        
        # Model initialization
        self.model = self._initialize_model()
        
        # Apply freezing if needed
        if pretrained and freeze_backbone:
            self._freeze_backbone()
            
    def _initialize_model(self):
        """
        Initialize the appropriate Swin Transformer model.
        
        Returns:
            SwinForImageClassification: The initialized model
        """
        if self.pretrained:
            # Load pretrained models from Hugging Face
            if self.model_type == 'tiny':
                model = SwinForImageClassification.from_pretrained(
                    "microsoft/swin-tiny-patch4-window7-224",
                    num_labels=self.num_classes,
                    ignore_mismatched_sizes=True  # Important for resizing head
                )
            elif self.model_type == 'small':
                model = SwinForImageClassification.from_pretrained(
                    "microsoft/swin-small-patch4-window7-224",
                    num_labels=self.num_classes,
                    ignore_mismatched_sizes=True  # Important for resizing head
                )
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        else:
            # Create model configuration based on model type
            if self.model_type == 'tiny':
                config = SwinConfig(
                    image_size=224,
                    patch_size=4,
                    num_channels=3,
                    embed_dim=96,
                    depths=[2, 2, 6, 2],
                    num_heads=[3, 6, 12, 24],
                    window_size=7,
                    num_labels=self.num_classes
                )
            elif self.model_type == 'small':
                config = SwinConfig(
                    image_size=224,
                    patch_size=4,
                    num_channels=3,
                    embed_dim=96,
                    depths=[2, 2, 18, 2],
                    num_heads=[3, 6, 12, 24],
                    window_size=7,
                    num_labels=self.num_classes
                )
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
            # Initialize model from scratch with the configuration
            model = SwinForImageClassification(config)
            
        return model
    
    def _freeze_backbone(self):
        """
        Freeze all parameters of the backbone layers, leaving only the classifier trainable.
        """
        # Freeze all parameters in the swin backbone
        for param in self.model.swin.parameters():
            param.requires_grad = False
            
        # Ensure classifier parameters are trainable
        for param in self.model.classifier.parameters():
            param.requires_grad = True
            
    def get_model(self):
        """
        Get the PyTorch model.
        
        Returns:
            nn.Module: The model
        """
        return self.model
    
    def get_total_parameters(self):
        """
        Count the total number of parameters in the model.
        
        Returns:
            int: Number of parameters
        """
        return sum(p.numel() for p in self.model.parameters())
    
    def get_trainable_parameters(self):
        """
        Count the number of trainable parameters in the model.
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_model_info(self):
        """
        Get a string with model information.
        
        Returns:
            str: Model information
        """
        model_name = f"Swin-{self.model_type.capitalize()}"
        if self.pretrained:
            model_name += " (pretrained"
            if self.freeze_backbone:
                model_name += ", backbone frozen)"
            else:
                model_name += ", fully fine-tuned)"
        else:
            model_name += " (trained from scratch)"
            
        total_params = self.get_total_parameters()
        trainable_params = self.get_trainable_parameters()
        
        info = (
            f"Model: {model_name}\n"
            f"Total parameters: {total_params:,}\n"
            f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})\n"
        )
        
        return info


# Factory function to create a model by name
def create_model(model_name, num_classes=100):
    """
    Create a model by name.
    
    Args:
        model_name (str): Name of the model to create
        num_classes (int): Number of output classes
        
    Returns:
        SwinTransformerModel: The created model wrapper
    """
    if model_name == 'swin_tiny_pretrained':
        return SwinTransformerModel('tiny', num_classes=num_classes, pretrained=True, freeze_backbone=True)
    elif model_name == 'swin_small_pretrained':
        return SwinTransformerModel('small', num_classes=num_classes, pretrained=True, freeze_backbone=True)
    elif model_name == 'swin_tiny_scratch':
        return SwinTransformerModel('tiny', num_classes=num_classes, pretrained=False, freeze_backbone=False)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


# Example usage
if __name__ == "__main__":
    # Create models and print their information
    for model_name in ['swin_tiny_pretrained', 'swin_small_pretrained', 'swin_tiny_scratch']:
        model_wrapper = create_model(model_name)
        print(model_wrapper.get_model_info())
        print("-" * 50)