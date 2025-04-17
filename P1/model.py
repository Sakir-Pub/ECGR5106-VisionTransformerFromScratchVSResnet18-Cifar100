import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models import resnet18


class PatchEmbedding(nn.Module):
    """
    Convert input images to patch embeddings.
    
    Args:
        img_size (int): Size of the input image (assumed to be square)
        patch_size (int): Size of the patches (assumed to be square)
        in_channels (int): Number of input channels
        embed_dim (int): Embedding dimension
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Linear projection of flattened patches
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # Create positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)  # +1 for cls token
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Initialize parameters
        self._init_weights()
        
    def _init_weights(self):
        # Initialize proj with a truncated normal distribution
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_layer_weights)
        
    def _init_layer_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        # x: (B, C, H, W) -> (B, E, H//P, W//P) where P is patch_size and E is embed_dim
        B, C, H, W = x.shape
        assert H == W == self.img_size, f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})"
        
        # Create patch embeddings - (B, E, H', W') -> (B, H'*W', E)
        x = self.proj(x).flatten(2).transpose(1, 2)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention module.
    
    Args:
        dim (int): Input dimension
        num_heads (int): Number of attention heads
        qkv_bias (bool): Whether to add bias to the qkv projections
        attn_drop (float): Dropout rate for attention matrix
        proj_drop (float): Dropout rate for projection
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Calculate query, key, value for all heads in batch
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Calculate attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """
    MLP module with GELU activation.
    
    Args:
        in_features (int): Number of input features
        hidden_features (int): Number of hidden features
        out_features (int): Number of output features
        drop (float): Dropout rate
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block.
    
    Args:
        dim (int): Input dimension
        num_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        qkv_bias (bool): Whether to add bias to the qkv projections
        drop (float): Dropout rate
        attn_drop (float): Dropout rate for attention matrix
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        
        # Multi-head attention
        self.attn = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        
        # Layer normalization
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        
        # MLP block
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop
        )
        
    def forward(self, x):
        # First attention block with residual connection
        x = x + self.attn(self.norm1(x))
        
        # MLP block with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model.
    
    Args:
        img_size (int): Input image size
        patch_size (int): Patch size
        in_channels (int): Number of input channels
        num_classes (int): Number of classes
        embed_dim (int): Embedding dimension
        depth (int): Number of transformer layers
        num_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        qkv_bias (bool): Whether to add bias to the qkv projections
        drop_rate (float): Dropout rate
        attn_drop_rate (float): Dropout rate for attention matrix
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=100,
                 embed_dim=256, depth=4, num_heads=4, mlp_ratio=2., qkv_bias=True,
                 drop_rate=0.1, attn_drop_rate=0.):
        super().__init__()
        
        # Store configuration
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.patch_size = patch_size
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Dropout after embedding
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        # Create patch embeddings
        x = self.patch_embed(x)
        
        # Apply dropout
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply layer normalization
        x = self.norm(x)
        
        # Extract cls token for classification
        x = x[:, 0]
        
        # Classification
        x = self.head(x)
        
        return x

    def get_config(self):
        """Return model configuration as a dictionary."""
        return {
            'model_type': 'ViT',
            'embed_dim': self.embed_dim,
            'depth': self.depth,
            'patch_size': self.patch_size,
            'num_classes': self.num_classes
        }


class ResNet18ForCIFAR(nn.Module):
    """
    Modified ResNet-18 for CIFAR-100.
    
    Args:
        num_classes (int): Number of classes
        pretrained (bool): Whether to use pretrained weights
    """
    def __init__(self, num_classes=100, pretrained=False):
        super().__init__()
        
        # Load base ResNet-18 model
        self.model = resnet18(pretrained=pretrained)
        
        # Modify first convolutional layer for 32x32 input
        # Original ResNet-18 has kernel=7, stride=2 which is too large for CIFAR
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Remove max pooling layer after conv1
        self.model.maxpool = nn.Identity()
        
        # Modify final fully connected layer for CIFAR-100 classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def get_config(self):
        """Return model configuration as a dictionary."""
        return {
            'model_type': 'ResNet18',
            'num_classes': 100
        }


def create_model(model_type='vit', **kwargs):
    """
    Create a model based on the specified type and configuration.
    
    Args:
        model_type (str): Model type ('vit' or 'resnet18')
        **kwargs: Additional configuration parameters
        
    Returns:
        nn.Module: Model instance
    """
    if model_type.lower() == 'vit':
        return VisionTransformer(**kwargs)
    elif model_type.lower() == 'resnet18':
        return ResNet18ForCIFAR(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    """
    Count the number of trainable parameters in the model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_models():
    """Test all model configurations and print their parameter counts."""
    # Test ViT configurations
    configs = [
        {'patch_size': 4, 'embed_dim': 256, 'depth': 4, 'num_heads': 2, 'mlp_ratio': 2.0},
        {'patch_size': 4, 'embed_dim': 256, 'depth': 8, 'num_heads': 4, 'mlp_ratio': 2.0},
        {'patch_size': 8, 'embed_dim': 512, 'depth': 4, 'num_heads': 2, 'mlp_ratio': 4.0},
        {'patch_size': 8, 'embed_dim': 512, 'depth': 8, 'num_heads': 4, 'mlp_ratio': 4.0},
    ]
    
    # Print header
    print(f"{'Model':<30} {'Parameters':<15} {'Config'}")
    print("-" * 80)
    
    # Test each ViT configuration
    for i, config in enumerate(configs):
        model = create_model('vit', **config)
        params = count_parameters(model)
        config_str = f"p={config['patch_size']}, e={config['embed_dim']}, d={config['depth']}, h={config['num_heads']}, mlp={config['mlp_ratio']}"
        print(f"ViT Config {i+1:<3} {params:<15,} {config_str}")
    
    # Test ResNet-18
    resnet = create_model('resnet18')
    params = count_parameters(resnet)
    print(f"ResNet-18 {params:<15,}")


# Add truncated normal initialization function for compatibility
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    
    if not hasattr(nn, 'trunc_normal_'):
        with torch.no_grad():
            # Values are generated from the total of the truncated distribution of
            # mean=mean and std=std and then scaled into [low, high]/[a, b]
            # get bounds
            l = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)
            
            # reject/accept method
            tensor.uniform_(2 * l - 1, 2 * u - 1)
            tensor.erfinv_()
            
            # transform to proper mean, std
            tensor.mul_(std * math.sqrt(2.))
            tensor.add_(mean)
            
            # clamp to get [low, high]/[a, b]
            tensor.clamp_(min=a, max=b)
            return tensor
    else:
        return nn.init.trunc_normal_(tensor, mean, std, a, b)

# Add to nn.init if it doesn't exist
if not hasattr(nn.init, 'trunc_normal_'):
    nn.init.trunc_normal_ = trunc_normal_


if __name__ == "__main__":
    # Test the models
    test_models()
    
    # Create a small input and test forward pass
    x = torch.randn(2, 3, 32, 32)
    
    # Test ViT with 4x4 patches
    vit = create_model('vit', patch_size=4, embed_dim=256, depth=4, num_heads=2)
    output = vit(x)
    print(f"ViT output shape: {output.shape}")
    
    # Test ResNet-18
    resnet = create_model('resnet18')
    output = resnet(x)
    print(f"ResNet-18 output shape: {output.shape}")