"""
DINOv3 integration module for YOLOv12 Triple Input.

This module provides DINOv3 backbone integration for enhanced feature extraction
in civil engineering applications. The DINOv3 features are used as a pre-backbone
feature extractor before the standard YOLOv12 processing pipeline.

Based on: https://github.com/facebookresearch/dinov3
HuggingFace models: facebook/dinov3-small, facebook/dinov3-base, facebook/dinov3-large
"""

import torch
import torch.nn as nn
from pathlib import Path
import warnings
from typing import Optional, Dict, Any, Tuple, List

try:
    from transformers import AutoModel, AutoImageProcessor
    from huggingface_hub import login, HfApi
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers not available. Install with: pip install transformers huggingface_hub")

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    warnings.warn("timm not available. Install with: pip install timm")


def setup_huggingface_auth():
    """
    Setup HuggingFace authentication for DINOv3 model access.
    
    Returns:
        tuple: (is_authenticated, token_source)
    """
    import os
    
    # Check for HuggingFace token in various locations
    token = None
    token_source = None
    
    # Method 1: Environment variable
    if 'HUGGINGFACE_HUB_TOKEN' in os.environ:
        token = os.environ['HUGGINGFACE_HUB_TOKEN']
        token_source = "environment variable"
    elif 'HF_TOKEN' in os.environ:
        token = os.environ['HF_TOKEN']
        token_source = "environment variable (HF_TOKEN)"
    
    # Method 2: Check for existing token file
    if not token and TRANSFORMERS_AVAILABLE:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
            if token:
                token_source = "saved token file"
        except:
            pass
    
    # Method 3: Try to login if token is provided
    if token and TRANSFORMERS_AVAILABLE:
        try:
            login(token=token, add_to_git_credential=False)
            print(f"✓ HuggingFace authentication successful (source: {token_source})")
            return True, token_source
        except Exception as e:
            print(f"⚠️ HuggingFace authentication failed: {e}")
            return False, token_source
    
    # No authentication found
    if not token:
        print("⚠️ No HuggingFace token found. DINOv3 models may not be accessible.")
        print("Please set up authentication:")
        print("  1. Get token from: https://huggingface.co/settings/tokens")
        print("  2. Set environment variable: export HUGGINGFACE_HUB_TOKEN='your_token'")
        print("  3. Or run: huggingface-cli login")
        return False, "not found"
    
    return False, "unknown"


def get_huggingface_token():
    """
    Get HuggingFace token from environment or saved location.
    
    Returns:
        str or None: The HuggingFace token if found
    """
    import os
    
    # Try environment variables first
    token = os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
    
    if not token and TRANSFORMERS_AVAILABLE:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except:
            pass
    
    return token


class DINOv3Backbone(nn.Module):
    """
    DINOv3 backbone for feature extraction before YOLOv12 processing.
    
    This module integrates DINOv3 as a frozen feature extractor that processes
    input images and outputs features compatible with YOLOv12's architecture.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dinov3-small",
        input_channels: int = 3,
        output_channels: int = 64,
        freeze: bool = True,
        use_cls_token: bool = False,
        patch_size: int = 14,
        image_size: int = 224,
        pretrained: bool = True
    ):
        """
        Initialize DINOv3 backbone.
        
        Args:
            model_name: HuggingFace model name or local path
            input_channels: Number of input channels (3 for RGB, 9 for triple input)
            output_channels: Number of output channels for YOLOv12 compatibility
            freeze: Whether to freeze DINOv3 parameters
            use_cls_token: Whether to use classification token
            patch_size: Patch size for vision transformer
            image_size: Input image size
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        self.model_name = model_name
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.freeze = freeze
        self.use_cls_token = use_cls_token
        self.patch_size = patch_size
        self.image_size = image_size
        self.pretrained = pretrained
        
        # Initialize DINOv3 model
        self.dino_model = None
        self.processor = None
        self._load_model()
        
        # Feature dimension from DINOv3
        self.feature_dim = self._get_feature_dim()
        
        # Adaptation layers
        self._build_adaptation_layers()
        
        # Handle non-RGB input channels
        if input_channels != 3:
            self._adapt_input_channels()
        
        # Freeze DINOv3 if requested
        if self.freeze:
            self._freeze_backbone()
    
    def _load_model(self):
        """Load DINOv3 model from HuggingFace or local path."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required for DINOv3. Install with: pip install transformers huggingface_hub")
        
        # Setup HuggingFace authentication
        auth_success, auth_source = setup_huggingface_auth()
        
        try:
            print(f"Loading DINOv3 model: {self.model_name}")
            
            # Get token for authenticated requests
            token = get_huggingface_token()
            
            # Load model and processor with authentication
            self.dino_model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                token=token  # Add token for authentication
            )
            
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                token=token  # Add token for authentication
            )
            
            print(f"✓ Successfully loaded DINOv3 model: {self.model_name}")
            
        except Exception as e:
            print(f"Failed to load from HuggingFace: {e}")
            if "authentication" in str(e).lower() or "token" in str(e).lower():
                print("❌ Authentication error detected. Please check your HuggingFace token:")
                print("  1. Get token from: https://huggingface.co/settings/tokens")
                print("  2. Set environment variable: export HUGGINGFACE_HUB_TOKEN='your_token'")
                print("  3. Or run: huggingface-cli login")
            print("Trying timm fallback...")
            self._load_timm_fallback()
    
    def _load_timm_fallback(self):
        """Fallback to timm implementation if HuggingFace fails."""
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for DINOv3 fallback. Install with: pip install timm")
        
        try:
            # Map HuggingFace names to timm names
            timm_name_map = {
                "facebook/dinov3-small": "vit_small_patch14_dinov3",
                "facebook/dinov3-base": "vit_base_patch14_dinov3", 
                "facebook/dinov3-large": "vit_large_patch14_dinov3"
            }
            
            timm_name = timm_name_map.get(self.model_name, "vit_small_patch14_dinov3")
            
            self.dino_model = timm.create_model(
                timm_name,
                pretrained=self.pretrained,
                num_classes=0,  # Remove classification head
                global_pool=""  # Remove global pooling
            )
            
            print(f"✓ Successfully loaded DINOv3 from timm: {timm_name}")
            
        except Exception as e:
            print(f"Failed to load from timm: {e}")
            raise RuntimeError("Could not load DINOv3 from either HuggingFace or timm")
    
    def _get_feature_dim(self) -> int:
        """Get feature dimension from DINOv3 model."""
        if hasattr(self.dino_model, 'config'):
            # HuggingFace model
            return self.dino_model.config.hidden_size
        elif hasattr(self.dino_model, 'embed_dim'):
            # timm model
            return self.dino_model.embed_dim
        else:
            # Default dimensions for different model sizes
            size_map = {
                "small": 384,
                "base": 768,
                "large": 1024
            }
            for size, dim in size_map.items():
                if size in self.model_name.lower():
                    return dim
            return 384  # Default to small
    
    def _build_adaptation_layers(self):
        """Build layers to adapt DINOv3 features for YOLOv12."""
        # Calculate number of patches
        num_patches = (self.image_size // self.patch_size) ** 2
        
        # Feature adaptation layers
        self.feature_adapter = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.output_channels * 4),
            nn.GELU(),
            nn.Linear(self.output_channels * 4, self.output_channels),
            nn.ReLU(inplace=True)
        )
        
        # Spatial reshape for compatibility with conv layers
        # Reshape from [B, N, C] to [B, C, H, W]
        self.spatial_size = int(num_patches ** 0.5)
        
        # Optional convolution for spatial processing
        self.spatial_conv = nn.Conv2d(
            self.output_channels, 
            self.output_channels, 
            kernel_size=3, 
            padding=1, 
            bias=False
        )
        self.spatial_bn = nn.BatchNorm2d(self.output_channels)
        self.spatial_act = nn.ReLU(inplace=True)
    
    def _adapt_input_channels(self):
        """Adapt DINOv3 for non-RGB input (e.g., 9-channel triple input)."""
        if self.input_channels == 3:
            return
        
        print(f"Adapting DINOv3 for {self.input_channels}-channel input")
        
        # Create input adaptation layer
        self.input_adapter = nn.Conv2d(
            self.input_channels, 
            3, 
            kernel_size=1, 
            bias=False
        )
        
        # Initialize to preserve RGB information if input contains RGB channels
        with torch.no_grad():
            if self.input_channels >= 3:
                # Initialize first 3 channels as identity
                self.input_adapter.weight[:, :3, 0, 0] = torch.eye(3)
                
                # Initialize additional channels with small random weights
                if self.input_channels > 3:
                    nn.init.normal_(self.input_adapter.weight[:, 3:, 0, 0], std=0.02)
            else:
                # For input_channels < 3, use normal initialization
                nn.init.normal_(self.input_adapter.weight, std=0.02)
    
    def _freeze_backbone(self):
        """Freeze DINOv3 backbone parameters."""
        print("Freezing DINOv3 backbone parameters")
        for param in self.dino_model.parameters():
            param.requires_grad = False
        
        # Ensure batch norm layers are in eval mode during training
        def set_bn_eval(module):
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                module.eval()
        
        self.dino_model.apply(set_bn_eval)
        self.dino_model.eval()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DINOv3 backbone.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Feature tensor [B, output_channels, H', W'] compatible with YOLOv12
        """
        B, C, H, W = x.shape
        
        # Adapt input channels if necessary
        if hasattr(self, 'input_adapter'):
            # Handle mismatch between expected channels and actual input
            if C != self.input_channels:
                if C == 3 and self.input_channels == 9:
                    # Validation/warmup with 3 channels, but model expects 9 channels
                    # Replicate the 3 channels to create 9 channels (3x RGB repetition)
                    x = x.repeat(1, 3, 1, 1)  # [B, 3, H, W] -> [B, 9, H, W]
                elif C == 9 and self.input_channels == 3:
                    # Use only first 3 channels if model expects 3 but receives 9
                    x = x[:, :3, :, :]  # [B, 9, H, W] -> [B, 3, H, W]
            
            x = self.input_adapter(x)
        
        # Resize to expected input size for DINOv3 if necessary
        if H != self.image_size or W != self.image_size:
            x = nn.functional.interpolate(
                x, 
                size=(self.image_size, self.image_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Ensure DINOv3 is in eval mode if frozen
        if self.freeze:
            self.dino_model.eval()
        
        # Extract features from DINOv3
        with torch.set_grad_enabled(not self.freeze):
            if hasattr(self.dino_model, 'forward_features'):
                # timm model
                features = self.dino_model.forward_features(x)
                if features.dim() == 3:  # [B, N, C]
                    # Remove CLS token if present
                    if not self.use_cls_token and features.size(1) > self.spatial_size ** 2:
                        features = features[:, 1:, :]  # Remove CLS token
                else:  # [B, C, H, W]
                    features = features.flatten(2).transpose(1, 2)  # [B, H*W, C]
            else:
                # HuggingFace model
                outputs = self.dino_model(x, output_hidden_states=True)
                features = outputs.last_hidden_state  # [B, N, C]
                
                # Remove CLS token if present and not wanted
                # DINOv3 typically has CLS token at position 0
                if not self.use_cls_token and features.size(1) % (14*14) != 0:
                    features = features[:, 1:, :]  # Remove CLS token
        
        # Adapt features for YOLOv12
        features = self.feature_adapter(features)  # [B, N, output_channels]
        
        # Reshape to spatial format [B, C, H, W]
        # Calculate actual spatial size from feature dimensions
        N = features.size(1)  # Number of patches
        C = features.size(2)  # Feature channels after adaptation
        
        # Handle non-perfect square by trimming to the largest perfect square
        perfect_square_size = int(N ** 0.5)
        perfect_square_patches = perfect_square_size ** 2
        
        if N != perfect_square_patches:
            # Trim to the largest perfect square
            features = features[:, :perfect_square_patches, :]
            
        features = features.transpose(1, 2).view(
            B, self.output_channels, perfect_square_size, perfect_square_size
        )
        
        # Apply spatial processing
        features = self.spatial_conv(features)
        features = self.spatial_bn(features)
        features = self.spatial_act(features)
        
        # Resize to original spatial dimensions if needed
        if H != self.image_size or W != self.image_size:
            target_h = H // 2  # Assuming we want to downsample by 2x
            target_w = W // 2
            features = nn.functional.interpolate(
                features,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            )
        
        return features
    
    def train(self, mode: bool = True):
        """Set training mode, keeping DINOv3 frozen if specified."""
        super().train(mode)
        
        if self.freeze:
            # Keep DINOv3 in eval mode
            self.dino_model.eval()
            
            # Keep batch norm layers in eval mode
            def set_bn_eval(module):
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                    module.eval()
            
            self.dino_model.apply(set_bn_eval)
        
        return self
    
    def unfreeze_backbone(self):
        """Unfreeze DINOv3 backbone for fine-tuning."""
        print("Unfreezing DINOv3 backbone parameters")
        self.freeze = False
        for param in self.dino_model.parameters():
            param.requires_grad = True
    
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get intermediate feature maps for analysis.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Dictionary of feature maps at different stages
        """
        feature_maps = {}
        
        # Input adaptation
        if hasattr(self, 'input_adapter'):
            x = self.input_adapter(x)
            feature_maps['input_adapted'] = x
        
        # DINOv3 features
        features = self.forward(x)
        feature_maps['dino_output'] = features
        
        return feature_maps


class P3FeatureEnhancer(nn.Module):
    """
    Feature enhancement module for P3 integration that uses conv operations
    instead of Vision Transformer for better compatibility with conv features.
    """
    
    def __init__(self, input_channels: int, output_channels: int, reduction_ratio: int = 4):
        """
        Initialize P3 feature enhancer.
        
        Args:
            input_channels: Number of input channels from P3 stage
            output_channels: Number of output channels for YOLOv12 compatibility
            reduction_ratio: Channel reduction ratio for bottleneck
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Channel attention mechanism
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        hidden_channels = max(input_channels // reduction_ratio, 16)
        
        self.channel_attention = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, input_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention mechanism
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        # Feature enhancement layers
        self.feature_enhance = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, padding=1, groups=input_channels),  # Depthwise
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, output_channels, 1),  # Pointwise
            nn.BatchNorm2d(output_channels)
        )
        
        # Residual connection if dimensions match
        self.use_residual = (input_channels == output_channels)
        if not self.use_residual:
            self.residual_proj = nn.Conv2d(input_channels, output_channels, 1, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention-based feature enhancement.
        
        Args:
            x: Input tensor [B, C, H, W] from P3 stage
            
        Returns:
            Enhanced feature tensor [B, output_channels, H, W]
        """
        identity = x
        
        # Channel attention
        ca_weight = self.global_pool(x)
        ca_weight = self.channel_attention(ca_weight)
        x = x * ca_weight
        
        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        sa_input = torch.cat([avg_pool, max_pool], dim=1)
        sa_weight = self.spatial_attention(sa_input)
        x = x * sa_weight
        
        # Feature enhancement
        x = self.feature_enhance(x)
        
        # Residual connection
        if self.use_residual:
            x = x + identity
        else:
            identity = self.residual_proj(identity)
            x = x + identity
        
        return x


class DINOv3TripleBackbone(DINOv3Backbone):
    """
    DINOv3 backbone specifically designed for triple input processing.
    
    This variant processes the 9-channel triple input more intelligently,
    either by using three separate DINOv3 branches or a single adapted model.
    """
    
    def __init__(self, model_name: str = "facebook/dinov3-small", input_channels: int = 9, output_channels: int = 64, freeze: bool = True, use_separate_branches: bool = False, **kwargs):
        """
        Initialize DINOv3 for triple input.
        
        Args:
            model_name: HuggingFace model name or local path
            input_channels: Number of input channels (9 for triple input)
            output_channels: Number of output channels for YOLOv12 compatibility
            freeze: Whether to freeze DINOv3 parameters
            use_separate_branches: Whether to use separate DINOv3 branches for each input
            **kwargs: Arguments passed to parent class
        """
        self.use_separate_branches = use_separate_branches
        
        if use_separate_branches:
            # Override input channels for separate branches
            input_channels = 3
        
        super().__init__(
            model_name=model_name,
            input_channels=input_channels,
            output_channels=output_channels,
            freeze=freeze,
            **kwargs
        )
        
        if use_separate_branches:
            self._build_triple_branches()
    
    def _build_triple_branches(self):
        """Build separate DINOv3 branches for triple input."""
        print("Building separate DINOv3 branches for triple input")
        
        # Create additional branches (already have one from parent)
        self.dino_branch2 = type(self.dino_model)(self.dino_model.config) if hasattr(self.dino_model, 'config') else \
                           timm.create_model(self.model_name, pretrained=self.pretrained, num_classes=0, global_pool="")
        self.dino_branch3 = type(self.dino_model)(self.dino_model.config) if hasattr(self.dino_model, 'config') else \
                           timm.create_model(self.model_name, pretrained=self.pretrained, num_classes=0, global_pool="")
        
        # Copy weights from the first branch
        if self.pretrained:
            self.dino_branch2.load_state_dict(self.dino_model.state_dict())
            self.dino_branch3.load_state_dict(self.dino_model.state_dict())
        
        # Freeze additional branches if needed
        if self.freeze:
            for param in self.dino_branch2.parameters():
                param.requires_grad = False
            for param in self.dino_branch3.parameters():
                param.requires_grad = False
            
            self.dino_branch2.eval()
            self.dino_branch3.eval()
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 3, self.feature_dim * 2),
            nn.GELU(),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for triple input.
        
        Args:
            x: Input tensor [B, 9, H, W] (triple input)
            
        Returns:
            Feature tensor [B, output_channels, H', W']
        """
        if not self.use_separate_branches:
            # Use single adapted branch
            return super().forward(x)
        
        # Split triple input
        B, C, H, W = x.shape
        assert C == 9, f"Expected 9 channels for triple input, got {C}"
        
        x1 = x[:, 0:3, :, :]  # First image
        x2 = x[:, 3:6, :, :]  # Second image
        x3 = x[:, 6:9, :, :]  # Third image
        
        # Process each branch
        features1 = self._extract_dino_features(self.dino_model, x1)
        features2 = self._extract_dino_features(self.dino_branch2, x2)
        features3 = self._extract_dino_features(self.dino_branch3, x3)
        
        # Fuse features
        features_cat = torch.cat([features1, features2, features3], dim=-1)  # [B, N, 3*C]
        features_fused = self.feature_fusion(features_cat)  # [B, N, C]
        
        # Adapt and reshape features
        features = self.feature_adapter(features_fused)
        # Calculate actual spatial size from feature dimensions
        N = features.size(1)  # Number of patches
        
        # Handle non-perfect square by trimming to the largest perfect square
        perfect_square_size = int(N ** 0.5)
        perfect_square_patches = perfect_square_size ** 2
        
        if N != perfect_square_patches:
            # Trim to the largest perfect square
            features = features[:, :perfect_square_patches, :]
            
        features = features.transpose(1, 2).view(
            B, self.output_channels, perfect_square_size, perfect_square_size
        )
        
        # Apply spatial processing
        features = self.spatial_conv(features)
        features = self.spatial_bn(features)
        features = self.spatial_act(features)
        
        return features
    
    def _extract_dino_features(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Extract features from a DINOv3 branch."""
        # Resize input if necessary
        if x.size(-1) != self.image_size or x.size(-2) != self.image_size:
            x = nn.functional.interpolate(
                x, 
                size=(self.image_size, self.image_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Extract features
        with torch.set_grad_enabled(not self.freeze):
            if hasattr(model, 'forward_features'):
                # timm model
                features = model.forward_features(x)
                if features.dim() == 3:  # [B, N, C]
                    if not self.use_cls_token and features.size(1) % (14*14) != 0:
                        features = features[:, 1:, :]  # Remove CLS token
                else:  # [B, C, H, W]
                    features = features.flatten(2).transpose(1, 2)  # [B, H*W, C]
            else:
                # HuggingFace model
                outputs = model(x, output_hidden_states=True)
                features = outputs.last_hidden_state  # [B, N, C]
                
                if not self.use_cls_token and features.size(1) % (14*14) != 0:
                    features = features[:, 1:, :]  # Remove CLS token
        
        return features


def create_dinov3_backbone(
    model_size: str = "small",
    input_channels: int = 3,
    output_channels: int = 64,
    freeze: bool = True,
    use_triple_branches: bool = False,
    **kwargs
) -> DINOv3Backbone:
    """
    Factory function to create DINOv3 backbone.
    
    Args:
        model_size: Size of DINOv3 model (small, base, large)
        input_channels: Number of input channels
        output_channels: Number of output channels
        freeze: Whether to freeze backbone
        use_triple_branches: Whether to use separate branches for triple input
        **kwargs: Additional arguments
        
    Returns:
        DINOv3Backbone instance
    """
    # Model mapping for correct HuggingFace repository names
    model_configs = {
        "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "small_plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        "base": "facebook/dinov3-vitb16-pretrain-lvd1689m", 
        "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "huge": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        "giant": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        "sat_large": "facebook/dinov3-vitl16-pretrain-lvd1689m",  # Use standard large for now
        "sat_giant": "facebook/dinov3-vit7b16-pretrain-sat493m",
    }
    
    model_name = model_configs.get(model_size, model_configs["small"])
    
    if input_channels == 9 and use_triple_branches:
        return DINOv3TripleBackbone(
            model_name=model_name,
            input_channels=input_channels,
            output_channels=output_channels,
            freeze=freeze,
            use_separate_branches=True,
            **kwargs
        )
    else:
        backbone_class = DINOv3TripleBackbone if input_channels == 9 else DINOv3Backbone
        return backbone_class(
            model_name=model_name,
            input_channels=input_channels,
            output_channels=output_channels,
            freeze=freeze,
            **kwargs
        )


class DINOv3ChannelAdapter(nn.Module):
    """
    Adapter layer to bridge between fixed DINOv3 channels and scaled YOLOv12 channels.
    
    This solves the fundamental scaling conflict by keeping DINOv3 fixed while 
    adapting its output to match YOLOv12's variant-scaled channel requirements.
    """
    
    def __init__(
        self, 
        dinov3_channels: int, 
        target_channels: int,
        use_residual: bool = True,
        activation: str = "ReLU"
    ):
        """
        Initialize channel adapter.
        
        Args:
            dinov3_channels: Fixed DINOv3 output channels
            target_channels: Target YOLOv12 channels (after variant scaling)
            use_residual: Whether to use residual connection when possible
            activation: Activation function (ReLU, GELU, SiLU)
        """
        super().__init__()
        
        self.dinov3_channels = dinov3_channels
        self.target_channels = target_channels
        self.use_residual = use_residual and (dinov3_channels == target_channels)
        
        # Build adapter layers
        if dinov3_channels == target_channels and not use_residual:
            # No adaptation needed - pass through
            self.adapter = nn.Identity()
        else:
            # Channel adaptation needed
            layers = []
            
            # Main adaptation layer
            layers.append(nn.Conv2d(dinov3_channels, target_channels, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(target_channels))
            
            # Activation function
            if activation == "ReLU":
                layers.append(nn.ReLU(inplace=True))
            elif activation == "GELU":
                layers.append(nn.GELU())
            elif activation == "SiLU":
                layers.append(nn.SiLU(inplace=True))
            
            self.adapter = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize adapter weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adapter.
        
        Args:
            x: Input tensor [B, dinov3_channels, H, W]
            
        Returns:
            Output tensor [B, target_channels, H, W]
        """
        if self.use_residual:
            # Residual connection when channels match
            return x + self.adapter(x)
        else:
            # Direct adaptation
            return self.adapter(x)
    
    def __repr__(self):
        return f"DINOv3ChannelAdapter({self.dinov3_channels} → {self.target_channels})"


class DINOv3BackboneWithAdapter(nn.Module):
    """
    DINOv3 backbone with built-in channel adapter for YOLOv12 variant scaling.
    
    This combines DINOv3Backbone with DINOv3ChannelAdapter to create a single
    module that handles both feature extraction and channel adaptation.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dinov3-small",
        input_channels: int = 3,
        dinov3_output_channels: int = 64,  # Fixed DINOv3 output
        target_channels: int = 64,  # Target YOLOv12 channels (will be scaled)
        freeze: bool = True,
        **kwargs
    ):
        """
        Initialize DINOv3 with adapter.
        
        Args:
            model_name: HuggingFace model name
            input_channels: Number of input channels
            dinov3_output_channels: Fixed DINOv3 output channels (not scaled)
            target_channels: Target YOLOv12 channels (will be scaled by variant)
            freeze: Whether to freeze DINOv3
            **kwargs: Additional arguments for DINOv3Backbone
        """
        super().__init__()
        
        # Fixed DINOv3 backbone (never scaled)
        self.dinov3 = DINOv3Backbone(
            model_name=model_name,
            input_channels=input_channels,
            output_channels=dinov3_output_channels,  # Fixed output
            freeze=freeze,
            **kwargs
        )
        
        # Channel adapter (handles scaling)
        self.adapter = DINOv3ChannelAdapter(
            dinov3_channels=dinov3_output_channels,
            target_channels=target_channels
        )
        
        self.dinov3_output_channels = dinov3_output_channels
        self.target_channels = target_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DINOv3 and adapter.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Adapted features [B, target_channels, H', W']
        """
        # Extract fixed DINOv3 features
        dinov3_features = self.dinov3(x)
        
        # Adapt channels to YOLOv12 requirements
        adapted_features = self.adapter(dinov3_features)
        
        return adapted_features
    
    def update_target_channels(self, new_target_channels: int):
        """
        Update target channels for different YOLOv12 variants.
        
        Args:
            new_target_channels: New target channel count
        """
        if new_target_channels != self.target_channels:
            print(f"Updating adapter: {self.target_channels} → {new_target_channels} channels")
            
            # Create new adapter
            self.adapter = DINOv3ChannelAdapter(
                dinov3_channels=self.dinov3_output_channels,
                target_channels=new_target_channels
            )
            self.target_channels = new_target_channels
    
    def train(self, mode: bool = True):
        """Set training mode, keeping DINOv3 frozen if specified."""
        super().train(mode)
        self.dinov3.train(mode)
        return self
    
    def __repr__(self):
        return f"DINOv3BackboneWithAdapter(DINOv3: {self.dinov3_output_channels}, Target: {self.target_channels})"