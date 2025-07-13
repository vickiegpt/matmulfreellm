"""
Fixed-point implementation of BitNet quantization functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .fixed_point import (
    FixedPointConfig, to_fixed_point, from_fixed_point,
    fixed_mul, fixed_div, fixed_abs, fixed_max, fixed_mean,
    fixed_clamp, fixed_reciprocal, fixed_sqrt, FixedPointSigmoid
)


def activation_quant_fixed(x: torch.Tensor) -> torch.Tensor:
    """
    Per-token quantization to 8 bits using fixed-point arithmetic.
    
    Args:
        x: Input tensor in fixed-point format (Q16.16)
        
    Returns:
        Quantized tensor in fixed-point format (Q16.16)
    """
    # Configuration
    frac_bits = FixedPointConfig.COMPUTE_FRAC_BITS
    
    # Find max absolute value per token (last dimension)
    x_abs = fixed_abs(x)
    max_vals = fixed_max(x_abs, dim=-1, keepdim=True)
    
    # Clamp minimum to avoid division by zero (1e-5 in fixed-point)
    min_val = int(1e-5 * (1 << frac_bits))  # Convert 1e-5 to fixed-point
    max_vals = torch.maximum(max_vals, torch.full_like(max_vals, min_val))
    
    # Compute scale = 127.0 / max_vals
    target_max = 127 << frac_bits  # 127 in fixed-point
    scale = fixed_div(torch.full_like(max_vals, target_max), max_vals, 
                     frac_bits, frac_bits, frac_bits)
    
    # Scale the input
    x_scaled = fixed_mul(x, scale, frac_bits, frac_bits, frac_bits)
    
    # Round and clamp to 8-bit range
    x_rounded = x_scaled  # Already integer in fixed-point
    x_clamped = fixed_clamp(x_rounded, -128 << frac_bits, 127 << frac_bits)
    
    # Rescale back
    scale_inv = fixed_reciprocal(scale, frac_bits, frac_bits)
    y = fixed_mul(x_clamped, scale_inv, frac_bits, frac_bits, frac_bits)
    
    return y


def weight_quant_fixed(w: torch.Tensor) -> torch.Tensor:
    """
    Per-tensor quantization to 1.58 bits (ternary: -1, 0, 1) using fixed-point.
    
    Args:
        w: Weight tensor in fixed-point format (Q16.16)
        
    Returns:
        Quantized weight tensor in fixed-point format (Q16.16)
    """
    # Configuration
    frac_bits = FixedPointConfig.COMPUTE_FRAC_BITS
    
    # Compute mean of absolute values
    w_abs = fixed_abs(w)
    mean_abs = fixed_mean(w_abs, frac_bits=frac_bits)
    
    # Clamp minimum to avoid division issues
    min_val = int(1e-5 * (1 << frac_bits))
    mean_abs = torch.maximum(mean_abs, torch.tensor(min_val, dtype=torch.int32, device=w.device))
    
    # Compute scale = 1.0 / mean_abs
    one = 1 << frac_bits
    scale = fixed_reciprocal(mean_abs, frac_bits, frac_bits)
    
    # Scale weights
    w_scaled = fixed_mul(w, scale.expand_as(w), frac_bits, frac_bits, frac_bits)
    
    # Quantize to exact ternary values
    threshold = one >> 1  # 0.5 in fixed-point
    w_ternary = torch.zeros_like(w_scaled)
    w_ternary = torch.where(w_scaled > threshold, one, w_ternary)
    w_ternary = torch.where(w_scaled < -threshold, -one, w_ternary)
    
    # Rescale back by multiplying with mean_abs
    u = fixed_mul(w_ternary, mean_abs.expand_as(w_ternary), 
                 frac_bits, frac_bits, frac_bits)
    
    return u


class FixedPointBitLinear(nn.Module):
    """
    Fixed-point implementation of BitLinear layer.
    Uses ternary weights and 8-bit activations with fixed-point arithmetic.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights in floating point, will be converted to fixed-point
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Fixed-point configuration
        self.frac_bits = FixedPointConfig.COMPUTE_FRAC_BITS
        
        # Pre-quantized weight storage (ternary values)
        self.register_buffer('weight_ternary', torch.zeros_like(self.weight, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.zeros(1, dtype=torch.int32))
        
    def quantize_weights(self):
        """Pre-quantize weights to ternary values and store scale"""
        # Convert weights to fixed-point
        w_fixed = to_fixed_point(self.weight, self.frac_bits)
        
        # Compute scale
        w_abs = fixed_abs(w_fixed)
        mean_abs = fixed_mean(w_abs, frac_bits=self.frac_bits)
        
        # Store scale
        self.weight_scale = mean_abs
        
        # Quantize to ternary
        w_scaled = fixed_div(w_fixed, mean_abs.expand_as(w_fixed),
                           self.frac_bits, self.frac_bits, 0)  # Output as integer
        
        # Round and clamp to {-1, 0, 1}
        w_ternary = torch.clamp(torch.round(from_fixed_point(w_scaled, 0)), -1, 1)
        self.weight_ternary = w_ternary.to(torch.int8)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using fixed-point arithmetic.
        
        Args:
            x: Input tensor (can be floating-point or fixed-point)
            
        Returns:
            Output tensor
        """
        # Convert input to fixed-point if needed
        if x.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x_fixed = to_fixed_point(x, self.frac_bits)
        else:
            x_fixed = x
            
        # Quantize activations
        x_quant = activation_quant_fixed(x_fixed)
        
        # For ternary weights, we can optimize the computation:
        # Instead of general matrix multiplication, we use:
        # - For weight = 1: add activation
        # - For weight = -1: subtract activation  
        # - For weight = 0: skip
        
        batch_size = x_quant.shape[0]
        seq_len = x_quant.shape[1] if x_quant.dim() > 2 else 1
        
        # Reshape for matrix operations
        if x_quant.dim() > 2:
            x_quant = x_quant.reshape(-1, self.in_features)
            
        # Initialize output
        output = torch.zeros(x_quant.shape[0], self.out_features, 
                           dtype=torch.int64, device=x_quant.device)
        
        # Efficient ternary matrix multiplication
        for i in range(self.out_features):
            # Get ternary weights for this output channel
            w_row = self.weight_ternary[i]
            
            # Positive weights: add corresponding inputs
            pos_mask = w_row == 1
            if pos_mask.any():
                output[:, i] += x_quant[:, pos_mask].sum(dim=1)
                
            # Negative weights: subtract corresponding inputs
            neg_mask = w_row == -1
            if neg_mask.any():
                output[:, i] -= x_quant[:, neg_mask].sum(dim=1)
                
            # Zero weights are automatically handled (no operation)
        
        # Convert back to fixed-point format and apply scale
        output = output.to(torch.int32)
        
        # Apply weight scale (reciprocal of mean absolute value)
        scale_inv = fixed_reciprocal(self.weight_scale, self.frac_bits, self.frac_bits)
        output = fixed_mul(output, scale_inv.expand_as(output[:, :1]).expand_as(output),
                         self.frac_bits, self.frac_bits, self.frac_bits)
        
        # Add bias if present
        if self.bias is not None:
            bias_fixed = to_fixed_point(self.bias, self.frac_bits)
            output = output + bias_fixed.unsqueeze(0).expand_as(output)
            
        # Reshape back if needed
        if seq_len > 1:
            output = output.reshape(batch_size, seq_len, self.out_features)
            
        # Convert back to floating point if input was floating point
        if x.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            output = from_fixed_point(output, self.frac_bits)
            
        return output


class FixedPointRMSNorm(nn.Module):
    """
    Fixed-point implementation of Root Mean Square Layer Normalization.
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        
        # Normalization weights
        self.weight = nn.Parameter(torch.ones(hidden_size))
        
        # Fixed-point configuration
        self.frac_bits = FixedPointConfig.NORM_FRAC_BITS
        self.eps_fixed = int(eps * (1 << self.frac_bits))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization using fixed-point arithmetic.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        # Convert to fixed-point if needed
        if x.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x_fixed = to_fixed_point(x, self.frac_bits)
            convert_back = True
        else:
            x_fixed = x
            convert_back = False
            
        # Compute variance: mean(x^2)
        x_squared = fixed_mul(x_fixed, x_fixed, self.frac_bits, self.frac_bits, self.frac_bits)
        variance = fixed_mean(x_squared, dim=-1, keepdim=True, frac_bits=self.frac_bits)
        
        # Add epsilon to variance
        variance = variance + self.eps_fixed
        
        # Compute reciprocal of standard deviation: 1/sqrt(variance)
        std = fixed_sqrt(variance, self.frac_bits)
        rstd = fixed_reciprocal(std, self.frac_bits, self.frac_bits)
        
        # Normalize: x * rstd
        x_norm = fixed_mul(x_fixed, rstd, self.frac_bits, self.frac_bits, self.frac_bits)
        
        # Apply learnable weights
        weight_fixed = to_fixed_point(self.weight, self.frac_bits)
        output = fixed_mul(x_norm, weight_fixed.unsqueeze(0).expand_as(x_norm),
                         self.frac_bits, self.frac_bits, self.frac_bits)
        
        # Convert back if needed
        if convert_back:
            output = from_fixed_point(output, self.frac_bits)
            
        return output