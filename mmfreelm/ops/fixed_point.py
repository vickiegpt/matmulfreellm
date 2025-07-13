"""
Fixed-point arithmetic utilities for MatMulFreeLLM
Implements fixed-point operations optimized for quantized neural networks
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class FixedPointConfig:
    """Configuration for fixed-point formats used in different parts of the model"""
    
    # Q8.8 format for activations (16-bit total)
    ACT_INT_BITS = 8
    ACT_FRAC_BITS = 8
    
    # Q8.24 format for weight scales (32-bit total)
    SCALE_INT_BITS = 8
    SCALE_FRAC_BITS = 24
    
    # Q16.16 format for intermediate computations (32-bit total)
    COMPUTE_INT_BITS = 16
    COMPUTE_FRAC_BITS = 16
    
    # Q24.8 format for normalization (32-bit total)
    NORM_INT_BITS = 24
    NORM_FRAC_BITS = 8
    
    # Q16.48 format for accumulators (64-bit total)
    ACC_INT_BITS = 16
    ACC_FRAC_BITS = 48


def to_fixed_point(x: torch.Tensor, frac_bits: int) -> torch.Tensor:
    """Convert floating-point tensor to fixed-point representation"""
    scale = 2 ** frac_bits
    return (x * scale).round().to(torch.int32)


def from_fixed_point(x: torch.Tensor, frac_bits: int) -> torch.Tensor:
    """Convert fixed-point tensor back to floating-point"""
    scale = 2 ** frac_bits
    return x.float() / scale


def fixed_mul(a: torch.Tensor, b: torch.Tensor, 
              a_frac: int, b_frac: int, out_frac: int) -> torch.Tensor:
    """Fixed-point multiplication with format conversion"""
    # Multiply as int64 to prevent overflow
    result = a.to(torch.int64) * b.to(torch.int64)
    
    # Adjust fractional bits
    shift = a_frac + b_frac - out_frac
    if shift > 0:
        result = result >> shift
    elif shift < 0:
        result = result << (-shift)
    
    return result.to(torch.int32)


def fixed_div(a: torch.Tensor, b: torch.Tensor,
              a_frac: int, b_frac: int, out_frac: int) -> torch.Tensor:
    """Fixed-point division with format conversion"""
    # Prevent division by zero
    b_safe = torch.where(b == 0, torch.ones_like(b), b)
    
    # Scale numerator to maintain precision
    shift = out_frac - a_frac + b_frac
    if shift > 0:
        a_scaled = a.to(torch.int64) << shift
    else:
        a_scaled = a.to(torch.int64) >> (-shift)
    
    result = a_scaled // b_safe.to(torch.int64)
    
    # Handle division by zero case
    result = torch.where(b == 0, torch.zeros_like(result), result)
    
    return result.to(torch.int32)


def fixed_sqrt(x: torch.Tensor, frac_bits: int, iterations: int = 5) -> torch.Tensor:
    """Fixed-point square root using Newton-Raphson method"""
    # Initial guess: x >> 1
    guess = x >> 1
    
    # Avoid sqrt(0)
    mask = x > 0
    x_safe = torch.where(mask, x, torch.ones_like(x))
    
    for _ in range(iterations):
        # Newton-Raphson: guess = (guess + x/guess) / 2
        x_div_guess = fixed_div(x_safe, guess, frac_bits, frac_bits, frac_bits)
        guess = (guess + x_div_guess) >> 1
    
    return torch.where(mask, guess, torch.zeros_like(guess))


def fixed_reciprocal(x: torch.Tensor, in_frac: int, out_frac: int) -> torch.Tensor:
    """Compute 1/x in fixed-point using Newton-Raphson"""
    # Prevent division by zero
    x_safe = torch.where(x == 0, torch.ones_like(x), x)
    
    # Initial guess using lookup table for common values
    # For now, use simple approximation
    one = 1 << in_frac
    
    # Newton-Raphson for 1/x: guess = guess * (2 - x * guess)
    guess = one // x_safe  # Initial approximation
    
    for _ in range(3):
        # x * guess
        x_guess = fixed_mul(x_safe, guess, in_frac, out_frac, out_frac)
        # 2 - x * guess
        two = 2 << out_frac
        error = two - x_guess
        # guess * error
        guess = fixed_mul(guess, error, out_frac, out_frac, out_frac)
    
    return torch.where(x == 0, torch.zeros_like(guess), guess)


class FixedPointSigmoid:
    """Fixed-point sigmoid using piecewise polynomial approximation"""
    
    def __init__(self, frac_bits: int = 16):
        self.frac_bits = frac_bits
        self.scale = 1 << frac_bits
        
        # Precomputed values for regions
        self.neg_sat = -4 << frac_bits  # x < -4: sigmoid ≈ 0
        self.pos_sat = 4 << frac_bits   # x > 4: sigmoid ≈ 1
        self.one = 1 << frac_bits
        self.half = 1 << (frac_bits - 1)
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sigmoid(x) in fixed-point"""
        # Simple piecewise linear approximation for better accuracy:
        # x <= -2.5: 0
        # -2.5 < x < 2.5: 0.2 * x + 0.5
        # x >= 2.5: 1
        
        # Thresholds in fixed-point
        neg_thresh = int(-2.5 * self.scale)
        pos_thresh = int(2.5 * self.scale)
        
        # Slope: 0.2 in fixed-point
        slope = self.scale // 5
        
        # Create output
        result = torch.zeros_like(x)
        
        # Saturated regions
        result = torch.where(x <= neg_thresh, torch.zeros_like(x), result)
        result = torch.where(x >= pos_thresh, self.one * torch.ones_like(x), result)
        
        # Linear region: 0.2 * x + 0.5
        mask = (x > neg_thresh) & (x < pos_thresh)
        if mask.any():
            x_masked = x[mask]
            # result = 0.2 * x + 0.5
            linear_term = fixed_mul(slope * torch.ones_like(x_masked), x_masked,
                                  self.frac_bits, self.frac_bits, self.frac_bits)
            result[mask] = linear_term + self.half
        
        return result


class FixedPointReLU:
    """Fixed-point ReLU activation"""
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.maximum(x, torch.zeros_like(x))


class FixedPointTanh:
    """Fixed-point tanh using rational approximation"""
    
    def __init__(self, frac_bits: int = 16):
        self.frac_bits = frac_bits
        self.scale = 1 << frac_bits
        self.sigmoid = FixedPointSigmoid(frac_bits)
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # tanh(x) = 2 * sigmoid(2*x) - 1
        x2 = x << 1  # 2*x
        sig_2x = self.sigmoid(x2)
        two_sig = sig_2x << 1  # 2 * sigmoid(2*x)
        return two_sig - self.scale  # 2 * sigmoid(2*x) - 1


def fixed_clamp(x: torch.Tensor, min_val: int, max_val: int) -> torch.Tensor:
    """Clamp values between min and max"""
    return torch.clamp(x, min_val, max_val)


def fixed_round(x: torch.Tensor) -> torch.Tensor:
    """Round to nearest integer (already integers in fixed-point)"""
    return x


def fixed_abs(x: torch.Tensor) -> torch.Tensor:
    """Absolute value in fixed-point"""
    return torch.abs(x)


def fixed_max(x: torch.Tensor, dim: Optional[int] = None, 
              keepdim: bool = False) -> torch.Tensor:
    """Maximum value along dimension"""
    if dim is None:
        return torch.max(x)
    else:
        return torch.max(x, dim=dim, keepdim=keepdim).values


def fixed_mean(x: torch.Tensor, dim: Optional[int] = None, 
               keepdim: bool = False, frac_bits: int = 16) -> torch.Tensor:
    """Mean value in fixed-point"""
    if dim is None:
        n = x.numel()
        sum_val = torch.sum(x.to(torch.int64)).to(torch.int32)
    else:
        n = x.shape[dim]
        sum_val = torch.sum(x.to(torch.int64), dim=dim, keepdim=keepdim).to(torch.int32)
    
    # Divide by n - use fixed number directly without shifting
    n_tensor = torch.full_like(sum_val if isinstance(sum_val, torch.Tensor) else x[..., 0], n, dtype=torch.int32)
    return fixed_div(sum_val, n_tensor, frac_bits, 0, frac_bits)


def fixed_sum(x: torch.Tensor, dim: Optional[int] = None, 
              keepdim: bool = False) -> torch.Tensor:
    """Sum values (accumulator should have more bits)"""
    return torch.sum(x.to(torch.int64), dim=dim, keepdim=keepdim).to(torch.int32)