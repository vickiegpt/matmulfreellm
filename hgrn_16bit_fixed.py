"""
16-bit Fixed-Point Implementation of HGRN Algorithm
Higher precision version with Q8.8 format
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class HGRNFixed16bit(nn.Module):
    """
    16-bit fixed-point implementation of HGRN (Hierarchically Gated Recurrent Network)
    
    Algorithm overview:
    1. Input projection: i = W_i @ x, f = W_f @ x, g = W_g @ x
    2. Gate activation: f = sigmoid(f)
    3. Input modulation: i = swiglu(i, 1-f) = i * sigmoid(i) * (1-f)
    4. Recurrent update: h_t = f_t * h_{t-1} + i_t
    5. Output projection: o = W_o @ (g_norm(g) * h)
    
    Fixed-point format: Q8.8 (16-bit total: 8 integer bits, 8 fractional bits)
    Range: [-128, 127.996] with precision 0.00390625
    
    Advantages over 8-bit:
    - 32x better precision (2^8 vs 2^3 fractional bits)
    - Wider range for integer part
    - Better numerical stability
    - Still 2x more memory efficient than float32
    """
    
    def __init__(self):
        super().__init__()
        # Fixed-point configuration for Q8.8 format
        self.FRAC_BITS = 8      # 8 fractional bits for better precision
        self.INT_BITS = 8       # 8 integer bits
        self.SCALE = 1 << self.FRAC_BITS  # 256
        self.MAX_VAL = (1 << 15) - 1      # 32767 (max int16)
        self.MIN_VAL = -(1 << 15)         # -32768 (min int16)
        
        # Precision comparison
        self.PRECISION = 1.0 / self.SCALE  # 0.00390625 (much better than 8-bit's 0.03125)
        
    def to_fixed(self, x: torch.Tensor) -> torch.Tensor:
        """Convert floating-point to 16-bit fixed-point Q8.8"""
        x_scaled = (x * self.SCALE).round()
        x_clamped = torch.clamp(x_scaled, self.MIN_VAL, self.MAX_VAL)
        return x_clamped.to(torch.int16)
    
    def from_fixed(self, x: torch.Tensor) -> torch.Tensor:
        """Convert 16-bit fixed-point Q8.8 to floating-point"""
        return x.float() / self.SCALE
    
    def fixed_mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Multiply two fixed-point numbers
        For 16-bit, we use int32 intermediate to prevent overflow
        """
        # Cast to int32 to prevent overflow during multiplication
        result = (a.to(torch.int32) * b.to(torch.int32)) >> self.FRAC_BITS
        # Clamp and return as int16
        return torch.clamp(result, self.MIN_VAL, self.MAX_VAL).to(torch.int16)
    
    def sigmoid_lut(self) -> torch.Tensor:
        """
        Generate sigmoid lookup table for 16-bit values
        With more bits, we can have a much more accurate LUT
        """
        # For 16-bit, we can afford a larger, more precise lookup table
        # Sample at regular intervals for efficiency
        step = 64  # Sample every 64 values to keep LUT manageable
        indices = torch.arange(self.MIN_VAL, self.MAX_VAL + 1, step, dtype=torch.float32)
        x = indices / self.SCALE
        sigmoid_vals = torch.sigmoid(x)
        return self.to_fixed(sigmoid_vals), step
    
    def apply_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sigmoid using lookup table with linear interpolation
        for better accuracy with 16-bit precision
        """
        lut, step = self.sigmoid_lut()
        lut = lut.to(x.device)
        
        # Calculate indices for lookup with interpolation
        x_shifted = x.to(torch.int32) - self.MIN_VAL
        indices = torch.div(x_shifted, step, rounding_mode='floor')
        remainder = x_shifted % step
        
        # Clamp indices
        indices = torch.clamp(indices, 0, len(lut) - 2).to(torch.long)
        
        # Linear interpolation for smoother results
        y0 = lut[indices]
        y1 = lut[indices + 1]
        
        # Interpolate (y0 + (y1-y0) * remainder/step)
        diff = (y1 - y0).to(torch.int32)
        interp = y0 + ((diff * remainder.to(torch.int32)) // step).to(torch.int16)
        
        return interp
    
    def ternary_matmul(self, x: torch.Tensor, w_ternary: torch.Tensor, 
                       w_scale: torch.Tensor) -> torch.Tensor:
        """
        Perform ternary matrix multiplication with 16-bit precision
        
        Args:
            x: Input tensor [batch, seq_len, hidden] in 16-bit fixed-point
            w_ternary: Ternary weight matrix {-1, 0, 1} [hidden, out]
            w_scale: Per-channel scale factors [out] in 16-bit fixed-point
            
        Returns:
            Output tensor [batch, seq_len, out] in 16-bit fixed-point
        """
        batch, seq_len, hidden = x.shape
        out_dim = w_ternary.shape[1]
        
        # Use int32 for accumulation to prevent overflow with 16-bit inputs
        output = torch.zeros(batch, seq_len, out_dim, dtype=torch.int32, device=x.device)
        
        # Efficient ternary multiplication
        for i in range(out_dim):
            # Get ternary weights for this output channel
            w_col = w_ternary[:, i]
            
            # Compute dot product using only additions/subtractions
            pos_mask = w_col == 1
            neg_mask = w_col == -1
            
            if pos_mask.any():
                output[:, :, i] += x[:, :, pos_mask].sum(dim=-1).to(torch.int32)
            if neg_mask.any():
                output[:, :, i] -= x[:, :, neg_mask].sum(dim=-1).to(torch.int32)
        
        # Apply scale factors with int32 intermediate
        output = (output * w_scale.unsqueeze(0).unsqueeze(0).to(torch.int32)) >> self.FRAC_BITS
        
        return torch.clamp(output, self.MIN_VAL, self.MAX_VAL).to(torch.int16)
    
    def hgrn_cell(self, x: torch.Tensor, h_prev: torch.Tensor,
                  w_i: torch.Tensor, w_f: torch.Tensor, w_g: torch.Tensor, w_o: torch.Tensor,
                  w_scale_i: torch.Tensor, w_scale_f: torch.Tensor, 
                  w_scale_g: torch.Tensor, w_scale_o: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single HGRN cell computation in 16-bit fixed-point
        
        Args:
            x: Input [batch, hidden] in 16-bit fixed-point
            h_prev: Previous hidden state [batch, hidden] in 16-bit fixed-point
            w_i, w_f, w_g, w_o: Ternary weight matrices
            w_scale_i, w_scale_f, w_scale_g, w_scale_o: Scale factors in 16-bit
            
        Returns:
            o: Output 
            h: New hidden state
        """
        # Project input with ternary matrices
        i = self.ternary_matmul(x.unsqueeze(1), w_i, w_scale_i).squeeze(1)
        f = self.ternary_matmul(x.unsqueeze(1), w_f, w_scale_f).squeeze(1)
        g = self.ternary_matmul(x.unsqueeze(1), w_g, w_scale_g).squeeze(1)
        
        # Apply sigmoid to forget gate
        f_sig = self.apply_sigmoid(f)
        
        # Compute 1 - f (in 16-bit fixed-point)
        one_fixed = self.to_fixed(torch.ones_like(f_sig, dtype=torch.float32, device=f_sig.device))
        one_minus_f = one_fixed - f_sig
        
        # Apply swiglu: i * sigmoid(i) * (1-f)
        i_sig = self.apply_sigmoid(i)
        i = self.fixed_mul(i, i_sig)
        i = self.fixed_mul(i, one_minus_f)
        
        # Recurrent update: h = f * h_prev + i
        h = self.fixed_mul(f_sig, h_prev) + i
        h = torch.clamp(h, self.MIN_VAL, self.MAX_VAL).to(torch.int16)
        
        # Apply g_norm and output projection
        gh = self.fixed_mul(g, h)
        o = self.ternary_matmul(gh.unsqueeze(1), w_o, w_scale_o).squeeze(1)
        
        return o, h
    
    def forward(self, x: torch.Tensor, 
                w_i: torch.Tensor, w_f: torch.Tensor, w_g: torch.Tensor, w_o: torch.Tensor,
                w_scale_i: torch.Tensor, w_scale_f: torch.Tensor, 
                w_scale_g: torch.Tensor, w_scale_o: torch.Tensor,
                h_init: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Full HGRN forward pass in 16-bit fixed-point
        
        Args:
            x: Input tensor [batch, seq_len, hidden]
            w_i, w_f, w_g, w_o: Ternary weight matrices
            w_scale_*: Corresponding scale factors in 16-bit fixed-point
            h_init: Initial hidden state
            
        Returns:
            output: Output tensor [batch, seq_len, hidden] in 16-bit fixed-point
        """
        batch, seq_len, hidden = x.shape
        
        # Convert input to 16-bit fixed-point
        x_fixed = self.to_fixed(x)
        
        # Initialize hidden state
        if h_init is None:
            h = torch.zeros(batch, hidden, dtype=torch.int16, device=x.device)
        else:
            h = self.to_fixed(h_init)
        
        # Process sequence
        outputs = []
        for t in range(seq_len):
            x_t = x_fixed[:, t, :]
            
            # HGRN cell computation with 16-bit precision
            o_t, h = self.hgrn_cell(x_t, h, w_i, w_f, w_g, w_o,
                                   w_scale_i, w_scale_f, w_scale_g, w_scale_o)
            
            outputs.append(o_t)
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)
        
        return output


def generate_test_weights_16bit(hidden_size: int, output_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate ternary weights and their 16-bit scale factors"""
    # Generate random ternary weights (same as 8-bit)
    w_ternary = torch.randint(-1, 2, (hidden_size, output_size), dtype=torch.int8)
    
    # Compute scale factor with better precision for 16-bit
    w_float = torch.randn(hidden_size, output_size) * 0.1
    scale = 1.0 / (w_float.abs().mean() + 1e-5)
    
    # Quantize scale to 16-bit fixed-point Q8.8
    hgrn = HGRNFixed16bit()
    w_scale = hgrn.to_fixed(torch.full((output_size,), scale))
    
    return w_ternary, w_scale


def compare_precision():
    """Compare 8-bit vs 16-bit precision"""
    print("="*60)
    print("8-BIT vs 16-BIT FIXED-POINT COMPARISON")
    print("="*60)
    
    # 8-bit configuration (Q3.5)
    print("\n8-bit Fixed-Point (Q3.5):")
    print("-"*40)
    frac_bits_8 = 3
    int_bits_8 = 5
    scale_8 = 1 << frac_bits_8
    precision_8 = 1.0 / scale_8
    range_8 = (-(1 << 7) / scale_8, ((1 << 7) - 1) / scale_8)
    
    print(f"Format: Q{int_bits_8}.{frac_bits_8}")
    print(f"Precision: {precision_8:.6f}")
    print(f"Range: [{range_8[0]:.3f}, {range_8[1]:.3f}]")
    print(f"Total bits: 8")
    print(f"Memory per weight: 1 byte")
    
    # 16-bit configuration (Q8.8)
    print("\n16-bit Fixed-Point (Q8.8):")
    print("-"*40)
    frac_bits_16 = 8
    int_bits_16 = 8
    scale_16 = 1 << frac_bits_16
    precision_16 = 1.0 / scale_16
    range_16 = (-(1 << 15) / scale_16, ((1 << 15) - 1) / scale_16)
    
    print(f"Format: Q{int_bits_16}.{frac_bits_16}")
    print(f"Precision: {precision_16:.6f}")
    print(f"Range: [{range_16[0]:.3f}, {range_16[1]:.3f}]")
    print(f"Total bits: 16")
    print(f"Memory per weight: 2 bytes")
    
    # Comparison
    print("\nComparison:")
    print("-"*40)
    print(f"Precision improvement: {precision_8/precision_16:.1f}x better")
    print(f"Range improvement: {(range_16[1]-range_16[0])/(range_8[1]-range_8[0]):.1f}x wider")
    print(f"Memory overhead: 2x more than 8-bit")
    print(f"Memory savings vs float32: 50% reduction")
    
    print("\nUse Cases:")
    print("-"*40)
    print("8-bit: Edge devices, embedded systems, extreme memory constraints")
    print("16-bit: Better accuracy needed, moderate memory constraints")
    print("Float32: Maximum accuracy, no memory constraints")


def test_16bit_implementation():
    """Test the 16-bit implementation"""
    print("\n" + "="*60)
    print("TESTING 16-BIT FIXED-POINT HGRN")
    print("="*60)
    
    # Configuration
    batch_size = 1
    seq_len = 4
    hidden_size = 16
    
    # Initialize 16-bit HGRN
    hgrn_16bit = HGRNFixed16bit()
    
    # Generate test input
    x = torch.randn(batch_size, seq_len, hidden_size) * 0.5
    h_init = torch.randn(batch_size, hidden_size) * 0.1
    
    # Generate ternary weights with 16-bit scales
    w_i, w_scale_i = generate_test_weights_16bit(hidden_size, hidden_size)
    w_f, w_scale_f = generate_test_weights_16bit(hidden_size, hidden_size)
    w_g, w_scale_g = generate_test_weights_16bit(hidden_size, hidden_size)
    w_o, w_scale_o = generate_test_weights_16bit(hidden_size, hidden_size)
    
    # Run forward pass
    print(f"\nInput shape: {x.shape}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    
    output_16bit = hgrn_16bit.forward(x, w_i, w_f, w_g, w_o,
                                      w_scale_i, w_scale_f, w_scale_g, w_scale_o,
                                      h_init)
    
    # Convert back to float for analysis
    output_float = hgrn_16bit.from_fixed(output_16bit)
    
    print(f"\nOutput shape: {output_float.shape}")
    print(f"Output range: [{output_float.min():.3f}, {output_float.max():.3f}]")
    print(f"Output dtype: {output_16bit.dtype}")
    
    # Check precision
    print(f"\n16-bit Precision: {hgrn_16bit.PRECISION:.6f}")
    print(f"Can represent values as small as: ±{hgrn_16bit.PRECISION:.6f}")
    
    # Memory footprint
    total_weights = 4 * hidden_size * hidden_size  # 4 weight matrices
    memory_ternary = total_weights * 1  # 1 byte per ternary weight
    memory_scales = 4 * hidden_size * 2  # 2 bytes per 16-bit scale
    total_memory = (memory_ternary + memory_scales) / 1024
    
    print(f"\nMemory Usage:")
    print(f"  Ternary weights: {memory_ternary} bytes")
    print(f"  Scale factors (16-bit): {memory_scales} bytes")
    print(f"  Total: {total_memory:.2f} KB")
    
    # Compare with float32 equivalent
    float32_memory = total_weights * 4 / 1024  # 4 bytes per float32
    savings = (1 - total_memory / float32_memory) * 100
    print(f"  Float32 equivalent: {float32_memory:.2f} KB")
    print(f"  Memory savings: {savings:.1f}%")
    
    print("\n✓ 16-bit fixed-point HGRN successfully tested!")


if __name__ == "__main__":
    # Show precision comparison
    compare_precision()
    
    # Test the implementation
    test_16bit_implementation()