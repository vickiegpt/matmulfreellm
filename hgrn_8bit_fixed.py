"""
8-bit Fixed-Point Implementation of HGRN Algorithm
Designed to match assembly accelerator implementation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class HGRNFixed8bit:
    """
    8-bit fixed-point implementation of HGRN (Hierarchically Gated Recurrent Network)
    
    Algorithm overview:
    1. Input projection: i = W_i @ x, f = W_f @ x, g = W_g @ x
    2. Gate activation: f = sigmoid(f)
    3. Input modulation: i = swiglu(i, 1-f) = i * sigmoid(i) * (1-f)
    4. Recurrent update: h_t = f_t * h_{t-1} + i_t
    5. Output projection: o = W_o @ (g_norm(g) * h)
    
    Fixed-point format: Q3.5 (8-bit total: 3 integer bits, 5 fractional bits)
    Range: [-4, 3.96875] with precision 0.03125
    """
    
    def __init__(self):
        # Fixed-point configuration
        self.FRAC_BITS = 5
        self.INT_BITS = 3
        self.SCALE = 1 << self.FRAC_BITS  # 32
        self.MAX_VAL = (1 << 7) - 1  # 127
        self.MIN_VAL = -(1 << 7)     # -128
        
    def to_fixed(self, x: torch.Tensor) -> torch.Tensor:
        """Convert floating-point to 8-bit fixed-point Q3.5"""
        x_scaled = (x * self.SCALE).round()
        x_clamped = torch.clamp(x_scaled, self.MIN_VAL, self.MAX_VAL)
        return x_clamped.to(torch.int8)
    
    def from_fixed(self, x: torch.Tensor) -> torch.Tensor:
        """Convert 8-bit fixed-point Q3.5 to floating-point"""
        return x.float() / self.SCALE
    
    def fixed_mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multiply two fixed-point numbers"""
        # Cast to int16 to prevent overflow
        result = (a.to(torch.int16) * b.to(torch.int16)) >> self.FRAC_BITS
        return torch.clamp(result, self.MIN_VAL, self.MAX_VAL).to(torch.int8)
    
    def sigmoid_lut(self) -> torch.Tensor:
        """Generate sigmoid lookup table for 8-bit values"""
        # Create lookup table for sigmoid function
        x = torch.arange(self.MIN_VAL, self.MAX_VAL + 1, dtype=torch.float32) / self.SCALE
        sigmoid_vals = torch.sigmoid(x)
        return self.to_fixed(sigmoid_vals)
    
    def apply_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid using lookup table"""
        lut = self.sigmoid_lut()
        # Shift indices to handle negative values
        indices = (x.to(torch.int16) - self.MIN_VAL).to(torch.long)
        indices = torch.clamp(indices, 0, len(lut) - 1)
        return lut[indices]
    
    def ternary_matmul(self, x: torch.Tensor, w_ternary: torch.Tensor, 
                       w_scale: torch.Tensor) -> torch.Tensor:
        """
        Perform matrix multiplication with ternary weights
        
        Args:
            x: Input tensor [batch, seq_len, hidden] in fixed-point
            w_ternary: Ternary weight matrix {-1, 0, 1} [hidden, out]
            w_scale: Per-channel scale factors [out] in fixed-point
            
        Returns:
            Output tensor [batch, seq_len, out] in fixed-point
        """
        batch, seq_len, hidden = x.shape
        out_dim = w_ternary.shape[1]
        
        # Initialize output
        output = torch.zeros(batch, seq_len, out_dim, dtype=torch.int16)
        
        # Efficient ternary multiplication
        for i in range(out_dim):
            # Get ternary weights for this output channel
            w_col = w_ternary[:, i]
            
            # Compute dot product using only additions/subtractions
            pos_mask = w_col == 1
            neg_mask = w_col == -1
            
            if pos_mask.any():
                output[:, :, i] += x[:, :, pos_mask].sum(dim=-1).to(torch.int16)
            if neg_mask.any():
                output[:, :, i] -= x[:, :, neg_mask].sum(dim=-1).to(torch.int16)
        
        # Apply scale factors
        output = (output * w_scale.unsqueeze(0).unsqueeze(0).to(torch.int16)) >> self.FRAC_BITS
        
        return torch.clamp(output, self.MIN_VAL, self.MAX_VAL).to(torch.int8)
    
    def hgrn_cell(self, x: torch.Tensor, h_prev: torch.Tensor,
                  w_i: torch.Tensor, w_f: torch.Tensor, w_g: torch.Tensor,
                  w_scale_i: torch.Tensor, w_scale_f: torch.Tensor, w_scale_g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single HGRN cell computation in fixed-point
        
        Args:
            x: Input [batch, hidden] in fixed-point
            h_prev: Previous hidden state [batch, hidden] in fixed-point
            w_i, w_f, w_g: Ternary weight matrices
            w_scale_i, w_scale_f, w_scale_g: Scale factors
            
        Returns:
            i: Input gate output
            h: New hidden state
        """
        # Project input with ternary matrices
        i = self.ternary_matmul(x.unsqueeze(1), w_i, w_scale_i).squeeze(1)
        f = self.ternary_matmul(x.unsqueeze(1), w_f, w_scale_f).squeeze(1)
        
        # Apply sigmoid to forget gate
        f_sig = self.apply_sigmoid(f)
        
        # Compute 1 - f (in fixed-point)
        one_fixed = self.to_fixed(torch.ones_like(f_sig, dtype=torch.float32))
        one_minus_f = one_fixed - f_sig
        
        # Apply swiglu: i * sigmoid(i) * (1-f)
        i_sig = self.apply_sigmoid(i)
        i = self.fixed_mul(i, i_sig)
        i = self.fixed_mul(i, one_minus_f)
        
        # Recurrent update: h = f * h_prev + i
        h = self.fixed_mul(f_sig, h_prev) + i
        h = torch.clamp(h, self.MIN_VAL, self.MAX_VAL).to(torch.int8)
        
        return i, h
    
    def forward(self, x: torch.Tensor, 
                w_i: torch.Tensor, w_f: torch.Tensor, w_g: torch.Tensor, w_o: torch.Tensor,
                w_scale_i: torch.Tensor, w_scale_f: torch.Tensor, 
                w_scale_g: torch.Tensor, w_scale_o: torch.Tensor,
                h_init: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Full HGRN forward pass in fixed-point
        
        Args:
            x: Input tensor [batch, seq_len, hidden]
            w_i, w_f, w_g, w_o: Ternary weight matrices
            w_scale_*: Corresponding scale factors
            h_init: Initial hidden state
            
        Returns:
            output: Output tensor [batch, seq_len, hidden]
        """
        batch, seq_len, hidden = x.shape
        
        # Convert input to fixed-point
        x_fixed = self.to_fixed(x)
        
        # Initialize hidden state
        if h_init is None:
            h = torch.zeros(batch, hidden, dtype=torch.int8)
        else:
            h = self.to_fixed(h_init)
        
        # Process sequence
        outputs = []
        for t in range(seq_len):
            x_t = x_fixed[:, t, :]
            
            # HGRN cell computation
            i_t, h = self.hgrn_cell(x_t, h, w_i, w_f, w_g, 
                                   w_scale_i, w_scale_f, w_scale_g)
            
            # Project with g and output
            g_t = self.ternary_matmul(x_t.unsqueeze(1), w_g, w_scale_g).squeeze(1)
            
            # g_norm (simplified as just multiplication for now)
            gh = self.fixed_mul(g_t, h)
            
            # Output projection
            o_t = self.ternary_matmul(gh.unsqueeze(1), w_o, w_scale_o).squeeze(1)
            outputs.append(o_t)
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)
        
        return output


def generate_test_weights(hidden_size: int, output_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate ternary weights and their scale factors"""
    # Generate random ternary weights
    w_ternary = torch.randint(-1, 2, (hidden_size, output_size), dtype=torch.int8)
    
    # Compute scale factor (mean of absolute values)
    w_float = torch.randn(hidden_size, output_size) * 0.1
    scale = 1.0 / (w_float.abs().mean() + 1e-5)
    
    # Quantize scale to fixed-point Q3.5
    hgrn = HGRNFixed8bit()
    w_scale = hgrn.to_fixed(torch.full((output_size,), scale))
    
    return w_ternary, w_scale


def generate_test_vectors():
    """Generate test vectors for validation"""
    # Configuration
    batch_size = 1
    seq_len = 4
    hidden_size = 16
    
    # Initialize HGRN
    hgrn = HGRNFixed8bit()
    
    # Generate input
    x = torch.randn(batch_size, seq_len, hidden_size) * 0.5
    h_init = torch.randn(batch_size, hidden_size) * 0.1
    
    # Generate ternary weights and scales
    w_i, w_scale_i = generate_test_weights(hidden_size, hidden_size)
    w_f, w_scale_f = generate_test_weights(hidden_size, hidden_size)
    w_g, w_scale_g = generate_test_weights(hidden_size, hidden_size)
    w_o, w_scale_o = generate_test_weights(hidden_size, hidden_size)
    
    # Run forward pass
    output = hgrn.forward(x, w_i, w_f, w_g, w_o,
                         w_scale_i, w_scale_f, w_scale_g, w_scale_o,
                         h_init)
    
    # Convert back to float for validation
    output_float = hgrn.from_fixed(output)
    
    # Save test vectors
    test_data = {
        'input': {
            'x': x,
            'h_init': h_init,
            'x_fixed': hgrn.to_fixed(x),
            'h_init_fixed': hgrn.to_fixed(h_init)
        },
        'weights': {
            'w_i': w_i, 'w_scale_i': w_scale_i,
            'w_f': w_f, 'w_scale_f': w_scale_f,
            'w_g': w_g, 'w_scale_g': w_scale_g,
            'w_o': w_o, 'w_scale_o': w_scale_o
        },
        'output': {
            'o_fixed': output,
            'o_float': output_float
        },
        'config': {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'hidden_size': hidden_size,
            'frac_bits': hgrn.FRAC_BITS,
            'int_bits': hgrn.INT_BITS
        }
    }
    
    return test_data


if __name__ == "__main__":
    # Generate test vectors
    test_data = generate_test_vectors()
    
    print("HGRN 8-bit Fixed-Point Test Vectors")
    print("="*50)
    print(f"Configuration:")
    print(f"  Batch size: {test_data['config']['batch_size']}")
    print(f"  Sequence length: {test_data['config']['seq_len']}")
    print(f"  Hidden size: {test_data['config']['hidden_size']}")
    print(f"  Fixed-point format: Q{test_data['config']['int_bits']}.{test_data['config']['frac_bits']}")
    print(f"\nInput shape: {test_data['input']['x'].shape}")
    print(f"Output shape: {test_data['output']['o_float'].shape}")
    print(f"\nInput range: [{test_data['input']['x'].min():.3f}, {test_data['input']['x'].max():.3f}]")
    print(f"Output range: [{test_data['output']['o_float'].min():.3f}, {test_data['output']['o_float'].max():.3f}]")
    
    # Save test vectors for assembly validation
    torch.save(test_data, 'hgrn_test_vectors.pt')