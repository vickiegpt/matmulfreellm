"""
Validation script to compare PyTorch HGRN with 8-bit fixed-point implementation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import os

from hgrn_8bit_fixed import HGRNFixed8bit, generate_test_weights
from mmfreelm.ops.hgrn.recurrent_fuse import fused_recurrent_hgrn
from mmfreelm.ops.fusedbitnet import weight_quant, activation_quant


class HGRNFloatingPoint(nn.Module):
    """Reference floating-point HGRN implementation for comparison"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Use ternary weights for fair comparison
        self.w_i = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.w_f = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.w_g = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.w_o = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        
    def quantize_weights(self):
        """Quantize weights to ternary values"""
        self.w_i.data = weight_quant(self.w_i.data)
        self.w_f.data = weight_quant(self.w_f.data)
        self.w_g.data = weight_quant(self.w_g.data)
        self.w_o.data = weight_quant(self.w_o.data)
        
    def forward(self, x: torch.Tensor, h_init: torch.Tensor = None) -> torch.Tensor:
        """Forward pass matching the HGRN algorithm"""
        batch, seq_len, hidden = x.shape
        
        if h_init is None:
            h = torch.zeros(batch, hidden, device=x.device)
        else:
            h = h_init
            
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Projections
            i = x_t @ self.w_i.T
            f = x_t @ self.w_f.T
            g = x_t @ self.w_g.T
            
            # Gate activation
            f_sig = torch.sigmoid(f)
            
            # Input modulation: swiglu(i, 1-f) = i * sigmoid(i) * (1-f)
            i = i * torch.sigmoid(i) * (1 - f_sig)
            
            # Recurrent update
            h = f_sig * h + i
            
            # Output projection
            o = (g * h) @ self.w_o.T
            outputs.append(o)
            
        return torch.stack(outputs, dim=1)


def compare_implementations(test_data: Dict) -> Dict:
    """Compare floating-point and fixed-point implementations"""
    
    # Extract test data
    x = test_data['input']['x']
    h_init = test_data['input']['h_init']
    hidden_size = test_data['config']['hidden_size']
    
    # Initialize models
    hgrn_float = HGRNFloatingPoint(hidden_size)
    hgrn_fixed = HGRNFixed8bit()
    
    # Set weights from test data
    # Convert ternary weights to float with scales
    w_i_float = test_data['weights']['w_i'].float() * hgrn_fixed.from_fixed(test_data['weights']['w_scale_i']).mean()
    w_f_float = test_data['weights']['w_f'].float() * hgrn_fixed.from_fixed(test_data['weights']['w_scale_f']).mean()
    w_g_float = test_data['weights']['w_g'].float() * hgrn_fixed.from_fixed(test_data['weights']['w_scale_g']).mean()
    w_o_float = test_data['weights']['w_o'].float() * hgrn_fixed.from_fixed(test_data['weights']['w_scale_o']).mean()
    
    hgrn_float.w_i.data = w_i_float
    hgrn_float.w_f.data = w_f_float
    hgrn_float.w_g.data = w_g_float
    hgrn_float.w_o.data = w_o_float
    
    # Run floating-point version
    with torch.no_grad():
        output_float = hgrn_float(x, h_init)
    
    # Get fixed-point output
    output_fixed = test_data['output']['o_float']
    
    # Compute errors
    abs_error = torch.abs(output_float - output_fixed)
    rel_error = abs_error / (torch.abs(output_float) + 1e-8)
    
    results = {
        'output_float': output_float,
        'output_fixed': output_fixed,
        'abs_error': abs_error,
        'rel_error': rel_error,
        'max_abs_error': abs_error.max().item(),
        'mean_abs_error': abs_error.mean().item(),
        'max_rel_error': rel_error.max().item(),
        'mean_rel_error': rel_error.mean().item()
    }
    
    return results


def generate_c_header(test_data: Dict, filename: str = "hgrn_test_vectors.h"):
    """Generate C header file with test vectors for assembly validation"""
    
    with open(filename, 'w') as f:
        f.write("/* HGRN 8-bit Fixed-Point Test Vectors */\n")
        f.write("/* Generated for assembly accelerator validation */\n\n")
        
        f.write("#ifndef HGRN_TEST_VECTORS_H\n")
        f.write("#define HGRN_TEST_VECTORS_H\n\n")
        
        # Configuration
        f.write(f"#define BATCH_SIZE {test_data['config']['batch_size']}\n")
        f.write(f"#define SEQ_LEN {test_data['config']['seq_len']}\n")
        f.write(f"#define HIDDEN_SIZE {test_data['config']['hidden_size']}\n")
        f.write(f"#define FRAC_BITS {test_data['config']['frac_bits']}\n")
        f.write(f"#define INT_BITS {test_data['config']['int_bits']}\n\n")
        
        # Helper function to write array
        def write_array(name: str, data: torch.Tensor, dtype: str = "int8_t"):
            f.write(f"const {dtype} {name}[] = {{\n")
            flat_data = data.flatten().tolist()
            for i in range(0, len(flat_data), 16):
                f.write("    ")
                f.write(", ".join(f"{int(x):4d}" for x in flat_data[i:i+16]))
                f.write(",\n" if i + 16 < len(flat_data) else "\n")
            f.write("};\n\n")
        
        # Write input vectors
        write_array("input_x", test_data['input']['x_fixed'])
        write_array("input_h_init", test_data['input']['h_init_fixed'])
        
        # Write weight matrices
        write_array("weight_i", test_data['weights']['w_i'])
        write_array("weight_f", test_data['weights']['w_f'])
        write_array("weight_g", test_data['weights']['w_g'])
        write_array("weight_o", test_data['weights']['w_o'])
        
        # Write scale factors
        write_array("scale_i", test_data['weights']['w_scale_i'])
        write_array("scale_f", test_data['weights']['w_scale_f'])
        write_array("scale_g", test_data['weights']['w_scale_g'])
        write_array("scale_o", test_data['weights']['w_scale_o'])
        
        # Write expected output
        write_array("expected_output", test_data['output']['o_fixed'])
        
        f.write("#endif /* HGRN_TEST_VECTORS_H */\n")
    
    print(f"Generated C header file: {filename}")


def visualize_comparison(results: Dict, test_data: Dict):
    """Visualize the comparison between implementations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Flatten for visualization
    output_float = results['output_float'].flatten().numpy()
    output_fixed = results['output_fixed'].flatten().numpy()
    abs_error = results['abs_error'].flatten().numpy()
    
    # 1. Output comparison
    ax = axes[0, 0]
    ax.scatter(output_float, output_fixed, alpha=0.5, s=10)
    ax.plot([output_float.min(), output_float.max()], 
            [output_float.min(), output_float.max()], 'r--', label='y=x')
    ax.set_xlabel('Floating-point output')
    ax.set_ylabel('Fixed-point output')
    ax.set_title('Output Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Absolute error distribution
    ax = axes[0, 1]
    ax.hist(abs_error, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Absolute Error Distribution\nMax: {results["max_abs_error"]:.6f}, Mean: {results["mean_abs_error"]:.6f}')
    ax.grid(True, alpha=0.3)
    
    # 3. Error over sequence
    ax = axes[1, 0]
    seq_len = test_data['config']['seq_len']
    hidden_size = test_data['config']['hidden_size']
    abs_error_seq = results['abs_error'].reshape(-1, seq_len, hidden_size)
    mean_error_per_step = abs_error_seq.mean(dim=(0, 2))
    ax.plot(range(seq_len), mean_error_per_step, 'b-o')
    ax.set_xlabel('Sequence Position')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Error Accumulation over Sequence')
    ax.grid(True, alpha=0.3)
    
    # 4. Relative error
    ax = axes[1, 1]
    rel_error_percent = results['rel_error'].flatten().numpy() * 100
    ax.hist(rel_error_percent[rel_error_percent < 50], bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Relative Error (%)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Relative Error Distribution\nMax: {results["max_rel_error"]*100:.2f}%, Mean: {results["mean_rel_error"]*100:.2f}%')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hgrn_comparison.png', dpi=150)
    print("Saved comparison plot: hgrn_comparison.png")


def main():
    print("HGRN Fixed-Point Validation")
    print("="*60)
    
    # Generate test vectors
    print("\n1. Generating test vectors...")
    test_data = torch.load('hgrn_test_vectors.pt') if os.path.exists('hgrn_test_vectors.pt') else None
    
    if test_data is None:
        from hgrn_8bit_fixed import generate_test_vectors
        test_data = generate_test_vectors()
        torch.save(test_data, 'hgrn_test_vectors.pt')
        print("   Generated and saved new test vectors")
    else:
        print("   Loaded existing test vectors")
    
    # Compare implementations
    print("\n2. Comparing implementations...")
    results = compare_implementations(test_data)
    
    print(f"\n   Results:")
    print(f"   - Max absolute error: {results['max_abs_error']:.6f}")
    print(f"   - Mean absolute error: {results['mean_abs_error']:.6f}")
    print(f"   - Max relative error: {results['max_rel_error']*100:.2f}%")
    print(f"   - Mean relative error: {results['mean_rel_error']*100:.2f}%")
    
    # Generate C header for assembly validation
    print("\n3. Generating C header file...")
    generate_c_header(test_data)
    
    # Visualize comparison
    print("\n4. Visualizing comparison...")
    visualize_comparison(results, test_data)
    
    # Generate assembly-friendly output format
    print("\n5. Assembly validation data:")
    print(f"\n   Input vector X (first 8 values):")
    x_fixed = test_data['input']['x_fixed'].flatten()[:8]
    print(f"   {x_fixed.tolist()}")
    
    print(f"\n   Initial hidden state H (first 8 values):")
    h_fixed = test_data['input']['h_init_fixed'].flatten()[:8]
    print(f"   {h_fixed.tolist()}")
    
    print(f"\n   Expected output O (first 8 values):")
    o_fixed = test_data['output']['o_fixed'].flatten()[:8]
    print(f"   {o_fixed.tolist()}")
    
    print("\n"+"="*60)
    print("Validation complete!")
    print("Use hgrn_test_vectors.h for assembly accelerator validation")


if __name__ == "__main__":
    main()