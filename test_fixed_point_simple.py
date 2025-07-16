#!/usr/bin/env python3
"""Simple test of the fixed-point HGRN implementation"""

import torch
from hgrn_8bit_fixed import HGRNFixed8bit, generate_test_weights

def test_tmatmul_operations():
    """Test that tmatmul is called 4 times per timestep"""
    print("Testing fixed-point HGRN with tmatmul operations...")
    
    # Configuration
    batch_size = 1
    seq_len = 2  # Short sequence for testing
    hidden_size = 16  # Small size for testing
    
    # Initialize HGRN
    hgrn = HGRNFixed8bit()
    
    # Generate test input
    x = torch.randn(batch_size, seq_len, hidden_size) * 0.5
    
    # Generate ternary weights
    w_i, w_scale_i = generate_test_weights(hidden_size, hidden_size)
    w_f, w_scale_f = generate_test_weights(hidden_size, hidden_size)
    w_g, w_scale_g = generate_test_weights(hidden_size, hidden_size)
    w_o, w_scale_o = generate_test_weights(hidden_size, hidden_size)
    
    # Count tmatmul calls
    tmatmul_count = 0
    original_tmatmul = hgrn.ternary_matmul
    
    def counting_tmatmul(self, x, w_ternary, w_scale):
        nonlocal tmatmul_count
        tmatmul_count += 1
        print(f"  tmatmul call #{tmatmul_count}: input shape {x.shape}")
        return original_tmatmul(x, w_ternary, w_scale)
    
    # Monkey patch to count calls
    hgrn.ternary_matmul = lambda x, w, s: counting_tmatmul(hgrn, x, w, s)
    
    # Run forward pass
    print(f"\nRunning forward pass with seq_len={seq_len}")
    output = hgrn.forward(x, w_i, w_f, w_g, w_o,
                         w_scale_i, w_scale_f, w_scale_g, w_scale_o)
    
    # Results
    print(f"\nResults:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Total tmatmul calls: {tmatmul_count}")
    print(f"  tmatmul calls per timestep: {tmatmul_count / seq_len}")
    
    # Verify
    expected_calls = seq_len * 4  # 4 tmatmul per timestep
    if tmatmul_count == expected_calls:
        print(f"\n✓ SUCCESS: Expected {expected_calls} tmatmul calls, got {tmatmul_count}")
        print("  Each timestep correctly performs 4 tmatmul operations:")
        print("    1. Input gate (w_i)")
        print("    2. Forget gate (w_f)")
        print("    3. Gate (w_g)")
        print("    4. Output projection (w_o)")
    else:
        print(f"\n✗ FAILURE: Expected {expected_calls} tmatmul calls, got {tmatmul_count}")
    
    return tmatmul_count == expected_calls

if __name__ == "__main__":
    success = test_tmatmul_operations()
    exit(0 if success else 1)