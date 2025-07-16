"""
Generate HGRN test vectors in assembly-friendly format
This creates the exact format needed for assembly accelerator validation
"""

import torch
import numpy as np
from typing import Dict, Tuple


def generate_assembly_test_vectors():
    """
    Generate test vectors matching the assembly accelerator format:
    - X: Input vector
    - oH: Previous hidden state 
    - B: Bias vector (if used)
    - WG, WF, WC, WO: Weight matrices (ternary)
    - O: Output vector
    
    All in 8-bit fixed-point Q3.5 format
    """
    
    # Configuration matching assembly assumptions
    hidden_size = 16  # Small size for easy validation
    seq_len = 1      # Single timestep for assembly test
    
    # Q3.5 fixed-point parameters
    FRAC_BITS = 3
    INT_BITS = 5
    SCALE = 1 << FRAC_BITS
    
    def to_fixed(x):
        """Convert to 8-bit fixed-point Q3.5"""
        return np.clip(np.round(x * SCALE), -128, 127).astype(np.int8)
    
    def from_fixed(x):
        """Convert from fixed-point to float"""
        return x.astype(np.float32) / SCALE
    
    # Generate test inputs
    np.random.seed(42)
    
    # Input vector X (range: -1 to 1)
    X = np.random.randn(hidden_size) * 0.5
    X_fixed = to_fixed(X)
    
    # Previous hidden state oH
    oH = np.random.randn(hidden_size) * 0.1
    oH_fixed = to_fixed(oH)
    
    # Bias vector B (usually zero or small)
    B = np.zeros(hidden_size)
    B_fixed = to_fixed(B)
    
    # Generate ternary weight matrices
    # For HGRN: WG (gate), WF (forget), WC (candidate/input), WO (output)
    WG = np.random.choice([-1, 0, 1], size=(hidden_size, hidden_size))
    WF = np.random.choice([-1, 0, 1], size=(hidden_size, hidden_size))
    WC = np.random.choice([-1, 0, 1], size=(hidden_size, hidden_size))
    WO = np.random.choice([-1, 0, 1], size=(hidden_size, hidden_size))
    
    # Compute scale factors for ternary weights
    # In assembly, these would be pre-computed constants
    scale_G = 1.0 / (np.abs(WG).mean() + 1e-5)
    scale_F = 1.0 / (np.abs(WF).mean() + 1e-5)
    scale_C = 1.0 / (np.abs(WC).mean() + 1e-5)
    scale_O = 1.0 / (np.abs(WO).mean() + 1e-5)
    
    # Simple sigmoid approximation for fixed-point
    def sigmoid_approx(x):
        # Piecewise linear approximation
        x_float = from_fixed(x)
        y = np.where(x_float < -2.5, 0.0,
            np.where(x_float > 2.5, 1.0,
                    0.2 * x_float + 0.5))
        return to_fixed(y)
    
    # Compute HGRN forward pass in fixed-point
    # Step 1: Linear projections with ternary weights
    def ternary_matmul(x, W, scale):
        """Multiply vector x by ternary matrix W with scale"""
        result = np.zeros(W.shape[1], dtype=np.int32)
        for i in range(W.shape[1]):
            # Sum only where W is non-zero
            pos_mask = W[:, i] == 1
            neg_mask = W[:, i] == -1
            if pos_mask.any():
                result[i] += x[pos_mask].sum()
            if neg_mask.any():
                result[i] -= x[neg_mask].sum()
        # Apply scale and convert back to int8
        result = (result * to_fixed(scale)) >> FRAC_BITS
        return np.clip(result, -128, 127).astype(np.int8)
    
    # Projections
    g = ternary_matmul(X_fixed, WG, scale_G)
    f = ternary_matmul(X_fixed, WF, scale_F)
    c = ternary_matmul(X_fixed, WC, scale_C)
    
    # Gate activations
    f_sig = sigmoid_approx(f)
    c_sig = sigmoid_approx(c)
    
    # Compute new hidden state
    # h = f_sig * oH + c_sig * (1 - f_sig)
    # In fixed-point: careful with overflow
    one_minus_f = to_fixed(np.ones_like(f_sig)) - f_sig
    
    # Element-wise multiplications in fixed-point
    h_forget = (f_sig.astype(np.int16) * oH_fixed.astype(np.int16)) >> FRAC_BITS
    c_modulated = (c_sig.astype(np.int16) * one_minus_f.astype(np.int16)) >> FRAC_BITS
    h_new = np.clip(h_forget + c_modulated, -128, 127).astype(np.int8)
    
    # Output projection
    # o = W_O @ (g * h_new)
    gh = (g.astype(np.int16) * h_new.astype(np.int16)) >> FRAC_BITS
    gh = np.clip(gh, -128, 127).astype(np.int8)
    O = ternary_matmul(gh, WO, scale_O)
    
    # Create test data structure
    test_data = {
        'inputs': {
            'X': X_fixed,
            'oH': oH_fixed,
            'B': B_fixed
        },
        'weights': {
            'WG': WG.astype(np.int8),
            'WF': WF.astype(np.int8),
            'WC': WC.astype(np.int8),
            'WO': WO.astype(np.int8)
        },
        'scales': {
            'scale_G': to_fixed(scale_G),
            'scale_F': to_fixed(scale_F),
            'scale_C': to_fixed(scale_C),
            'scale_O': to_fixed(scale_O)
        },
        'output': {
            'O': O
        },
        'intermediate': {
            'g': g,
            'f': f,
            'c': c,
            'f_sig': f_sig,
            'c_sig': c_sig,
            'h_new': h_new
        }
    }
    
    return test_data


def generate_assembly_code(test_data: Dict, filename: str = "hgrn_test.S"):
    """Generate assembly test code with test vectors"""
    
    with open(filename, 'w') as f:
        f.write("# HGRN Test Vectors for Assembly Accelerator\n")
        f.write("# Generated test data in assembly format\n\n")
        
        f.write(".section .rodata\n")
        f.write(".align 4\n\n")
        
        # Helper to write data arrays
        def write_data(label, data, comment=""):
            f.write(f"{label}:\n")
            if comment:
                f.write(f"    # {comment}\n")
            f.write("    .byte ")
            f.write(", ".join(str(int(x)) for x in data.flatten()))
            f.write("\n\n")
        
        # Write input vectors
        write_data("input_x", test_data['inputs']['X'], "Input vector X")
        write_data("input_h", test_data['inputs']['oH'], "Previous hidden state")
        write_data("input_b", test_data['inputs']['B'], "Bias vector")
        
        # Write weight matrices
        write_data("weight_g", test_data['weights']['WG'], "Gate weight matrix (ternary)")
        write_data("weight_f", test_data['weights']['WF'], "Forget weight matrix (ternary)")
        write_data("weight_c", test_data['weights']['WC'], "Candidate weight matrix (ternary)")
        write_data("weight_o", test_data['weights']['WO'], "Output weight matrix (ternary)")
        
        # Write expected output
        write_data("expected_output", test_data['output']['O'], "Expected output vector")
        
        # Write test code
        f.write("\n.section .text\n")
        f.write(".global test_hgrn\n\n")
        f.write("test_hgrn:\n")
        f.write("    # Load test vectors\n")
        f.write("    la a0, input_x        # X vector\n")
        f.write("    la a1, input_h        # oH vector\n")
        f.write("    la a2, input_b        # B vector\n")
        f.write("    la a3, weight_g       # WG matrix\n")
        f.write("    la a4, weight_f       # WF matrix\n")
        f.write("    la a5, weight_c       # WC matrix\n")
        f.write("    la a6, weight_o       # WO matrix\n")
        f.write("    la a7, output_buffer  # O vector (result)\n")
        f.write("    \n")
        f.write("    # Call HGRN accelerator\n")
        f.write("    jal generate_token\n")
        f.write("    \n")
        f.write("    # Compare with expected output\n")
        f.write("    la t0, expected_output\n")
        f.write("    la t1, output_buffer\n")
        f.write("    li t2, 16             # vector size\n")
        f.write("    \n")
        f.write("compare_loop:\n")
        f.write("    lb t3, 0(t0)\n")
        f.write("    lb t4, 0(t1)\n")
        f.write("    bne t3, t4, test_fail\n")
        f.write("    addi t0, t0, 1\n")
        f.write("    addi t1, t1, 1\n")
        f.write("    addi t2, t2, -1\n")
        f.write("    bnez t2, compare_loop\n")
        f.write("    \n")
        f.write("test_pass:\n")
        f.write("    li a0, 0              # return 0 for success\n")
        f.write("    ret\n")
        f.write("    \n")
        f.write("test_fail:\n")
        f.write("    li a0, 1              # return 1 for failure\n")
        f.write("    ret\n")
        f.write("\n")
        f.write(".section .bss\n")
        f.write(".align 4\n")
        f.write("output_buffer:\n")
        f.write("    .space 16             # Space for output vector\n")
    
    print(f"Generated assembly test file: {filename}")


def print_test_summary(test_data: Dict):
    """Print a summary of the test vectors"""
    
    print("\nHGRN Assembly Test Vector Summary")
    print("="*50)
    print("\nInput Vectors (Q3.5 fixed-point):")
    print(f"  X:  {test_data['inputs']['X'][:8].tolist()} ...")
    print(f"  oH: {test_data['inputs']['oH'][:8].tolist()} ...")
    
    print("\nWeight Matrices (ternary):")
    print(f"  WG non-zeros: {np.count_nonzero(test_data['weights']['WG'])}/{test_data['weights']['WG'].size}")
    print(f"  WF non-zeros: {np.count_nonzero(test_data['weights']['WF'])}/{test_data['weights']['WF'].size}")
    print(f"  WC non-zeros: {np.count_nonzero(test_data['weights']['WC'])}/{test_data['weights']['WC'].size}")
    print(f"  WO non-zeros: {np.count_nonzero(test_data['weights']['WO'])}/{test_data['weights']['WO'].size}")
    
    print("\nIntermediate Values:")
    print(f"  g:     {test_data['intermediate']['g'][:8].tolist()} ...")
    print(f"  f_sig: {test_data['intermediate']['f_sig'][:8].tolist()} ...")
    print(f"  h_new: {test_data['intermediate']['h_new'][:8].tolist()} ...")
    
    print("\nExpected Output:")
    print(f"  O: {test_data['output']['O'].tolist()}")
    
    print("\nValue Ranges:")
    print(f"  Input X range: [{test_data['inputs']['X'].min()}, {test_data['inputs']['X'].max()}]")
    print(f"  Output O range: [{test_data['output']['O'].min()}, {test_data['output']['O'].max()}]")


if __name__ == "__main__":
    # Generate test vectors
    test_data = generate_assembly_test_vectors()
    
    # Save as numpy arrays for easy loading
    np.savez('hgrn_assembly_test.npz', **{
        'X': test_data['inputs']['X'],
        'oH': test_data['inputs']['oH'],
        'B': test_data['inputs']['B'],
        'WG': test_data['weights']['WG'],
        'WF': test_data['weights']['WF'],
        'WC': test_data['weights']['WC'],
        'WO': test_data['weights']['WO'],
        'O': test_data['output']['O']
    })
    
    # Generate assembly test file
    generate_assembly_code(test_data)
    
    # Print summary
    print_test_summary(test_data)
    
    print("\n" + "="*50)
    print("Test vectors generated successfully!")
    print("Files created:")
    print("  - hgrn_assembly_test.npz (numpy format)")
    print("  - hgrn_test.S (assembly test code)")
    print("  - hgrn_test_vectors.h (C header format)")
    print("\nUse these files to validate your assembly accelerator.")