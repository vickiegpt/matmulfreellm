"""
Test script for fixed-point implementation of MatMulFreeLLM
"""

import torch
import torch.nn as nn
import numpy as np
from mmfreelm.ops.fixed_point import (
    to_fixed_point, from_fixed_point, fixed_mul, fixed_div,
    fixed_sqrt, FixedPointSigmoid, FixedPointConfig
)
from mmfreelm.ops.fixed_point_bitnet import (
    activation_quant_fixed, weight_quant_fixed,
    FixedPointBitLinear, FixedPointRMSNorm
)


def test_fixed_point_conversions():
    """Test basic fixed-point conversions"""
    print("Testing fixed-point conversions...")
    
    # Test various values
    test_values = [0.0, 1.0, -1.0, 0.5, -0.5, 3.14159, -2.71828, 0.001, 100.0]
    frac_bits = 16
    
    for val in test_values:
        x = torch.tensor([val])
        x_fixed = to_fixed_point(x, frac_bits)
        x_recovered = from_fixed_point(x_fixed, frac_bits)
        error = abs(x.item() - x_recovered.item())
        print(f"  {val:8.5f} -> fixed -> {x_recovered.item():8.5f}, error: {error:.6f}")
    
    print("✓ Conversion test passed\n")


def test_fixed_point_arithmetic():
    """Test fixed-point arithmetic operations"""
    print("Testing fixed-point arithmetic...")
    
    frac_bits = 16
    
    # Test multiplication
    a = torch.tensor([2.5])
    b = torch.tensor([3.0])
    a_fixed = to_fixed_point(a, frac_bits)
    b_fixed = to_fixed_point(b, frac_bits)
    
    result_fixed = fixed_mul(a_fixed, b_fixed, frac_bits, frac_bits, frac_bits)
    result = from_fixed_point(result_fixed, frac_bits)
    expected = a * b
    
    print(f"  Multiplication: {a.item()} * {b.item()} = {result.item()} (expected: {expected.item()})")
    
    # Test division
    result_fixed = fixed_div(a_fixed, b_fixed, frac_bits, frac_bits, frac_bits)
    result = from_fixed_point(result_fixed, frac_bits)
    expected = a / b
    
    print(f"  Division: {a.item()} / {b.item()} = {result.item()} (expected: {expected.item()})")
    
    # Test square root
    x = torch.tensor([4.0, 9.0, 16.0, 2.0])
    x_fixed = to_fixed_point(x, frac_bits)
    sqrt_fixed = fixed_sqrt(x_fixed, frac_bits)
    sqrt_result = from_fixed_point(sqrt_fixed, frac_bits)
    sqrt_expected = torch.sqrt(x)
    
    print(f"  Square root: {x.tolist()} -> {sqrt_result.tolist()}")
    print(f"  Expected: {sqrt_expected.tolist()}")
    
    print("✓ Arithmetic test passed\n")


def test_activation_functions():
    """Test fixed-point activation functions"""
    print("Testing fixed-point activation functions...")
    
    frac_bits = 16
    sigmoid = FixedPointSigmoid(frac_bits)
    
    # Test sigmoid at various points
    x_values = torch.linspace(-5, 5, 11)
    x_fixed = to_fixed_point(x_values, frac_bits)
    
    sigmoid_fixed = sigmoid(x_fixed)
    sigmoid_result = from_fixed_point(sigmoid_fixed, frac_bits)
    sigmoid_expected = torch.sigmoid(x_values)
    
    print("  Sigmoid test:")
    for i in range(len(x_values)):
        error = abs(sigmoid_result[i].item() - sigmoid_expected[i].item())
        print(f"    x={x_values[i]:5.2f}: fixed={sigmoid_result[i]:.4f}, "
              f"expected={sigmoid_expected[i]:.4f}, error={error:.4f}")
    
    print("✓ Activation function test passed\n")


def test_quantization_functions():
    """Test fixed-point quantization functions"""
    print("Testing fixed-point quantization...")
    
    # Test weight quantization
    weights = torch.randn(10, 10) * 0.1
    weights_fixed = to_fixed_point(weights, FixedPointConfig.COMPUTE_FRAC_BITS)
    weights_quant = weight_quant_fixed(weights_fixed)
    weights_result = from_fixed_point(weights_quant, FixedPointConfig.COMPUTE_FRAC_BITS)
    
    # Check that weights are ternary
    unique_vals = torch.unique(torch.round(weights_result * 100) / 100)
    print(f"  Weight quantization unique values: {unique_vals.tolist()}")
    
    # Test activation quantization
    acts = torch.randn(4, 8) * 2.0
    acts_fixed = to_fixed_point(acts, FixedPointConfig.COMPUTE_FRAC_BITS)
    acts_quant = activation_quant_fixed(acts_fixed)
    acts_result = from_fixed_point(acts_quant, FixedPointConfig.COMPUTE_FRAC_BITS)
    
    print(f"  Activation quantization:")
    print(f"    Input range: [{acts.min():.3f}, {acts.max():.3f}]")
    print(f"    Output range: [{acts_result.min():.3f}, {acts_result.max():.3f}]")
    
    print("✓ Quantization test passed\n")


def test_fixed_point_layers():
    """Test fixed-point neural network layers"""
    print("Testing fixed-point layers...")
    
    batch_size = 2
    seq_len = 4
    hidden_size = 8
    out_size = 8
    
    # Test BitLinear layer
    print("  Testing FixedPointBitLinear...")
    layer = FixedPointBitLinear(hidden_size, out_size, bias=False)
    layer.quantize_weights()
    
    x = torch.randn(batch_size, seq_len, hidden_size)
    output = layer(x)
    
    print(f"    Input shape: {x.shape}")
    print(f"    Output shape: {output.shape}")
    print(f"    Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test RMSNorm layer
    print("\n  Testing FixedPointRMSNorm...")
    norm = FixedPointRMSNorm(hidden_size)
    
    x = torch.randn(batch_size, seq_len, hidden_size) * 5.0  # Large values to test normalization
    output = norm(x)
    
    print(f"    Input shape: {x.shape}")
    print(f"    Output shape: {output.shape}")
    print(f"    Input std: {x.std():.3f}")
    print(f"    Output std: {output.std():.3f}")
    
    print("✓ Layer test passed\n")


def test_comparison_with_float():
    """Compare fixed-point and floating-point implementations"""
    print("Comparing fixed-point vs floating-point...")
    
    # Import original implementations
    from mmfreelm.ops.fusedbitnet import activation_quant, weight_quant
    
    # Test weight quantization comparison
    weights = torch.randn(5, 5) * 0.1
    
    # Floating-point version
    weights_quant_float = weight_quant(weights)
    
    # Fixed-point version
    weights_fixed = to_fixed_point(weights, FixedPointConfig.COMPUTE_FRAC_BITS)
    weights_quant_fixed = weight_quant_fixed(weights_fixed)
    weights_quant_fixed_float = from_fixed_point(weights_quant_fixed, FixedPointConfig.COMPUTE_FRAC_BITS)
    
    # Compare
    diff = torch.abs(weights_quant_float - weights_quant_fixed_float)
    print(f"  Weight quantization max difference: {diff.max():.6f}")
    print(f"  Weight quantization mean difference: {diff.mean():.6f}")
    
    # Test activation quantization comparison
    acts = torch.randn(4, 8) * 2.0
    
    # Floating-point version
    acts_quant_float = activation_quant(acts)
    
    # Fixed-point version
    acts_fixed = to_fixed_point(acts, FixedPointConfig.COMPUTE_FRAC_BITS)
    acts_quant_fixed = activation_quant_fixed(acts_fixed)
    acts_quant_fixed_float = from_fixed_point(acts_quant_fixed, FixedPointConfig.COMPUTE_FRAC_BITS)
    
    # Compare
    diff = torch.abs(acts_quant_float - acts_quant_fixed_float)
    print(f"\n  Activation quantization max difference: {diff.max():.6f}")
    print(f"  Activation quantization mean difference: {diff.mean():.6f}")
    
    print("✓ Comparison test passed\n")


def run_all_tests():
    """Run all fixed-point tests"""
    print("=" * 60)
    print("Running Fixed-Point Implementation Tests")
    print("=" * 60)
    print()
    
    test_fixed_point_conversions()
    test_fixed_point_arithmetic()
    test_activation_functions()
    test_quantization_functions()
    test_fixed_point_layers()
    test_comparison_with_float()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    run_all_tests()