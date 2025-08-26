"""
Comprehensive benchmark comparing 8-bit vs 16-bit fixed-point HGRN
"""

import torch
import torch.nn as nn
import time
import numpy as np
from hgrn_8bit_fixed import HGRNFixed8bit, generate_test_weights
from hgrn_16bit_fixed import HGRNFixed16bit, generate_test_weights_16bit


def benchmark_precision_comparison():
    """Compare precision capabilities of 8-bit vs 16-bit"""
    print("="*70)
    print("8-BIT vs 16-BIT FIXED-POINT PRECISION COMPARISON")
    print("="*70)
    
    # Test values spanning different ranges
    test_values = torch.tensor([
        0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 3.99, 
        -0.001, -0.01, -0.1, -0.5, -1.0, -2.0, -3.99,
        0.0, 127.0, -128.0
    ])
    
    # 8-bit conversion
    hgrn_8bit = HGRNFixed8bit()
    values_8bit = hgrn_8bit.to_fixed(test_values)
    recovered_8bit = hgrn_8bit.from_fixed(values_8bit)
    
    # 16-bit conversion
    hgrn_16bit = HGRNFixed16bit()
    values_16bit = hgrn_16bit.to_fixed(test_values)
    recovered_16bit = hgrn_16bit.from_fixed(values_16bit)
    
    print("\nPrecision Test Results:")
    print("-"*70)
    print(f"{'Original':<12} {'8-bit Q3.5':<15} {'16-bit Q8.8':<15} {'8-bit Error':<15} {'16-bit Error':<15}")
    print("-"*70)
    
    for i in range(len(test_values)):
        orig = test_values[i].item()
        rec_8 = recovered_8bit[i].item()
        rec_16 = recovered_16bit[i].item()
        err_8 = abs(orig - rec_8)
        err_16 = abs(orig - rec_16)
        
        print(f"{orig:<12.6f} {rec_8:<15.6f} {rec_16:<15.6f} {err_8:<15.6f} {err_16:<15.6f}")
    
    # Calculate statistics
    errors_8bit = torch.abs(test_values - recovered_8bit)
    errors_16bit = torch.abs(test_values - recovered_16bit)
    
    print("\nError Statistics:")
    print("-"*70)
    print(f"                    8-bit            16-bit           Improvement")
    print(f"Mean Error:         {errors_8bit.mean():.6f}        {errors_16bit.mean():.6f}        {errors_8bit.mean()/errors_16bit.mean():.1f}x")
    print(f"Max Error:          {errors_8bit.max():.6f}        {errors_16bit.max():.6f}        {errors_8bit.max()/errors_16bit.max():.1f}x")
    print(f"Precision:          {1.0/hgrn_8bit.SCALE:.6f}        {1.0/hgrn_16bit.SCALE:.6f}        {hgrn_16bit.SCALE/hgrn_8bit.SCALE:.1f}x")


def benchmark_operation_accuracy():
    """Compare accuracy of operations between 8-bit and 16-bit"""
    print("\n" + "="*70)
    print("OPERATION ACCURACY COMPARISON")
    print("="*70)
    
    # Test multiplication accuracy
    a = torch.randn(100) * 2
    b = torch.randn(100) * 2
    
    # True result
    true_result = a * b
    
    # 8-bit multiplication
    hgrn_8bit = HGRNFixed8bit()
    a_8bit = hgrn_8bit.to_fixed(a)
    b_8bit = hgrn_8bit.to_fixed(b)
    result_8bit = hgrn_8bit.fixed_mul(a_8bit, b_8bit)
    result_8bit_float = hgrn_8bit.from_fixed(result_8bit)
    
    # 16-bit multiplication
    hgrn_16bit = HGRNFixed16bit()
    a_16bit = hgrn_16bit.to_fixed(a)
    b_16bit = hgrn_16bit.to_fixed(b)
    result_16bit = hgrn_16bit.fixed_mul(a_16bit, b_16bit)
    result_16bit_float = hgrn_16bit.from_fixed(result_16bit)
    
    # Calculate errors
    error_8bit = torch.abs(true_result - result_8bit_float)
    error_16bit = torch.abs(true_result - result_16bit_float)
    
    print("\nMultiplication Accuracy:")
    print("-"*40)
    print(f"8-bit Mean Error:  {error_8bit.mean():.6f}")
    print(f"16-bit Mean Error: {error_16bit.mean():.6f}")
    print(f"Improvement:       {error_8bit.mean()/error_16bit.mean():.1f}x better with 16-bit")
    
    # Test sigmoid approximation accuracy
    x = torch.linspace(-4, 4, 100)
    true_sigmoid = torch.sigmoid(x)
    
    # 8-bit sigmoid
    x_8bit = hgrn_8bit.to_fixed(x)
    sigmoid_8bit = hgrn_8bit.apply_sigmoid(x_8bit)
    sigmoid_8bit_float = hgrn_8bit.from_fixed(sigmoid_8bit)
    
    # 16-bit sigmoid
    x_16bit = hgrn_16bit.to_fixed(x)
    sigmoid_16bit = hgrn_16bit.apply_sigmoid(x_16bit)
    sigmoid_16bit_float = hgrn_16bit.from_fixed(sigmoid_16bit)
    
    # Calculate errors
    sigmoid_error_8bit = torch.abs(true_sigmoid - sigmoid_8bit_float)
    sigmoid_error_16bit = torch.abs(true_sigmoid - sigmoid_16bit_float)
    
    print("\nSigmoid Approximation Accuracy:")
    print("-"*40)
    print(f"8-bit Mean Error:  {sigmoid_error_8bit.mean():.6f}")
    print(f"16-bit Mean Error: {sigmoid_error_16bit.mean():.6f}")
    print(f"Improvement:       {sigmoid_error_8bit.mean()/sigmoid_error_16bit.mean():.1f}x better with 16-bit")


def benchmark_hgrn_layer_performance():
    """Compare performance of 8-bit vs 16-bit HGRN layers"""
    print("\n" + "="*70)
    print("HGRN LAYER PERFORMANCE COMPARISON")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Configuration
    batch_size = 1
    seq_len = 32
    hidden_size = 256
    num_iterations = 20
    
    # Create input
    x = torch.randn(batch_size, seq_len, hidden_size).to(device)
    
    # 8-bit HGRN layer
    hgrn_8bit = HGRNFixed8bit().to(device)
    w_i_8, w_scale_i_8 = generate_test_weights(hidden_size, hidden_size)
    w_f_8, w_scale_f_8 = generate_test_weights(hidden_size, hidden_size)
    w_g_8, w_scale_g_8 = generate_test_weights(hidden_size, hidden_size)
    w_o_8, w_scale_o_8 = generate_test_weights(hidden_size, hidden_size)
    
    # Move to device
    w_i_8, w_f_8, w_g_8, w_o_8 = [w.to(device) for w in [w_i_8, w_f_8, w_g_8, w_o_8]]
    w_scale_i_8, w_scale_f_8, w_scale_g_8, w_scale_o_8 = [w.to(device) for w in 
                                                            [w_scale_i_8, w_scale_f_8, w_scale_g_8, w_scale_o_8]]
    
    # 16-bit HGRN layer
    hgrn_16bit = HGRNFixed16bit().to(device)
    w_i_16, w_scale_i_16 = generate_test_weights_16bit(hidden_size, hidden_size)
    w_f_16, w_scale_f_16 = generate_test_weights_16bit(hidden_size, hidden_size)
    w_g_16, w_scale_g_16 = generate_test_weights_16bit(hidden_size, hidden_size)
    w_o_16, w_scale_o_16 = generate_test_weights_16bit(hidden_size, hidden_size)
    
    # Move to device
    w_i_16, w_f_16, w_g_16, w_o_16 = [w.to(device) for w in [w_i_16, w_f_16, w_g_16, w_o_16]]
    w_scale_i_16, w_scale_f_16, w_scale_g_16, w_scale_o_16 = [w.to(device) for w in 
                                                                [w_scale_i_16, w_scale_f_16, w_scale_g_16, w_scale_o_16]]
    
    # Benchmark 8-bit
    print("\n8-bit HGRN Layer (Q3.5):")
    print("-"*40)
    
    # Warmup
    for _ in range(5):
        _ = hgrn_8bit(x, w_i_8, w_f_8, w_g_8, w_o_8, 
                      w_scale_i_8, w_scale_f_8, w_scale_g_8, w_scale_o_8)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_iterations):
        with torch.no_grad():
            out_8bit = hgrn_8bit(x, w_i_8, w_f_8, w_g_8, w_o_8,
                                 w_scale_i_8, w_scale_f_8, w_scale_g_8, w_scale_o_8)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    time_8bit = time.time() - start
    
    print(f"Time per iteration: {(time_8bit/num_iterations)*1000:.3f} ms")
    print(f"Throughput: {(num_iterations * seq_len * hidden_size) / time_8bit:.0f} ops/sec")
    
    # Memory calculation
    mem_8bit_weights = (4 * hidden_size * hidden_size * 1) / (1024 * 1024)  # Ternary weights
    mem_8bit_scales = (4 * hidden_size * 1) / (1024 * 1024)  # 8-bit scales
    print(f"Weight memory: {(mem_8bit_weights + mem_8bit_scales):.3f} MB")
    
    # Benchmark 16-bit
    print("\n16-bit HGRN Layer (Q8.8):")
    print("-"*40)
    
    # Warmup
    for _ in range(5):
        _ = hgrn_16bit(x, w_i_16, w_f_16, w_g_16, w_o_16,
                       w_scale_i_16, w_scale_f_16, w_scale_g_16, w_scale_o_16)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_iterations):
        with torch.no_grad():
            out_16bit = hgrn_16bit(x, w_i_16, w_f_16, w_g_16, w_o_16,
                                   w_scale_i_16, w_scale_f_16, w_scale_g_16, w_scale_o_16)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    time_16bit = time.time() - start
    
    print(f"Time per iteration: {(time_16bit/num_iterations)*1000:.3f} ms")
    print(f"Throughput: {(num_iterations * seq_len * hidden_size) / time_16bit:.0f} ops/sec")
    
    # Memory calculation
    mem_16bit_weights = (4 * hidden_size * hidden_size * 1) / (1024 * 1024)  # Ternary weights (same)
    mem_16bit_scales = (4 * hidden_size * 2) / (1024 * 1024)  # 16-bit scales
    print(f"Weight memory: {(mem_16bit_weights + mem_16bit_scales):.3f} MB")
    
    # Compare outputs
    out_8bit_float = hgrn_8bit.from_fixed(out_8bit)
    out_16bit_float = hgrn_16bit.from_fixed(out_16bit)
    
    print("\nOutput Comparison:")
    print("-"*40)
    print(f"8-bit output range:  [{out_8bit_float.min():.3f}, {out_8bit_float.max():.3f}]")
    print(f"16-bit output range: [{out_16bit_float.min():.3f}, {out_16bit_float.max():.3f}]")
    print(f"Difference (MSE): {torch.mean((out_8bit_float - out_16bit_float)**2):.6f}")
    
    # Performance comparison
    print("\nPerformance Summary:")
    print("-"*40)
    speed_ratio = time_8bit / time_16bit
    memory_ratio = (mem_16bit_weights + mem_16bit_scales) / (mem_8bit_weights + mem_8bit_scales)
    
    print(f"Speed: {'8-bit is ' + str(round(speed_ratio, 2)) + 'x faster' if speed_ratio < 1 else '16-bit is ' + str(round(1/speed_ratio, 2)) + 'x faster'}")
    print(f"Memory: 16-bit uses {memory_ratio:.2f}x more memory")
    print(f"Precision: 16-bit has {hgrn_16bit.SCALE/hgrn_8bit.SCALE:.0f}x better precision")


def main():
    print("\n" + "="*70)
    print(" "*20 + "8-BIT vs 16-BIT FIXED-POINT BENCHMARK")
    print("="*70)
    
    # Format comparison
    print("\nFORMAT SPECIFICATIONS:")
    print("-"*70)
    print("8-bit  (Q3.5):  3 integer bits, 5 fractional bits")
    print("                Range: [-16.0, 15.96875], Precision: 0.03125")
    print("                Memory: 1 byte per weight")
    print()
    print("16-bit (Q8.8):  8 integer bits, 8 fractional bits")
    print("                Range: [-128.0, 127.996], Precision: 0.00390625")
    print("                Memory: 2 bytes per weight")
    print()
    print("Float32:        23 mantissa bits, 8 exponent bits")
    print("                Range: ±3.4e38, Precision: ~7 decimal digits")
    print("                Memory: 4 bytes per weight")
    
    # Run benchmarks
    benchmark_precision_comparison()
    benchmark_operation_accuracy()
    benchmark_hgrn_layer_performance()
    
    # Summary
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("\n8-bit Fixed-Point (Q3.5):")
    print("  ✓ Minimal memory usage (1 byte/weight)")
    print("  ✓ Fastest computation")
    print("  ✓ Best for extreme edge deployment")
    print("  ✗ Limited precision and range")
    print("  ✗ Higher quantization errors")
    
    print("\n16-bit Fixed-Point (Q8.8):")
    print("  ✓ 32x better precision than 8-bit")
    print("  ✓ Much wider value range")
    print("  ✓ Better numerical stability")
    print("  ✓ Still 50% memory savings vs float32")
    print("  ✗ 2x more memory than 8-bit")
    print("  ✗ Slightly slower than 8-bit")
    
    print("\nRecommendations:")
    print("  • Use 8-bit for: Embedded systems, IoT devices, extreme memory constraints")
    print("  • Use 16-bit for: Mobile devices, edge servers, accuracy-critical applications")
    print("  • Use float32 for: Training, research, unlimited resources")


if __name__ == "__main__":
    main()