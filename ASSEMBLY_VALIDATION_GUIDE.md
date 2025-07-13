# HGRN Assembly Accelerator Validation Guide

## Overview

This guide explains how to validate your assembly implementation of the HGRN (Hierarchically Gated Recurrent Network) algorithm against the PyTorch MatMulFreeLLM model. We provide 8-bit fixed-point test vectors and validation tools.

## Algorithm Summary

The HGRN algorithm implemented in `generate_token.S` should perform:

```
1. Input projections (using ternary weights):
   g = WG @ X
   f = WF @ X  
   c = WC @ X

2. Gate activations:
   f_sig = sigmoid(f)
   c_sig = sigmoid(c)

3. Hidden state update:
   h = f_sig * h_prev + c_sig * (1 - f_sig)

4. Output projection:
   O = WO @ (g * h)
```

## Fixed-Point Format

All computations use **Q3.5 fixed-point** format:
- 8 bits total: 3 integer bits, 5 fractional bits
- Range: [-4, 3.96875]
- Precision: 0.03125 (1/32)

## Test Vectors

We provide test vectors in multiple formats:

### 1. C Header Format (`hgrn_test_vectors.h`)

```c
#define HIDDEN_SIZE 16
#define FRAC_BITS 5

// Input vectors (Q3.5 format)
const int8_t input_x[] = {8, -2, 10, 24, -4, -4, 25, 12, ...};
const int8_t input_h_init[] = {-3, 1, -3, -5, 5, -1, 0, -5, ...};

// Ternary weight matrices {-1, 0, 1}
const int8_t weight_g[] = {...};  // 16x16 matrix
const int8_t weight_f[] = {...};  // 16x16 matrix
const int8_t weight_c[] = {...};  // 16x16 matrix
const int8_t weight_o[] = {...};  // 16x16 matrix

// Expected output
const int8_t expected_output[] = {-33, 20, 37, 42, -8, 20, ...};
```

### 2. Assembly Format (`hgrn_test.S`)

Pre-formatted assembly code with test vectors and validation routine.

### 3. NumPy Format (`hgrn_assembly_test.npz`)

For Python-based validation and debugging.

## Validation Steps

### Step 1: Implement Fixed-Point Operations

Your assembly code needs these operations:

```assembly
# Ternary matrix multiplication (no actual multiplies!)
# For each output element:
#   - Add input values where weight = 1
#   - Subtract input values where weight = -1
#   - Skip where weight = 0

# Fixed-point sigmoid approximation
# if x < -80 (-2.5 in Q3.5): return 0
# if x > 80 (2.5 in Q3.5): return 32 (1.0 in Q3.5)
# else: return (x/5 + 16) (0.2*x + 0.5 in Q3.5)

# Fixed-point multiplication
# result = (a * b) >> 5  (arithmetic shift right by FRAC_BITS)
```

### Step 2: Expected Assembly Interface

Your `generate_token` function should expect:

```assembly
# Inputs:
# a0: pointer to X vector (16 bytes)
# a1: pointer to oH vector (16 bytes) 
# a2: pointer to B vector (16 bytes, usually zeros)
# a3: pointer to WG matrix (256 bytes, row-major)
# a4: pointer to WF matrix (256 bytes, row-major)
# a5: pointer to WC matrix (256 bytes, row-major)
# a6: pointer to WO matrix (256 bytes, row-major)
# a7: pointer to O output vector (16 bytes)
```

### Step 3: Run Validation

1. Load test vectors
2. Call your `generate_token` function
3. Compare output with expected values
4. Check intermediate values if output doesn't match

## Debugging Tips

### Common Issues

1. **Overflow in accumulation**: Use 16-bit or 32-bit accumulators for matrix operations
2. **Incorrect sigmoid**: Ensure thresholds match Q3.5 format
3. **Wrong matrix layout**: Weights are in row-major order
4. **Scale factors**: Ternary weights may need per-channel scaling

### Intermediate Value Checks

Expected intermediate values for debugging:

```
After projections:
  g: [18, 0, -128, 27, 35, -34, 86, 31, ...]
  f: [10, 19, -49, -69, 7, -22, -86, -31, ...]
  c: [...] 

After sigmoid:
  f_sig: [32, 32, 12, 5, 32, 28, 2, 20, ...]
  c_sig: [...]

After hidden update:
  h_new: [-3, 1, -1, 26, 5, 1, 1, 0, ...]

Final output:
  O: [-33, 20, 37, 42, -8, 20, -33, -4, ...]
```

## Python Validation Script

Use `validate_hgrn_fixed_point.py` to:
- Generate new test vectors
- Compare floating-point vs fixed-point
- Visualize error distributions
- Export vectors in different formats

```bash
python hgrn_assembly_format.py  # Generate test vectors
python validate_hgrn_fixed_point.py  # Run validation
```

## Example Assembly Implementation

Here's a simplified example of ternary matrix-vector multiplication:

```assembly
# Compute y = W @ x where W is ternary
# a0: input vector x (16 elements)
# a1: ternary matrix W (16x16, row-major)
# a2: output vector y (16 elements)
# a3: matrix row counter

ternary_matvec:
    li t0, 16          # output size
outer_loop:
    li t1, 0           # accumulator
    li t2, 16          # input size
    mv t3, a0          # reset input pointer
    
inner_loop:
    lb t4, 0(a1)       # load weight
    lb t5, 0(t3)       # load input
    
    # Ternary multiplication
    beqz t4, skip      # if weight == 0, skip
    li t6, 1
    beq t4, t6, add_val # if weight == 1, add
    sub t1, t1, t5     # else weight == -1, subtract
    j next
add_val:
    add t1, t1, t5
skip:
next:
    addi a1, a1, 1     # next weight
    addi t3, t3, 1     # next input
    addi t2, t2, -1
    bnez t2, inner_loop
    
    # Store result (with saturation)
    # ... saturation logic ...
    sb t1, 0(a2)
    addi a2, a2, 1
    addi t0, t0, -1
    bnez t0, outer_loop
    ret
```

## Success Criteria

Your implementation is correct if:
1. Output vector O matches expected values exactly
2. No overflow or underflow occurs
3. Performance is better than floating-point implementation

## Contact

For questions about the test vectors or algorithm, refer to:
- Original paper: "Scalable MatMul-free Language Modeling"
- Repository: https://github.com/ridgerchu/matmulfreellm