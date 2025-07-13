"""
Demo script showing how to use the fixed-point implementation of MatMulFreeLLM
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from mmfreelm.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from mmfreelm.models.hgrn_bit.modeling_hgrn_bit_fixed import HGRNBitForCausalLMFixed


def create_small_model():
    """Create a small model for demonstration"""
    config = HGRNBitConfig(
        vocab_size=50257,  # GPT-2 tokenizer size
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_heads=4,
        max_position_embeddings=512,
        hidden_act="swish",
        rms_norm_eps=1e-5,
        pad_token_id=0,
    )
    
    # Create model with fixed-point arithmetic
    model = HGRNBitForCausalLMFixed(config, use_fixed_point=True)
    return model


def demo_inference():
    """Demonstrate inference with fixed-point model"""
    print("Creating fixed-point model...")
    model = create_small_model()
    model.eval()
    
    # Use GPT-2 tokenizer for simplicity
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Example input
    text = "The future of AI is"
    print(f"\nInput text: {text}")
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    print(f"Input tokens: {input_ids}")
    
    # Generate with fixed-point model
    with torch.no_grad():
        # Get logits for next token
        outputs = model(input_ids)
        logits = outputs.logits
        
        # Get next token prediction
        next_token_logits = logits[0, -1, :]
        next_token_id = torch.argmax(next_token_logits).item()
        next_token = tokenizer.decode(next_token_id)
        
        print(f"\nNext token prediction: '{next_token}'")
        print(f"Top 5 predictions:")
        
        top5_values, top5_indices = torch.topk(next_token_logits, 5)
        for i, (value, idx) in enumerate(zip(top5_values, top5_indices)):
            token = tokenizer.decode(idx.item())
            print(f"  {i+1}. '{token}' (score: {value.item():.3f})")


def compare_fixed_vs_float():
    """Compare fixed-point and floating-point implementations"""
    print("\n" + "="*60)
    print("Comparing Fixed-Point vs Floating-Point")
    print("="*60)
    
    # Create config
    config = HGRNBitConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_heads=2,
        hidden_act="swish",
    )
    
    # Create both models
    model_fixed = HGRNBitForCausalLMFixed(config, use_fixed_point=True)
    model_float = HGRNBitForCausalLMFixed(config, use_fixed_point=False)
    
    # Copy weights from float to fixed model
    model_fixed.load_state_dict(model_float.state_dict())
    
    # Set to eval mode
    model_fixed.eval()
    model_float.eval()
    
    # Create random input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    with torch.no_grad():
        output_fixed = model_fixed(input_ids)
        output_float = model_float(input_ids)
        
        # Compare outputs
        diff = torch.abs(output_fixed.logits - output_float.logits)
        
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {output_fixed.logits.shape}")
        print(f"\nOutput difference statistics:")
        print(f"  Max difference: {diff.max().item():.6f}")
        print(f"  Mean difference: {diff.mean().item():.6f}")
        print(f"  Min difference: {diff.min().item():.6f}")
        
        # Check relative error
        rel_error = diff / (torch.abs(output_float.logits) + 1e-8)
        print(f"\nRelative error statistics:")
        print(f"  Max relative error: {rel_error.max().item():.2%}")
        print(f"  Mean relative error: {rel_error.mean().item():.2%}")


def demonstrate_memory_efficiency():
    """Show memory efficiency of fixed-point implementation"""
    print("\n" + "="*60)
    print("Memory Efficiency Analysis")
    print("="*60)
    
    # Configuration for comparison
    config = HGRNBitConfig(
        vocab_size=50257,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_heads=8,
    )
    
    # Calculate memory usage
    # Floating-point model
    float_params = 0
    float_params += config.vocab_size * config.hidden_size  # embeddings
    float_params += config.num_hidden_layers * (
        4 * config.hidden_size * config.hidden_size +  # attention (q,k,v,o)
        2 * config.hidden_size * config.intermediate_size  # MLP
    )
    float_params += config.vocab_size * config.hidden_size  # lm_head
    
    float_memory_mb = (float_params * 4) / (1024 * 1024)  # 4 bytes per float32
    
    # Fixed-point model
    # Ternary weights: 2 bits per weight (can represent -1, 0, 1)
    # Scales: one 32-bit scale per layer
    ternary_bits = float_params * 2  # 2 bits per weight
    scale_params = config.num_hidden_layers * 10  # approximate scales per layer
    
    fixed_memory_mb = (ternary_bits / 8 + scale_params * 4) / (1024 * 1024)
    
    print(f"Model configuration:")
    print(f"  Vocabulary size: {config.vocab_size}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Number of layers: {config.num_hidden_layers}")
    print(f"  Total parameters: {float_params:,}")
    
    print(f"\nMemory usage:")
    print(f"  Floating-point (FP32): {float_memory_mb:.2f} MB")
    print(f"  Fixed-point (ternary): {fixed_memory_mb:.2f} MB")
    print(f"  Compression ratio: {float_memory_mb / fixed_memory_mb:.1f}x")
    
    print(f"\nAdditional benefits of fixed-point:")
    print(f"  - Deterministic computation (bit-exact reproducibility)")
    print(f"  - Lower power consumption (simpler ALU operations)")
    print(f"  - Faster inference on specialized hardware")
    print(f"  - No matrix multiplications needed (ternary weights)")


if __name__ == "__main__":
    print("MatMulFreeLLM Fixed-Point Implementation Demo")
    print("=" * 60)
    
    # Run demonstrations
    demo_inference()
    compare_fixed_vs_float()
    demonstrate_memory_efficiency()
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)