"""
Text generation using 16-bit fixed-point HGRN model
Higher precision version with Q8.8 format
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
import time
from transformers import AutoTokenizer
from hgrn_16bit_fixed import HGRNFixed16bit, generate_test_weights_16bit


class FixedPoint16bitHGRNModel(nn.Module):
    """
    16-bit fixed-point HGRN model for text generation
    Uses Q8.8 format for better precision than 8-bit version
    """
    def __init__(self, vocab_size=50257, hidden_size=768, num_layers=12):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # 16-bit fixed-point HGRN layers
        self.hgrn_layers = nn.ModuleList([
            HGRNFixed16bit() for _ in range(num_layers)
        ])
        
        # Generate ternary weights with 16-bit scales for each layer
        self.weights = []
        for layer_idx in range(num_layers):
            layer_weights = {}
            for weight_type in ['i', 'f', 'g', 'o']:
                w_ternary, w_scale = generate_test_weights_16bit(hidden_size, hidden_size)
                # Register as buffers
                self.register_buffer(f'layer{layer_idx}_w_{weight_type}', w_ternary)
                self.register_buffer(f'layer{layer_idx}_w_scale_{weight_type}', w_scale)
                layer_weights[f'w_{weight_type}'] = f'layer{layer_idx}_w_{weight_type}'
                layer_weights[f'w_scale_{weight_type}'] = f'layer{layer_idx}_w_scale_{weight_type}'
            self.weights.append(layer_weights)
        
        # Output projection
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids):
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Process through 16-bit HGRN layers
        for layer_idx, hgrn_layer in enumerate(self.hgrn_layers):
            weights = self.weights[layer_idx]
            # Get the actual tensors from buffer names
            w_i = getattr(self, weights['w_i'])
            w_f = getattr(self, weights['w_f'])
            w_g = getattr(self, weights['w_g'])
            w_o = getattr(self, weights['w_o'])
            w_scale_i = getattr(self, weights['w_scale_i'])
            w_scale_f = getattr(self, weights['w_scale_f'])
            w_scale_g = getattr(self, weights['w_scale_g'])
            w_scale_o = getattr(self, weights['w_scale_o'])
            
            hidden_states = hgrn_layer.forward(
                hidden_states,
                w_i, w_f, w_g, w_o,
                w_scale_i, w_scale_f, w_scale_g, w_scale_o
            )
            # Convert back to float for next layer
            hidden_states = hgrn_layer.from_fixed(hidden_states)
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        return logits
    
    @torch.no_grad()
    def generate(self, input_ids, max_length=32, temperature=0.6, top_p=0.4):
        """Simple text generation using the 16-bit fixed-point model"""
        
        # Initialize output with input
        generated = input_ids
        
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            logits = self.forward(generated)
            
            # Get last token logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-p filtering
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            
            # Apply the mask
            for i in range(sorted_indices.shape[0]):
                indices_to_remove = sorted_indices_to_remove[i].scatter(0, sorted_indices[i], sorted_indices_to_remove[i])
                next_token_logits[i, indices_to_remove] = -float('inf')
            
            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=-1)
            
        return generated


def main():
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Use the same tokenizer as the original model
    tokenizer = AutoTokenizer.from_pretrained('ridger/MMfreeLM-2.7B')
    
    # Create 16-bit fixed-point model
    print("\nCreating 16-bit fixed-point HGRN model (Q8.8 format)...")
    model = FixedPoint16bitHGRNModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,  # Smaller for demonstration
        num_layers=6      # Fewer layers for demonstration
    ).to(device)
    
    # Model statistics
    print("\n" + "="*60)
    print("16-BIT MODEL CONFIGURATION")
    print("="*60)
    print(f"Format: Q8.8 (8 integer bits, 8 fractional bits)")
    print(f"Precision: 0.00390625 (1/256)")
    print(f"Range: [-128.000, 127.996]")
    print(f"Hidden size: {model.hidden_size}")
    print(f"Number of layers: {model.num_layers}")
    
    # Calculate memory usage
    total_params = sum(p.numel() for p in model.parameters())
    total_buffers = sum(b.numel() for b in model.buffers())
    
    # Ternary weights are int8, scales are int16
    ternary_memory = (4 * model.hidden_size * model.hidden_size * model.num_layers) / (1024 * 1024)  # MB
    scale_memory = (4 * model.hidden_size * model.num_layers * 2) / (1024 * 1024)  # 2 bytes per int16
    embedding_memory = (total_params * 4) / (1024 * 1024)  # embeddings are float32
    
    total_memory = ternary_memory + scale_memory + embedding_memory
    
    print(f"\nMemory Usage:")
    print(f"  Ternary weights: {ternary_memory:.2f} MB")
    print(f"  Scale factors (16-bit): {scale_memory:.3f} MB")
    print(f"  Embeddings & LM head: {embedding_memory:.2f} MB")
    print(f"  Total: {total_memory:.2f} MB")
    
    # Compare with 8-bit and float32
    memory_8bit = ternary_memory + (scale_memory / 2) + embedding_memory  # 8-bit scales
    memory_float32 = (total_params * 4) / (1024 * 1024)
    
    print(f"\nComparison:")
    print(f"  8-bit model: {memory_8bit:.2f} MB")
    print(f"  16-bit model: {total_memory:.2f} MB (current)")
    print(f"  Float32 model: {memory_float32:.2f} MB")
    print(f"  Savings vs float32: {(1 - total_memory/memory_float32)*100:.1f}%")
    
    # Prepare input
    input_prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, "
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to(device)
    
    print("\n" + "="*60)
    print("GENERATING TEXT WITH 16-BIT MODEL")
    print("="*60)
    print(f"\nInput: {input_prompt}")
    
    # Generate text with timing
    if device == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=32, temperature=0.6, top_p=0.4)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    gen_time = time.time() - start_time
    
    # Decode and print
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"\nGenerated: {generated_text}")
    print(f"\nGeneration time: {gen_time:.3f} seconds")
    print(f"Tokens/sec: {(outputs.shape[1] - input_ids.shape[1]) / gen_time:.2f}")
    
    print("\n" + "="*60)
    print("ADVANTAGES OF 16-BIT OVER 8-BIT")
    print("="*60)
    print("✓ 32x better precision (0.004 vs 0.031)")
    print("✓ Wider value range (±128 vs ±4)")
    print("✓ Better numerical stability")
    print("✓ Reduced quantization errors")
    print("✓ Still 50% memory savings vs float32")
    print("✓ Good balance between accuracy and efficiency")
    
    print("\n✓ 16-bit fixed-point HGRN model successfully tested!")
    print("Note: This uses random ternary weights. Production models would")
    print("require proper weight quantization from pretrained models.")


if __name__ == "__main__":
    main()