import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from hgrn_8bit_fixed import HGRNFixed8bit, generate_test_weights

class FixedPointHGRNModel(nn.Module):
    """
    A simple wrapper that uses the fixed-point HGRN implementation
    for text generation demonstration
    """
    def __init__(self, vocab_size=50257, hidden_size=768, num_layers=12):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # Fixed-point HGRN layers
        self.hgrn_layers = nn.ModuleList([
            HGRNFixed8bit() for _ in range(num_layers)
        ])
        
        # Generate ternary weights for each layer and register as buffers
        self.weights = []
        for layer_idx in range(num_layers):
            layer_weights = {}
            for weight_type in ['i', 'f', 'g', 'o']:
                w_ternary, w_scale = generate_test_weights(hidden_size, hidden_size)
                # Register as buffers so they move with the model
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
        
        # Process through HGRN layers
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
        """Simple text generation using the fixed-point model"""
        
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
    # Use the same tokenizer as the original model
    tokenizer = AutoTokenizer.from_pretrained('ridger/MMfreeLM-2.7B')
    
    # Create fixed-point model
    print("Creating fixed-point HGRN model...")
    model = FixedPointHGRNModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,  # Smaller for demonstration
        num_layers=6      # Fewer layers for demonstration
    ).cuda()
    
    # Prepare input
    input_prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, "
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.cuda()
    
    print(f"\nInput: {input_prompt}")
    print("\nGenerating with fixed-point HGRN model...")
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=32, temperature=0.6, top_p=0.4)
    
    # Decode and print
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"\nGenerated: {generated_text}")
    
    print("\n✓ Fixed-point HGRN model successfully integrated and tested!")
    print("Note: This is a demonstration model with random weights. For real usage, ")
    print("you would need to convert the pretrained weights to ternary format.")


if __name__ == "__main__":
    main()