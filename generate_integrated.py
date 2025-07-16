import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
from typing import Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from hgrn_8bit_fixed import HGRNFixed8bit
from mmfreelm.layers.hgrn_bit import HGRNBitAttention


class FixedPointHGRNAttention(nn.Module):
    """
    Drop-in replacement for HGRNBitAttention that uses fixed-point arithmetic
    """
    def __init__(self, original_attention: HGRNBitAttention):
        super().__init__()
        self.hidden_size = original_attention.hidden_size
        self.num_heads = original_attention.num_heads
        self.expand_ratio = original_attention.expand_ratio
        self.input_dim = original_attention.input_dim
        self.layer_idx = original_attention.layer_idx
        
        # Initialize fixed-point HGRN
        self.hgrn_fixed = HGRNFixed8bit()
        
        # Extract and quantize weights from original model
        self.quantize_weights(original_attention)
        
        # Keep other components
        self.g_norm = original_attention.g_norm
        self.use_short_conv = original_attention.use_short_conv
        if self.use_short_conv:
            self.h_conv1d = original_attention.h_conv1d
    
    def quantize_weights(self, original_attention):
        """Convert floating-point weights to ternary format"""
        # Extract weight matrices
        w_i = original_attention.i_proj.weight.data.T  # Transpose for correct dimension
        w_f = original_attention.f_proj.weight.data.T
        w_g = original_attention.g_proj.weight.data.T
        w_o = original_attention.o_proj.weight.data
        
        # Quantize to ternary
        self.w_i_ternary = self.to_ternary(w_i)
        self.w_f_ternary = self.to_ternary(w_f)
        self.w_g_ternary = self.to_ternary(w_g)
        self.w_o_ternary = self.to_ternary(w_o)
        
        # Compute scale factors
        self.w_scale_i = self.compute_scale(w_i)
        self.w_scale_f = self.compute_scale(w_f)
        self.w_scale_g = self.compute_scale(w_g)
        self.w_scale_o = self.compute_scale(w_o)
    
    def to_ternary(self, w):
        """Convert weight matrix to ternary {-1, 0, 1}"""
        scale = w.abs().mean()
        w_normalized = w / (scale + 1e-5)
        w_ternary = torch.sign(w_normalized)
        w_ternary[w_normalized.abs() < 0.5] = 0
        return w_ternary.to(torch.int8)
    
    def compute_scale(self, w):
        """Compute per-channel scale factors"""
        scale = w.abs().mean(dim=0)
        return self.hgrn_fixed.to_fixed(scale)
    
    def forward(self, hidden_states, attention_mask=None, past_key_values=None, 
                use_cache=False, output_attentions=False, lower_bound=None, **kwargs):
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Apply short convolution if needed
        if self.use_short_conv:
            conv_state = past_key_values[self.layer_idx][0] if use_cache and past_key_values else None
            hidden_states = self.h_conv1d(hidden_states, attention_mask, conv_state)
        
        # Run fixed-point HGRN forward pass
        output_fixed = self.hgrn_fixed.forward(
            hidden_states,
            self.w_i_ternary, self.w_f_ternary, self.w_g_ternary, self.w_o_ternary,
            self.w_scale_i, self.w_scale_f, self.w_scale_g, self.w_scale_o
        )
        
        # Convert back to float
        output = self.hgrn_fixed.from_fixed(output_fixed)
        
        # Update cache if needed
        if use_cache and past_key_values is not None:
            if self.use_short_conv:
                last_state = (conv_state, torch.zeros(batch_size, self.num_heads, 1, self.input_dim // self.num_heads))
            else:
                last_state = (torch.zeros(batch_size, self.num_heads, 1, self.input_dim // self.num_heads),)
            past_key_values.update(last_state, self.layer_idx, seq_len)
        
        return output, None, past_key_values
    
    def init_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        """Initialize state for caching during generation"""
        # Use one of the weight tensors to get device and dtype
        device = self.w_i_ternary.device
        dtype = torch.float16  # Use half precision like the model
        
        state = tuple()
        if self.use_short_conv:
            conv_size = getattr(self.h_conv1d, 'kernel_size', 4)
            if isinstance(conv_size, tuple):
                conv_size = conv_size[0]
            state += (torch.zeros(batch_size, self.hidden_size, conv_size, 
                                device=device, dtype=dtype),)
        # Add recurrent state
        state += (torch.zeros(batch_size, self.num_heads, 1, 
                            self.input_dim // self.num_heads, 
                            device=device, dtype=dtype),)
        return state


def replace_with_fixed_point_hgrn(model):
    """Replace all HGRNBitAttention layers with fixed-point implementation"""
    for name, module in model.named_modules():
        if isinstance(module, HGRNBitAttention):
            # Get parent module and attribute name
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            
            # Replace with fixed-point version
            fixed_point_module = FixedPointHGRNAttention(module)
            setattr(parent, parts[-1], fixed_point_module)
            print(f"Replaced {name} with fixed-point implementation")
    
    return model


def main():
    # Load model and tokenizer
    name = 'ridger/MMfreeLM-2.7B'
    print(f"Loading model from {name}...")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).cuda().half()
    
    # Replace HGRN layers with fixed-point implementation
    print("\nReplacing HGRN layers with fixed-point implementation...")
    model = replace_with_fixed_point_hgrn(model)
    
    # Test generation
    input_prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, "
    print(f"\nInput: {input_prompt}")
    
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.cuda()
    
    print("\nGenerating with fixed-point HGRN layers...")
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=32, do_sample=True, top_p=0.4, temperature=0.6)
    
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"\nGenerated: {generated_text}")
    
    print("\n✓ Successfully integrated fixed-point HGRN with MMfreeLM model!")
    print("The model now uses ternary_matmul (ternary_matmul) operations for all HGRN layers.")


if __name__ == "__main__":
    main()