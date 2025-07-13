# -*- coding: utf-8 -*-
"""
Fixed-point implementation of HGRN-Bit model for MatMulFreeLLM
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.utils import logging

from mmfreelm.layers.hgrn_bit import HGRNBitAttention
from mmfreelm.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from mmfreelm.models.utils import RecurrentCache
from mmfreelm.modules import FusedCrossEntropyLoss
from mmfreelm.ops.fixed_point_bitnet import FixedPointBitLinear, FixedPointRMSNorm
from mmfreelm.ops.fixed_point import (
    FixedPointConfig, FixedPointSigmoid, FixedPointTanh,
    to_fixed_point, from_fixed_point, fixed_mul
)

logger = logging.get_logger(__name__)


class FixedPointSwish(nn.Module):
    """Fixed-point implementation of Swish activation: x * sigmoid(x)"""
    
    def __init__(self, frac_bits: int = FixedPointConfig.COMPUTE_FRAC_BITS):
        super().__init__()
        self.frac_bits = frac_bits
        self.sigmoid = FixedPointSigmoid(frac_bits)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x_fixed = to_fixed_point(x, self.frac_bits)
            sig_x = self.sigmoid(x_fixed)
            output = fixed_mul(x_fixed, sig_x, self.frac_bits, self.frac_bits, self.frac_bits)
            return from_fixed_point(output, self.frac_bits)
        else:
            sig_x = self.sigmoid(x)
            return fixed_mul(x, sig_x, self.frac_bits, self.frac_bits, self.frac_bits)


class HGRNBitMLPFixed(nn.Module):
    """Fixed-point implementation of HGRN MLP block"""
    
    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'swish',
        use_fixed_point: bool = True
    ) -> None:
        super().__init__()
        
        self.hidden_size = hidden_size
        self.use_fixed_point = use_fixed_point
        
        # Calculate intermediate size
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        
        # Use fixed-point or regular BitLinear layers
        if use_fixed_point:
            self.gate_proj = FixedPointBitLinear(self.hidden_size, self.intermediate_size * 2, bias=False)
            self.down_proj = FixedPointBitLinear(self.intermediate_size, self.hidden_size, bias=False)
            
            # Pre-quantize weights
            self.gate_proj.quantize_weights()
            self.down_proj.quantize_weights()
            
            # Fixed-point activation
            if hidden_act == 'swish':
                self.act_fn = FixedPointSwish()
            else:
                # Fallback to floating-point for unsupported activations
                self.act_fn = ACT2FN[hidden_act]
                warnings.warn(f"Fixed-point version of {hidden_act} not implemented, using floating-point")
        else:
            # Import original BitLinear
            from mmfreelm.ops.fusedbitnet import FusedBitLinear as BitLinear
            self.gate_proj = BitLinear(self.hidden_size, self.intermediate_size * 2, bias=False)
            self.down_proj = BitLinear(self.intermediate_size, self.hidden_size, bias=False)
            self.act_fn = ACT2FN[hidden_act]
    
    def forward(self, x):
        y = self.gate_proj(x)
        gate, y = y.chunk(2, -1)
        
        if self.use_fixed_point and hasattr(self.act_fn, 'forward'):
            # Custom fixed-point activation
            gate_activated = self.act_fn(gate)
            z = fixed_mul(gate_activated, y, 
                         FixedPointConfig.COMPUTE_FRAC_BITS,
                         FixedPointConfig.COMPUTE_FRAC_BITS,
                         FixedPointConfig.COMPUTE_FRAC_BITS)
        else:
            # Standard swish: gate * sigmoid(gate) * y
            z = gate * torch.sigmoid(gate) * y
            
        z = self.down_proj(z)
        return z


class HGRNBitBlockFixed(nn.Module):
    """Fixed-point implementation of HGRN-Bit block"""
    
    def __init__(self, config: HGRNBitConfig, layer_idx: int, use_fixed_point: bool = True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.use_fixed_point = use_fixed_point
        
        # Normalization layers
        if use_fixed_point:
            self.attn_norm = FixedPointRMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
            self.mlp_norm = FixedPointRMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        else:
            from mmfreelm.modules import RMSNorm
            self.attn_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
            self.mlp_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
            
        # Attention layer (keeping original for now, can be converted later)
        self.attn = HGRNBitAttention(
            mode=config.attn_mode,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            expand_ratio=config.expand_ratio,
            use_short_conv=config.use_short_conv,
            conv_size=config.conv_size,
            share_conv_kernel=config.share_conv_kernel,
            layernorm_eps=config.rms_norm_eps,
            layer_idx=layer_idx
        )
        
        # MLP layer
        self.mlp = HGRNBitMLPFixed(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            use_fixed_point=use_fixed_point
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        lower_bound: Optional[torch.Tensor] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            lower_bound=lower_bound
        )
        
        # Handle residual connection with normalization
        if self.use_fixed_point:
            # For fixed-point, we need to handle residual carefully
            hidden_states = self.mlp_norm(hidden_states)
            hidden_states = hidden_states + residual
        else:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
            
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states, attentions, past_key_values)
        
        return outputs


class HGRNBitPreTrainedModelFixed(PreTrainedModel):
    """Fixed-point version of HGRN pretrained model base class"""
    
    config_class = HGRNBitConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ['HGRNBitBlockFixed']
    
    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        
    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = True,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d, FixedPointBitLinear)):
            # Initialize weights with normal distribution
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
                
            # Pre-quantize weights for fixed-point layers
            if isinstance(module, FixedPointBitLinear):
                module.quantize_weights()
                
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
                
        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme
            for name, p in module.named_parameters():
                if name in ["o_proj.weight", "down_proj.weight"]:
                    # Scale residual layer weights
                    nn.init.normal_(
                        p, 
                        mean=0.0, 
                        std=self.config.initializer_range / math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)
                    )


class HGRNBitModelFixed(HGRNBitPreTrainedModelFixed):
    """Fixed-point HGRN-Bit model"""
    
    def __init__(self, config: HGRNBitConfig, use_fixed_point: bool = True):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.use_fixed_point = use_fixed_point
        
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            HGRNBitBlockFixed(config, layer_idx=idx, use_fixed_point=use_fixed_point)
            for idx in range(config.num_hidden_layers)
        ])
        
        if use_fixed_point:
            self.norm = FixedPointRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            from mmfreelm.modules import RMSNorm
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            
        self.gradient_checkpointing = False
        
        # Initialize weights
        self.post_init()
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
            
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
            
        hidden_states = inputs_embeds
        
        # Process through layers
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                    
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    past_key_value,
                    use_cache,
                    output_attentions
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions += (layer_outputs[1],)
                
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        if not return_dict:
            return tuple(v for v in [hidden_states, None, all_hidden_states, all_attentions] if v is not None)
            
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class HGRNBitForCausalLMFixed(HGRNBitPreTrainedModelFixed, GenerationMixin):
    """Fixed-point HGRN-Bit model for causal language modeling"""
    
    def __init__(self, config: HGRNBitConfig, use_fixed_point: bool = True):
        super().__init__(config)
        self.model = HGRNBitModelFixed(config, use_fixed_point=use_fixed_point)
        self.vocab_size = config.vocab_size
        
        # Output projection can use fixed-point BitLinear
        if use_fixed_point:
            self.lm_head = FixedPointBitLinear(config.hidden_size, config.vocab_size, bias=False)
            self.lm_head.quantize_weights()
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            
        # Initialize weights
        self.post_init()
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Model forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Compute loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = loss_fct(shift_logits, shift_labels)
            
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
            
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )