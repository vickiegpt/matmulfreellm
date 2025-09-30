# -*- coding: utf-8 -*-

# "HGRN2: Gated Linear RNNs with State Expansion"[https://arxiv.org/abs/2404.07904]

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from transformers.cache_utils import Cache

from mmfreelm.modules import FusedRMSNormSwishGate, ShortConvolution
from mmfreelm.modules.activations import swiglu
from mmfreelm.ops.hgrn.recurrent_fuse import fused_recurrent_hgrn

#from mmfreelm.ops.bitnet import BitLinear_Fuse as BitLinear
from mmfreelm.ops.fusedbitnet import FusedBitLinear as BitLinear


class HGRNBitAttention(nn.Module):

    def __init__(
        self,
        mode: str = 'fused_recurrent',
        hidden_size: int = 1024,
        num_heads: Optional[int] = None,
        expand_ratio: Optional[int] = 1,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        share_conv_kernel: bool = True,
        layernorm_eps: float = 1e-5,
        layer_idx: int = None
    ) -> HGRNAttention:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expand_ratio = expand_ratio
        self.input_dim = int(hidden_size * expand_ratio)
        self.head_dim = self.input_dim // self.num_heads

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.share_conv_kernel = share_conv_kernel

        self.layer_idx = layer_idx

        assert mode in ['fused_recurrent'], f"Not suppoerted mode `{mode}`."
        assert self.hidden_size % num_heads == 0, f"hidden size must be divisible by num_heads of {num_heads}"

        self.i_proj = BitLinear(hidden_size, self.input_dim, bias=False)
        self.f_proj = BitLinear(hidden_size, self.input_dim, bias=False)
        self.g_proj = BitLinear(hidden_size, self.input_dim, bias=False)

        if use_short_conv:
            self.conv_size = conv_size
            if share_conv_kernel:
                self.h_conv1d = ShortConvolution(hidden_size, conv_size, activation='silu')
            else:
                self.q_conv1d = ShortConvolution(self.input_dim, conv_size, activation='silu')
                self.f_conv1d = ShortConvolution(self.input_dim, conv_size, activation='silu')
                self.i_conv1d = ShortConvolution(self.input_dim, conv_size, activation='silu')

        self.g_norm = FusedRMSNormSwishGate(self.input_dim, layernorm_eps)
        self.o_proj = BitLinear(self.input_dim, hidden_size, bias=False)

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, (nn.Linear, BitLinear)):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        lower_bound: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        # launching the triton kernel for just one token will actually be slower
        mode = 'fused_recurrent' if hidden_states.shape[1] == 1 else self.mode

        cache_requested = past_key_values is not None
        cached_state = None
        if cache_requested:
            try:
                cached_state = past_key_values[self.layer_idx]
            except (IndexError, KeyError, TypeError):
                cached_state = None
        if self.use_short_conv:
            conv_state = cached_state[0] if (cached_state is not None and self.share_conv_kernel) else None
            if self.share_conv_kernel:
                # conv state is updated inplace
                hidden_states = self.h_conv1d(hidden_states, attention_mask, conv_state)
                i = self.i_proj(hidden_states)
                f = self.f_proj(hidden_states)
            else:
                conv_state_i = cached_state[2] if cached_state is not None else None
                conv_state_f = cached_state[1] if cached_state is not None else None
                i = self.i_conv1d(self.i_proj(hidden_states), attention_mask, conv_state_i)
                f = self.f_conv1d(self.f_proj(hidden_states), attention_mask, conv_state_f)
        else:
            i = self.i_proj(hidden_states)
            f = self.f_proj(hidden_states)

        f = f.sigmoid()
        # the lower bound for the first layer is zero
        if lower_bound is not None and self.layer_idx > 0:
            f = lower_bound + (1 - lower_bound) * f
        i = swiglu(i, 1 - f)
        # dealing with left-padding
        if attention_mask is not None:
            i = i.mul_(attention_mask.unsqueeze(-1))
        i, f = map(lambda x: rearrange(x, 'b l (h d) -> b h l d', h=self.num_heads), (i, f))

        recurrent_state = cached_state[-1] if (cached_state is not None and use_cache) else None
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_hgrn(
                i,
                f,
                initial_state=recurrent_state,
                output_final_state=cache_requested,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if cache_requested:
            if recurrent_state is None:
                raise RuntimeError("Expected recurrent state when caching is requested but received `None`.")
            if self.use_short_conv:
                if self.share_conv_kernel:
                    if conv_state is None and cached_state is not None:
                        conv_state = cached_state[0]
                    if conv_state is None:
                        raise RuntimeError("Missing convolution state for caching; ensure `past_key_values` are initialized.")
                    state_to_store = (conv_state, recurrent_state)
                else:
                    if conv_state_i is None or conv_state_f is None:
                        if cached_state is None:
                            raise RuntimeError("Missing convolution states for caching; ensure `past_key_values` are initialized.")
                        conv_state_i = cached_state[2]
                        conv_state_f = cached_state[1]
                    state_to_store = (conv_state_i, conv_state_f, recurrent_state)
            else:
                state_to_store = (recurrent_state,)
            past_key_values.update(state_to_store, self.layer_idx, i.shape[2])

        o = self.g_norm(self.g_proj(hidden_states), rearrange(o, 'b h l d -> b l (h d)'))
        o = self.o_proj(o)

        return o, None, past_key_values

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        param = next(self.parameters())
        state = tuple()
        if self.use_short_conv:
            if self.share_conv_kernel:
                state += (param.new_zeros(batch_size, self.hidden_size, self.conv_size),)
            else:
                state += (param.new_zeros(batch_size, self.hidden_size, self.conv_size),
                          param.new_zeros(batch_size, self.hidden_size, self.conv_size),
                          param.new_zeros(batch_size, self.hidden_size, self.conv_size))
        state += (param.new_zeros(batch_size, self.num_heads, self.head_dim),)
        return state

    def state_size(self, **kwargs) -> int:
        state_size = self.hidden_size
        for module in self.children():
            if isinstance(module, ShortConvolution):
                state_size += module.state_size
        return state_size
