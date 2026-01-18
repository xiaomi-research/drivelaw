import math
from typing import Any, Dict, Optional, Tuple


import torch
import torch.nn as nn

from diffusers.models.attention import FeedForward
from diffusers.models.normalization import AdaLayerNormSingle, RMSNorm
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import Attention

from timm.models.layers import Mlp
from .blocks.encoder import (
    StateAttentionEncoder,
    SwiGLUFFN,
)
from diffusers.utils import USE_PEFT_BACKEND, deprecate, is_torch_version
from diffusers.models.attention_dispatch import dispatch_attention_fn

class LTXVideoAttentionProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`LTXVideoAttentionProcessor2_0` is deprecated and this will be removed in a future version. Please use `LTXVideoAttnProcessor`"
        deprecate("LTXVideoAttentionProcessor2_0", "1.0.0", deprecation_message)

        return LTXVideoAttnProcessor(*args, **kwargs)


class LTXVideoAttnProcessor:
    r"""
    Processor for implementing attention (SDPA is used by default if you're using PyTorch 2.0). This is used in the LTX
    model. It applies a normalization layer and rotary embedding on the query and key vector.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if is_torch_version("<", "2.0"):
            raise ValueError(
                "LTX attention processors require a minimum PyTorch version of 2.0. Please upgrade your PyTorch installation."
            )

    def __call__(
        self,
        attn: "LTXAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            #parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

class ActionTrajectoryScorer(nn.Module):
    def __init__(self,
                 action_dim=3,
                 hidden=512,         # scorer 内部维度 D
                 heads=8,
                 dim_head=64,
                 ctx_dim=128        # 你的视频上下文 token 维度（inner_dim）
                 ):
        super().__init__()
        D = hidden

        self.act_in = nn.Linear(action_dim, D)
        self.rope   = ActionRotaryPosEmbed(dim=D, base_seq_length=128)

        self.ctx_proj = nn.Linear(ctx_dim, D)

        self.q_norm = nn.LayerNorm(D)
        self.attn   = Attention(
            query_dim=D, heads=heads, kv_heads=heads, dim_head=dim_head,
            cross_attention_dim=D, bias=True, out_bias=True, qk_norm="rms_norm_across_heads",processor=LTXVideoAttentionProcessor2_0()
        )

        self.ffn  = SwiGLUFFN(D, bias=True)
        self.out  = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1))

    def forward(self,
                actions_bk: torch.Tensor,
                ctx_tokens: torch.Tensor,
                ctx_mask: torch.Tensor = None 
                ):
        B, K, L, Ca = actions_bk.shape
        D = self.act_in.out_features

        x = self.act_in(actions_bk.view(B*K, L, Ca))   # (B*K, L, D)
        cos, sin = self.rope(x, L)                     # (1, L, D)
        x = apply_rotary_emb(x, (cos, sin))            # (B*K, L, D)
        x = self.q_norm(x)

        kv = self.ctx_proj(ctx_tokens.detach())        # (B, S_ctx, D)
        #kv = kv.repeat_interleave(K, dim=0)            # (B*K, S_ctx, D)


        
        x = self.attn(x, encoder_hidden_states=kv, attention_mask=None)
        x = self.ffn(x)                                  # (B*K, L, D)

        x = x.mean(dim=1)                                # (B*K, D)
        logits = self.out(x).view(B, K)                  # (B, K)

        return logits


def apply_rotary_emb(x, freqs):
    cos, sin = freqs
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)  # [B, S, C // 2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out

class ActionRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        dim: int,
        base_seq_length: int = 57,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.base_seq_length = base_seq_length
        self.theta = theta

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Always compute rope in fp32
        grid = torch.arange(seq_length, dtype=torch.float32, device=hidden_states.device).unsqueeze(0)

        grid = grid / self.base_seq_length

        grid = grid.unsqueeze(-1)

        start = 1.0
        end = self.theta
        freqs = self.theta ** torch.linspace(
            math.log(start, self.theta),
            math.log(end, self.theta),
            self.dim // 2,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        freqs = freqs * math.pi / 2.0
        freqs = freqs * (grid * 2 - 1)

        cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

        if self.dim % 2 != 0:
            cos_padding = torch.ones_like(cos_freqs[:, :, : self.dim % 2])
            sin_padding = torch.zeros_like(sin_freqs[:, :, : self.dim % 2])
            cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
            sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)

        return cos_freqs, sin_freqs




@maybe_allow_in_graph
class ActionTransformerBlock(nn.Module):
    r"""
    Modified from Transformer block used in [LTX](https://huggingface.co/Lightricks/LTX-Video).

    Args:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        qk_norm (`str`, defaults to `"rms_norm"`):
            The normalization layer to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
    """

    def __init__(
        self,
        attention_class,
        attention_args,
        dim: int = 512,
        num_attention_heads: int = 16,
        attention_head_dim: int = 32,
        cross_attention_dim: int = 2048,
        qk_norm: str = "rms_norm_across_heads",
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        attn3_cross_attention_dim = 2048,
        num_latent_downsample_block = 0,
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn1 = attention_class(
            **(attention_args[0]),
        )

        self.norm2 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn2 = attention_class(
            **(attention_args[1]),
        )

        #self.ff = FeedForward(dim, activation_fn=activation_fn)
        self.ff = SwiGLUFFN(dim, bias=True)

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

        self.num_latent_downsample_block = num_latent_downsample_block
        if self.num_latent_downsample_block > 0:
            self.latent_downsample_block = nn.ModuleList()
            for _i in range(self.num_latent_downsample_block):
                self.latent_downsample_block.append(
                    downsampling_block()
                )


    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Tq, D = hidden_states.shape

        batch_size = hidden_states.size(0)
        norm_hidden_states = self.norm1(hidden_states)

        num_ada_params = self.scale_shift_table.shape[0]
        if temb.shape[-1] == D * num_ada_params:
            ada_values = self.scale_shift_table[None, None] + temb.view(B, Tq, num_ada_params, D)
        elif temb.shape[-1] == D:
            ada_core = temb.unsqueeze(2).expand(B, Tq, num_ada_params, D)
            ada_values = self.scale_shift_table[None, None] + ada_core
        else:
            raise ValueError(f"temb last dim {temb.shape[-1]} not in {{D, 6*D}} with D={D}")

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        
        attn_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            image_rotary_emb=rotary_emb,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa

        attn_hidden_states = self.attn2(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=None,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = hidden_states + attn_hidden_states

        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp
        

        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp
        

        return hidden_states



def add_action_expert(
    self,
    num_layers: int = 28,
    inner_dim: int = 2048,
    activation_fn: str = "gelu",
    norm_eps: float = 1e-6,
    action_in_channels: int = 3,
    action_out_channels: int = 3,
    action_num_attention_heads: int = 16,
    action_attention_head_dim: int = 32,
    action_rope_dim: int = None,
    action_final_embeddings: bool = True,
    learnable_action_state: bool = False,
    norm_elementwise_affine: bool = False,
    attention_bias: bool = True,
    attention_out_bias: bool = True,
    qk_norm: str = "rms_norm_across_heads",
    attention_class = None,
    attention_processor = None,
    **kwargs,
):

    if action_out_channels is None:
        action_out_channels = action_in_channels

    self.action_inner_dim = action_num_attention_heads * action_attention_head_dim

    self.learnable_action_state = learnable_action_state
    if self.learnable_action_state:
        self.action_state = nn.Parameter(torch.randn(1, 1, action_in_channels))

    self.action_proj_in = nn.Linear(action_in_channels, self.action_inner_dim)
    self.action_scale_shift_table = nn.Parameter(torch.randn(2, self.action_inner_dim) / self.action_inner_dim**0.5)
    self.action_time_embed = AdaLayerNormSingle(self.action_inner_dim, use_additional_conditions=False)

    self.action_ego_status_encoder = Mlp(
        in_features=8,
        hidden_features=self.action_inner_dim*2,
        out_features=self.action_inner_dim,
        norm_layer=nn.LayerNorm
    )
    # self.action_his_traj_encoder = Mlp(in_features=12,hidden_features=self.action_inner_dim*2,out_features=inner_dim,norm_layer=nn.LayerNorm)
    self.action_state_film = nn.Sequential(
        nn.Linear(self.action_inner_dim, self.action_inner_dim * 2),
        nn.SiLU(),
        nn.Linear(self.action_inner_dim * 2, self.action_inner_dim * 6),
    )
    if action_rope_dim is None:
        action_rope_dim = self.action_inner_dim
    # set to a fixed value currently, should adjust according to the action length
    self.action_rope = ActionRotaryPosEmbed(
        dim=action_rope_dim,
        base_seq_length=8,
        theta=10000.0,
    )

    attention_args = []
    attention_args.append(dict(
        query_dim=self.action_inner_dim,
        heads=action_num_attention_heads,
        kv_heads=action_num_attention_heads,
        dim_head=action_attention_head_dim,
        bias=attention_bias,
        cross_attention_dim=None,
        out_bias=attention_out_bias,
        qk_norm=qk_norm,
        processor=attention_processor,
    ))
    attention_args.append(dict(
        query_dim=self.action_inner_dim,
        heads=action_num_attention_heads,
        kv_heads=action_num_attention_heads,
        dim_head=action_attention_head_dim,
        bias=attention_bias,
        cross_attention_dim=inner_dim,
        out_bias=attention_out_bias,
        qk_norm=qk_norm,
        processor=attention_processor,
    ))

    self.action_blocks = nn.ModuleList(
        [
            ActionTransformerBlock(
                attention_class = attention_class,
                attention_args = attention_args,
                dim=self.action_inner_dim,
                num_attention_heads=action_num_attention_heads,
                attention_head_dim=action_attention_head_dim,
                cross_attention_dim=inner_dim,
                qk_norm=qk_norm,
                activation_fn=activation_fn,
                attention_bias=attention_bias,
                attention_out_bias=attention_out_bias,
                eps=norm_eps,
                elementwise_affine=norm_elementwise_affine,
            )
            for _ in range(num_layers)
        ]
    )

    self.action_proj_out = nn.Linear(self.action_inner_dim, action_out_channels) 
    self.action_final_embeddings = action_final_embeddings
    if not self.action_final_embeddings:
        self.action_proj_extra = nn.Linear(self.action_inner_dim, self.action_inner_dim)

    self.action_norm_out = nn.LayerNorm(self.action_inner_dim, eps=1e-6, elementwise_affine=False)


def preprocessing_action_states(
    self,
    action_states: torch.Tensor = None,
    action_timestep: torch.LongTensor = None,
    context_dict: Optional[Dict[str, torch.Tensor]] = None,
):

    assert self.action_expert == True
    assert action_states is not None and action_timestep is not None
    assert context_dict is not None, "context_dict is required for ego/hist injection"

    batch_size = action_states.shape[0]

    device = action_states.device
    dtype  = action_states.dtype
    B, T, _ = action_states.shape

    action_seq_length = action_states.shape[1]

    if getattr(self, "learnable_action_state") and self.learnable_action_state:
        action_states = self.action_state.repeat(batch_size, action_seq_length, 1).to(dtype=action_states.dtype, device=action_states.device)

    action_rotary_emb = self.action_rope(action_states, action_seq_length)
    action_hidden_states = self.action_proj_in(action_states)

    action_temb, action_embedded_timestep = self.action_time_embed(
        action_timestep.flatten(),
        batch_size=batch_size,
        hidden_dtype=action_hidden_states.dtype,
    )
    action_temb = action_temb.view(batch_size, -1, action_temb.size(-1))
    action_embedded_timestep = action_embedded_timestep.view(batch_size, -1, action_embedded_timestep.size(-1))
    
    cmd_onehot = context_dict["cmd_onehot"].to(device=device, dtype=dtype)     # (B, Ccmd)
    vel        = context_dict["vel"].to(device=device, dtype=dtype)            # (B, 2)
    acc        = context_dict["acc"].to(device=device, dtype=dtype)            # (B, 2)

    ego_state = torch.cat([vel, acc,cmd_onehot], dim=-1)       # (B, 8)
    ego_emb = self.action_ego_status_encoder(ego_state)                # (B, D)
    ego_emb = ego_emb.unsqueeze(1).repeat(1, T, 1)              # (B, T, D)
    state_film_6D = self.action_state_film(ego_emb).view(B, T, 6, -1)               # (B,T,6,D_action)
    temb_time_6D  = action_temb.view(B, T, 6, -1)                             # (B,T,6,D_action)  ✅
    temb_fused    = (state_film_6D + temb_time_6D).reshape(B, T, -1)          # (B,T,6*D_action)


    return temb_fused, action_embedded_timestep, action_rotary_emb, action_hidden_states
