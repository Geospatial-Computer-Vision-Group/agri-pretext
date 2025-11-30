from collections import OrderedDict
import math
from typing import Callable, List, Optional, Sequence, Tuple, Union
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

import torch
from utils import to_2tuple
from pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid
from einops import rearrange

class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x

class DateEmbed(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.d_model = d_model
        self.register_buffer("div_term",torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2))))
    def forward(self,months,years):
        assert months.size(0) == years.size(0)
        B = months.size(0)
        month_pe = torch.empty((B,self.d_model//2),device=self.div_term.device)
        year_pe = torch.empty((B,self.d_model//2),device=self.div_term.device)
        month_pe[:, 0::2] = torch.sin(months.view(-1,1) * self.div_term)
        month_pe[:, 1::2] = torch.cos(months.view(-1,1) * self.div_term)
        year_pe[:, 0::2] = torch.sin(years.view(-1,1) * self.div_term)
        year_pe[:, 1::2] = torch.cos(years.view(-1,1) * self.div_term)
        return torch.cat((month_pe,year_pe),dim=1)

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            scaled_cosine: bool = False,
            scale_heads: bool = False,
            logit_scale_max: float = math.log(1. / 0.01),
            batch_first: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max
        self.batch_first = batch_first
        self.use_fsdpa = hasattr(nn.functional, 'scaled_dot_product_attention')

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        if self.batch_first:
            x = x.transpose(0, 1)

        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.reshape(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.reshape(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.reshape(L, N * self.num_heads, -1).transpose(0, 1)

        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        if self.logit_scale is not None:
            attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = torch.bmm(attn, v)
        else:
            if self.use_fsdpa:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
            else:
                q = q * self.scale
                attn = torch.bmm(q, k.transpose(-1, -2))
                if attn_mask is not None:
                    attn += attn_mask
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = torch.bmm(attn, v)

        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)

        x = x.transpose(0, 1).reshape(L, N, C)

        if self.batch_first:
            x = x.transpose(0, 1)

        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class AttentionalPooler(nn.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim, batch_first=True)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor):
        N = x.shape[0]
        x = self.ln_k(x)
        q = self.ln_q(self.query)
        out = self.attn(q.unsqueeze(0).expand(N, -1, -1), x, x, need_weights=False)[0]
        return out


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
            batch_first: bool = True,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=batch_first)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(
            q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask
        )[0]

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class CustomResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
            batch_first: bool = True,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = Attention(
            d_model,
            n_head,
            scaled_cosine=scale_cosine_attn,
            scale_heads=scale_heads,
            batch_first=batch_first,
        )
        self.ln_attn = norm_layer(d_model) if scale_attn else nn.Identity()
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ('ln', norm_layer(mlp_width) if scale_fc else nn.Identity()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def get_reference_weight(self):
        return self.mlp.c_fc.weight

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.ls_1(self.ln_attn(self.attn(self.ln_1(x), attn_mask=attn_mask)))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


class Transformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            batch_first: bool = True,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.batch_first = batch_first
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                batch_first=batch_first,
            )
            for _ in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()    # NLD -> LND
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        if not self.batch_first:
            x = x.transpose(0, 1)    # LND -> NLD
        return x
    
    def forward_with_features(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        features = []
        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()    # NLD -> LND
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
            features.append(x)
        if not self.batch_first:
            x = x.transpose(0, 1)    # LND -> NLD
        return x,features


class CustomTransformer(nn.Module):
    """ A custom transformer that can use different block types. """
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            batch_first: bool = True,
            block_types: Union[str, List[str]] = 'CustomResidualAttentionBlock',
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.batch_first = batch_first  # run trasnformer stack in batch first (N, L, D)
        self.grad_checkpointing = False

        if isinstance(block_types, str):
            block_types = [block_types] * layers
        assert len(block_types) == layers

        def _create_block(bt: str):
            if bt == 'CustomResidualAttentionBlock':
                return CustomResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio=mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    batch_first=batch_first,
                )
            else:
                assert False

        self.resblocks = nn.ModuleList([
            _create_block(bt)
            for bt in block_types
        ])

    def get_cast_dtype(self) -> torch.dtype:
        weight = self.resblocks[0].get_reference_weight()
        if hasattr(weight, 'int8_original_dtype'):
            return weight.int8_original_dtype
        return weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if not self.batch_first:
            x = x.transpose(0, 1)  # NLD -> LND

        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)

        if not self.batch_first:
            x = x.transpose(0, 1)  # NLD -> LND
        return x
    
class TimeTranslator(nn.Module):
    def __init__(self,encoder_dim,img_size,patch_size,depth,heads,norm_layer: Callable = LayerNorm):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.depth = depth
        self.transformer = Transformer(width=encoder_dim,layers=depth,heads=heads)
        self.t1_embed = DateEmbed(encoder_dim // 3)
        self.t2_embed = DateEmbed(encoder_dim // 3)
        pe = get_2d_sincos_pos_embed(encoder_dim // 3, img_size // patch_size)
        self.register_buffer('pe', torch.from_numpy(pe).float())
        self.norm = norm_layer(encoder_dim)
    def forward(self,x,m1,m2,y1,y2):
        B,N,D = x.shape
        t1e = self.t1_embed(m1,y1)
        t2e = self.t1_embed(m2,y2)
        te = torch.cat((t1e,t2e),dim=-1) # B x 1 x 2*D//3
        pe = torch.cat((self.pe.unsqueeze(0).expand(B,-1,-1),te.unsqueeze(1).expand(-1,N,-1)),dim=-1) # 1 x N x D//3
        x = x + pe
        x = self.transformer(x)
        x = self.norm(x)
        return x

class Decoder(nn.Module):
    def __init__(self,encoder_dim,dim,img_size,patch_size,in_chans,depth,heads,norm_layer: Callable = LayerNorm):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.dim = dim
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.to_decoder_embed = nn.Linear(encoder_dim,dim)
        self.transformer = Transformer(width=dim,layers=depth,heads=heads)
        pe = get_2d_sincos_pos_embed(dim, img_size // patch_size)
        self.register_buffer('pe', torch.from_numpy(pe).float())
        self.norm = norm_layer(dim)
        self.decoder_pred = nn.Linear(dim,patch_size**2 * in_chans)
    def forward(self,x):
        x = self.to_decoder_embed(x)
        x = x + self.pe
        x = self.transformer(x)
        x = self.norm(x)
        x = self.decoder_pred(x)
        return x

class FreqDecoder(nn.Module):
    def __init__(self,encoder_dim,dim,img_size,patch_size,out_chans,depth,heads,norm_layer: Callable = LayerNorm):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.dim = dim
        self.depth = depth
        self.to_decoder_embed = nn.Linear(encoder_dim,dim//2)
        self.decoder_pred = nn.Linear(dim,patch_size**2 * out_chans)
        self.transformer = Transformer(width=dim,layers=depth,heads=heads)
        self.t1_embed = DateEmbed(dim // 3)
        self.t2_embed = DateEmbed(dim // 3)
        pe = get_2d_sincos_pos_embed(dim // 3, img_size // patch_size)
        self.register_buffer('pe', torch.from_numpy(pe).float())
        self.norm = norm_layer(dim)
    def forward(self,x1,x2,m1,m2,y1,y2):
        x1 = self.to_decoder_embed(x1)
        x2 = self.to_decoder_embed(x2)
        x = torch.cat((x1,x2),dim=-1)
        B,N,D = x.shape
        t1e = self.t1_embed(m1,y1)
        t2e = self.t2_embed(m2,y2)
        pe = torch.cat((self.pe.unsqueeze(0).expand(B,-1,-1),t1e.unsqueeze(1).expand(-1,N,-1),t2e.unsqueeze(1).expand(-1,N,-1)),dim=-1)
        x = x + pe
        x = self.transformer(x)
        x = self.norm(x)
        x = self.decoder_pred(x)
        return x

def patchify(imgs, p):
    """
    imgs: (N, C, H, W)
    x: (N, L, patch_size**2 *C)
    p: int
    """
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    C = imgs.size(1)
    x = imgs.reshape(shape=(imgs.shape[0], C, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * C))
    return x

def unpatchify(x, p, C):
    """
    x: (N, L, patch_size**2 *C)
    imgs: (N, C, H, W)
    p: int
    """
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]
    
    x = x.reshape(shape=(x.shape[0], h, w, p, p, C))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], C, h * p, h * p))
    return imgs
