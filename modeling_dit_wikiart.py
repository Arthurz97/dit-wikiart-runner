import torch
import torch.nn as nn
import math

from huggingface_hub import PyTorchModelHubMixin
from typing import Union, Optional, Tuple


class TimestepEmbedder(nn.Module):
    """Module to create timestep's embedding."""
    def __init__(self,hidden_size,frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size,hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size,hidden_size)
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0,end=half) / half
        ).to(device=t.device)
        args = torch.einsum('i,j->ij', t, freqs.to(t.device))
        freqs = torch.cat([torch.cos(args),torch.sin(args)],dim=-1)

        mlp_input_dtype = next(self.mlp.parameters()).dtype
        freqs_casted = freqs.to(mlp_input_dtype)

        return self.mlp(freqs_casted)

class ViTAttn(nn.Module):
    def __init__(self,hidden_size,num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size,num_heads,bias=True,add_bias_kv=True,batch_first=True)
    
    def forward(self,x):
        attn, _ = self.attn(x,x,x)
        return attn
    
class DiTBlock(nn.Module):
    """
    DiT Block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Using post-norm
    """
    def __init__(self,hidden_size,num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size,elementwise_affine=False,eps=1e-6)
        self.attn = ViTAttn(hidden_size,num_heads)
        self.norm2 = nn.LayerNorm(hidden_size,elementwise_affine=False,eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size,4*hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(4*hidden_size,hidden_size)
        )
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size,6*hidden_size)
        )
    
    def forward(self,x,c):
        gamma_1,beta_1,alpha_1,gamma_2,beta_2,alpha_2 = self.adaLN(c).chunk(6,dim=1)
        x = self.norm1(x + alpha_1.unsqueeze(1) * self.attn(x))
        x = x * (1+gamma_1.unsqueeze(1)) + beta_1.unsqueeze(1)
        x = self.norm2(x + alpha_2.unsqueeze(1) * self.mlp(x))
        x = x * (1+gamma_2.unsqueeze(1)) + beta_2.unsqueeze(1)
        return x

class DiTWikiartModel(nn.Module,
                      PyTorchModelHubMixin):
    def __init__(self,
                 num_blocks=8,
                 hidden_size=384,
                 num_heads=6,
                 patch_size=2,
                 num_channels=4,
                 img_size=32,
                 num_genres=42,
                 num_styles=137):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.seq_len = (img_size // patch_size)**2
        self.img_size = img_size
        self.blocks = nn.ModuleList(
            DiTBlock(hidden_size,num_heads) for _ in range(num_blocks)
        )
        self.timestep_embed = TimestepEmbedder(hidden_size)

        self.num_genres = num_genres
        self.num_styles = num_styles
        self.genre_condition = nn.Embedding(num_genres+1,hidden_size) # +1 for null condition
        self.style_condition = nn.Embedding(num_styles+1,hidden_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, hidden_size))

        patch_dim = num_channels * patch_size * patch_size
        self.proj_in = nn.Linear(patch_dim,hidden_size)
        self.proj_out = nn.Linear(hidden_size,patch_dim)

        self.norm_out = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2*hidden_size)
        )

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.proj_out.weight, std=0.02)
        nn.init.zeros_(self.proj_out.bias)
        nn.init.normal_(self.proj_in.weight, std=0.02)
        nn.init.zeros_(self.proj_in.bias)

        nn.init.normal_(self.timestep_embed.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.timestep_embed.mlp[0].bias)
        nn.init.normal_(self.timestep_embed.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.timestep_embed.mlp[2].bias)

        for block in self.blocks:
            nn.init.zeros_(block.adaLN[-1].weight)
            nn.init.zeros_(block.adaLN[-1].bias)

        nn.init.zeros_(self.adaLN_final[-1].weight)
        nn.init.zeros_(self.adaLN_final[-1].bias)

        nn.init.normal_(self.genre_condition.weight, std=0.02)
        nn.init.normal_(self.style_condition.weight, std=0.02)

    def patchify(self,z):
        """
        from (batch_size,6,32,32) -> (batch_size,256,24) -> (batch_size,256,hidden_size)
        """
        b,_,_,_ = z.shape
        c = self.num_channels
        p = self.patch_size
        z = z.unfold(2,p,p).unfold(3,p,p) # (b,c,h//p,p,w//p,p)
        z = z.contiguous().view(b,c,-1,p,p) # (b,c,hw//p**2,p,p)
        z = torch.einsum('bcapq->bacpq',z).contiguous().view(b,-1,c*p**2) # (b,hw//p**2,c*p**2)
        return self.proj_in(z) # (b,hw//p**2,hidden_size)

    def unpatchify(self,z):
        """
        from (batch_size,256,hidden_size) -> (batch_size,256,24) -> (batch_size,6,32,32)
        """
        b,_,_ = z.shape
        c = self.num_channels
        p = self.patch_size
        s = int(self.seq_len ** 0.5)
        i = self.img_size
        z = self.proj_out(z) # (b,hw//p**2,c*p**2)
        z = z.view(b,s,s,c,p,p) # (b,h/p,w/p,c,p,p)
        z = torch.einsum('befcpq->bcepfq',z) # (b,c,h/p,p,w/p,p)
        z = z.contiguous().view(b,c,i,i)
        return z

    def forward(self,z,t,g,s):
        t_embed = self.timestep_embed(t) # t_embed: (batch_size, hidden_size)
        g_embed = self.genre_condition(g)
        s_embed = self.style_condition(s)

        c = t_embed + g_embed + s_embed

        z = self.patchify(z)
        z = z + self.pos_embed

        for block in self.blocks:
            z = block(z,c)
        
        gamma, beta = self.adaLN_final(c).chunk(2,dim=-1)
        z = self.norm_out(z)
        z = z * (1+gamma.unsqueeze(1)) + beta.unsqueeze(1)

        return self.unpatchify(z)

