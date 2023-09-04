import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import inline
import torchsde    

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        embedding = -math.log(10000) / (dim//2 - 1)
        self.register_buffer('pos', torch.exp(torch.arange(dim//2) * embedding))

    def forward(self, x):
        y = x[:,None].mul(1000) * self.pos[None,:]
        y = torch.cat((y.sin(), y.cos()), -1)
        return y

class LinearTimeSelfAttention1d(nn.Module):
    def __init__(self, dim, heads=4, head_dim=32, groups=32):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, dim)
        self.heads = heads
        self.hidden_dim = head_dim * heads
        self.to_qkv = nn.Conv1d(dim, self.hidden_dim * 3, 1)
        self.to_out = nn.Conv1d(self.hidden_dim, dim, 1)

    def forward(self, x):
        b, c, w = x.shape
        x = self.group_norm(x)
        qkv = self.to_qkv(x)
        q,k,v = qkv.view(b,3,self.heads, self.hidden_dim // self.heads, w).permute(1,0,2,3,4)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = out.reshape(b,self.hidden_dim,w)
        return self.to_out(out)

class ResnetBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, groups=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.time_dim = time_dim
        
        self.time_mlp = nn.Sequential(
            nn.Mish(), 
            nn.Linear(time_dim, out_channels)
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.Mish(),
            nn.Conv1d(in_channels, out_channels, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, out_channels),
            nn.Mish(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1)
        )
        self.skip_conv = nn.Conv1d(in_channels, out_channels, 1)
        
    def forward(self, t, x):
        t = self.time_mlp(t)[...,None]
        y = self.block1(x) + t
        y = self.block2(y)
        return y + self.skip_conv(x)

class Blur1d(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.Tensor([1, 2, 1]) / 4
        self.register_buffer("kernel", kernel)

    def forward(self, x):
        return F.conv1d(x, self.kernel.repeat(x.shape[1],1,1), groups=x.shape[1], padding=1)

class Downsample1d(nn.Sequential):
    def __init__(self, channels, factor=4):
        super().__init__(
            Blur1d(),
            nn.Conv1d(channels, channels, factor+1, stride=factor, padding=factor//2),
        )

class Upsample1d(nn.Sequential):
    def __init__(self, channels, factor=4):
        super().__init__(
            nn.ConvTranspose1d(channels, channels, factor*2, stride=factor, padding=factor//2),
            Blur1d()
        )

class Unet1d(nn.Module):
    def __init__(self, in_channels, scales=(1,2,4,8), hidden_channels=64, groups=32, heads=4, head_dim=32, scale_factor=4):
        super().__init__()
        self.posenc = PositionalEncoding(hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels*4),
            nn.Mish(),
            nn.Linear(hidden_channels*4, hidden_channels)
        )
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, 3, padding=1)
        
        ds_blocks, us_blocks = [],[]
        prev_channels = hidden_channels
        for i,scale in enumerate(scales):
            out_channels = hidden_channels * scale
            ds_block, us_block = [],[]
            ds_block = nn.ModuleList((
                nn.ModuleList((
                    ResnetBlock1d(prev_channels, out_channels, time_dim=hidden_channels, groups=groups),
                    ResnetBlock1d(out_channels, out_channels, time_dim=hidden_channels, groups=groups)
                )),
                Downsample1d(out_channels, scale_factor)
            ))
                
            us_block = nn.ModuleList((
                Upsample1d(out_channels, scale_factor),
                nn.ModuleList((
                    ResnetBlock1d(out_channels*2, out_channels, time_dim=hidden_channels, groups=groups),
                    ResnetBlock1d(out_channels*2, prev_channels, time_dim=hidden_channels, groups=groups)
                ))
            ))
            
            ds_blocks.append(ds_block)
            us_blocks.append(us_block)
            prev_channels = out_channels
            
        self.ds_blocks = nn.ModuleList(ds_blocks)
        self.us_blocks = nn.ModuleList(us_blocks)
        self.mid_block = nn.ModuleList((
            ResnetBlock1d(prev_channels, prev_channels, time_dim=hidden_channels, groups=groups),
            LinearTimeSelfAttention1d(prev_channels,heads=heads, head_dim=head_dim, groups=groups),
            ResnetBlock1d(prev_channels, prev_channels, time_dim=hidden_channels, groups=groups),
        ))
        self.conv_out = nn.Sequential(
            nn.GroupNorm(groups, hidden_channels),
            nn.Mish(),
            nn.Conv1d(hidden_channels, in_channels, 1)
        )
    
    def forward(self, t, x):
        t = self.mlp(self.posenc(t))
        x = self.conv1(x)
        ys = [x]
        
        for i, blocks in enumerate(self.ds_blocks):
            for resnet_block in blocks[0]:
                x = resnet_block(t,x)
                ys.append(x)            
            
            x = blocks[1](x)
                
        x = self.mid_block[0](t,x)
        x = x + self.mid_block[1](x)
        x = self.mid_block[2](t,x)

        for i, blocks in enumerate(reversed(self.us_blocks)):
            x = blocks[0](x)
            for resnet_block in blocks[1]:
                x = torch.cat((x, ys.pop()), 1)
                x = resnet_block(t,x)

        x = self.conv_out(x)
        return x

def fill_tail_dims(y: torch.Tensor, y_like: torch.Tensor):
    return y[(...,) + (None,) * (y_like.dim() - y.dim())]

class ScoreMatchingSDE(nn.Module):
    def __init__(self, input_size=(1, 256*256)):
        super().__init__()
        self.input_size = input_size
        self.denoiser = Unet1d(input_size[0], scales=(1,2,4,8,16))  

    def score(self, t, y):
        if isinstance(t, (int,float)):
            t = y.new_tensor(float(t))
        if t.dim() == 0:
            t = t.repeat(y.shape[0])
        return self.denoiser(t, y)

    def _beta(self, t):
        return 0.1 + t * 19.9

    def _indefinite_int(self, t):
        """Indefinite integral of beta(t)."""
        return 0.1 * t + .5 * t ** 2 * 19.9

    def analytical_mean(self, t, x_t0):
        mean_coeff = (-.5 * (self._indefinite_int(t) - self._indefinite_int(0))).exp()
        mean = x_t0 * fill_tail_dims(mean_coeff, x_t0)
        return mean

    def analytical_var(self, t, x_t0):
        analytical_var = 1 - (-self._indefinite_int(t) + self._indefinite_int(0)).exp()
        return analytical_var

    @torch.no_grad()
    def analytical_sample(self, t, x_t0):
        mean = self.analytical_mean(t, x_t0)
        var = self.analytical_var(t, x_t0)
        return mean + torch.randn_like(mean) * fill_tail_dims(var.sqrt(), mean)

    @torch.no_grad()
    def analytical_score(self, x_t, t, x_t0):
        mean = self.analytical_mean(t, x_t0)
        var = self.analytical_var(t, x_t0)
        return - (x_t - mean) / fill_tail_dims(var, mean).clamp_min(1e-5)

    def f(self, t, y):
        return -0.5 * self._beta(t) * y

    def g(self, t, y):
        return fill_tail_dims(self._beta(t).sqrt(), y).expand_as(y)

    def sample_t1_marginal(self, batch_size, tau=1.):
        return torch.randn(size=(batch_size, *self.input_size), device=next(self.parameters()).device) * math.sqrt(tau)

    def lambda_t(self, t):
        return self.analytical_var(t, None)

    def forward(self, x_t0, partitions=1):
        """Compute the score matching objective.
        Split [t0, t1] into partitions; sample uniformly on each partition to reduce gradient variance.
        """
        u = torch.rand(size=(x_t0.shape[0], partitions), dtype=x_t0.dtype, device=x_t0.device)
        u.mul_(1 / partitions)
        shifts = torch.arange(0, partitions, device=x_t0.device, dtype=x_t0.dtype)[None, :]
        shifts.mul_(1 / partitions)
        t = (u + shifts).reshape(-1)
        lambda_t = self.lambda_t(t)

        x_t0 = x_t0.repeat_interleave(partitions, dim=0)
        x_t = self.analytical_sample(t, x_t0)

        fake_score = self.score(t, x_t)
        true_score = self.analytical_score(x_t, t, x_t0)
        loss = (fake_score - true_score) ** 2
        #loss *= inline.gaussian2d(32)
        loss = (lambda_t * loss.flatten(start_dim=1).sum(dim=1))
        return loss

class ReverseSDE(nn.Module):
    noise_type = "diagonal"
    sde_type = "stratonovich"

    def __init__(self, module: ScoreMatchingSDE):
        super().__init__()
        self.module = module

    def f(self, t, y):
        y = y.view(-1, *self.module.input_size)
        out = -(self.module.f(-t, y) - self.module.g(-t, y) ** 2 * self.module.score(-t, y))
        return out.flatten(start_dim=1)

    def g(self, t, y):
        y = y.view(-1, *self.module.input_size)
        out = -self.module.g(-t, y)
        return out.flatten(start_dim=1)

    def sample_t1_marginal(self, batch_size, tau=1.):
        return self.module.sample_t1_marginal(batch_size, tau)

    @torch.no_grad()
    def sde_sample(self, batch_size=8, tau=1., t=None, y=None, dt=1e-2, tweedie_correction=True):
        self.module.eval()

        t = torch.tensor([-1., 0.], device=next(self.parameters()).device) if t is None else t
        y = self.sample_t1_marginal(batch_size, tau) if y is None else y

        ys = torchsde.sdeint(self, y.flatten(start_dim=1), t, dt=dt)
        ys = ys.view(len(t), *y.size())
        if tweedie_correction:
            ys[-1] = self.tweedie_correction(0., ys[-1], dt)
        return ys

    @torch.no_grad()
    def sde_sample_final(self, batch_size=1, tau=1., t=None, y=None, dt=1e-2):
        return self.sde_sample(batch_size, tau, t, y, dt)[-1]

    def tweedie_correction(self, t, y, dt):
        return y + dt ** 2 * self.module.score(t, y)

