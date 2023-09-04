import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.nn.parameter import Parameter
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from typing import Union,Tuple,List
from collections import OrderedDict

def seq(items, as_sequential=True):
    if items is None: 
        return None
    if isinstance(items, nn.Module):
        return items
    if isinstance(items, (tuple, list)):
        if len(items) == 0: 
            return nn.Identity()
        elif len(items) == 1:
            return seq(items[0], as_sequential=as_sequential)
        elif as_sequential:
            return nn.Sequential(*[seq(item) for item in items])
        else:
            return [seq(item) for item in items]
    if isinstance(items, dict):
        if len(items) == 0:
            return nn.Identity()
        elif as_sequential:
            return nn.Sequential(OrderedDict({key: seq(item) for key,item in items.items()}))
        else:
            return OrderedDict({key: seq(item) for key,item in items.items()})
    assert False, "Unexpected type while unpacking sequential"

def all_combinations(*args):
    assert len(args)
    if len(args) == 1: return [[a] for a in args[0]]
    else: return [a + b for a in all_combinations(*args[:len(args)//2]) for b in all_combinations(*args[len(args)//2:])]

def all_cosine_similarities(a,b, absolute=False):
    assert a.dtype == b.dtype
    if a.dtype in { torch.cfloat, torch.cdouble }:
        if absolute: r = (a @ b.mT.conj()).abs()
        else: r = (a @ b.mT.conj()).real
        r = r / (a.abs().pow(2).sum(-1).unsqueeze(-1) * b.abs().pow(2).sum(-1).unsqueeze(-2)).sqrt()
    else:
        r = (a @ b.mT)
        r = r / (a.pow(2).sum(-1).unsqueeze(-1) * b.pow(2).sum(-1).unsqueeze(-2)).sqrt()
    return torch.nan_to_num(r)

def padstack(tensors, dim=0, pad_mode="constant", pad_value=0):
    assert len(tensors) > 0
    ndims = tensors[0].dim()
    for t in tensors[1:]: assert t.dim() ==  ndims
    new_shape = [max([t.shape[d] for t in tensors]) for d in range(ndims)]
    padding = [[new_shape[d//2] - t.shape[d//2] if (d & 1) else 0 for d in range(ndims*2,0,-1)] for t in tensors]
    return torch.stack([F.pad(t,p,pad_mode,pad_value) for t,p in zip(tensors,padding)], dim=dim)

def padcat(tensors, dim=0, pad_mode="constant", pad_value=0):
    assert len(tensors) > 0
    ndims = tensors[0].dim()
    for t in tensors[1:]: assert t.dim() ==  ndims
    new_shape = [max([t.shape[d] for t in tensors]) if d != dim else -1 for d in range(ndims)]
    padding = [[new_shape[d//2] - t.shape[d//2] if (d & 1 and d//2 != dim) else 0 for d in range(ndims*2,0,-1)] for t in tensors]
    return torch.cat([F.pad(t,p,pad_mode,pad_value) for t,p in zip(tensors,padding)], dim=dim)

def rgb_to_yuv(rgb: torch.Tensor, clamp=True) -> torch.Tensor:
    if rgb.dtype == torch.uint8: 
        rgb = rgb / 255
    m = torch.tensor([
        [ 0.21260,  0.71520, 0.07220],
        [-0.09991, -0.33609, 0.43600],
        [ 0.61500, -0.55861,-0.05639]],
        device=rgb.device)
    yuv = (m @ rgb.flatten(-2)).view(rgb.shape)
    if clamp:
        yuv.select(-3, 0).clamp_(0,1)
        yuv.select(-3,[1,2]).clamp_(-1,1)
    return yuv

def yuv_to_rgb(yuv: torch.Tensor, clamp=True) -> torch.Tensor:
    m = torch.tensor([
        [1, 0.00000, 1.28033],
        [1,-0.21482,-0.38059],
        [1, 2.12798, 0.00000]], 
        device=yuv.device)
    rgb = (m @ yuv.flatten(-2)).view(yuv.shape)
    return rgb.clamp(0,1) if clamp else rgb

def nth_color(i: int, n: int) -> tuple: 
    """Pick color i out of n colors equaly divided over a color circle"""
    return (0.5+math.cos((i/n + 0/3) * 2 * math.pi)/2, 0.5+math.cos((i/n + 1/3) * 2 * math.pi)/2,0.5+math.cos((i/n + 2/3) * 2 * math.pi)/2) 

def nth_color_u8(i: int, n: int) -> tuple: 
    """Pick color i out of n colors equaly divided over a color circle"""
    return (int(127.5+math.cos((i/n + 0/3) * 2 * math.pi)*127.5), int(127.5+math.cos((i/n + 1/3) * 2 * math.pi)*127.5),int(127.5+math.cos((i/n + 2/3) * 2 * math.pi)*127.5)) 

def n_colors(n: int) -> list:
    return [nth_color(i, n) for i in range(n)]

def n_colors_u8(n: int) -> list:
    return [nth_color_u8(i, n) for i in range(n)]

def gaussian1d(kernel_size: int, sigma=None, sym:bool=True) -> torch.Tensor:
    if kernel_size == 1: return torch.ones(1)
    odd = kernel_size % 2
    if sigma is None:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    if not sym and not odd:
        kernel_size = kernel_size + 1
    n = torch.arange(0, kernel_size) - (kernel_size - 1.0) / 2.0
    sig2 = 2 * sigma * sigma
    w = torch.exp(-n ** 2 / sig2)
    if not sym and not odd:
        w = w[:-1]
    return w / w.sum()

def gaussian2d(kernel_size: int, std=None, sym:bool=True) -> torch.Tensor:
    w = gaussian1d(kernel_size, std, sym)
    w = torch.outer(w,w)
    return w

def unflatten_dict(d):
    def _insert(d, key, v):
        if "." in key:
            head,tail = key.split(".",1)
            d[head] =_insert(d[head] if head in d else {}, tail, v)
        else: d[key] = v
        return d
    r = {}
    for key in d: r = _insert(r, key, d[key])
    return r

def all_detach(data):
    if isinstance(data, torch.Tensor):
        return data.detach()
    if isinstance(data, list):
        return list([all_detach(d) for d in data])
    if isinstance(data, tuple):
        return tuple([all_detach(d) for d in data])
    if isinstance(data, dict):
        copy = {}
        for k in data:
            copy[k] = all_detach(data[k])
        return copy
    return data

def all_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, list):
        return list([all_to_device(d, device) for d in data])
    if isinstance(data, tuple):
        return tuple([all_to_device(d, device) for d in data])
    if isinstance(data, dict):
        copy = {}
        for k in data:
            copy[k] = all_to_device(data[k], device)
        return copy
    return data


def norm(t, mean=None, std=None, clamp=(None,None), eps:float=1e-8) -> torch.Tensor:
    if t.dtype == torch.uint8: 
        t = t / 255
    
    if mean is None:
        mean = t.mean((-1,-2), keepdim=True)
    if std is None:
        std = t.std((-1,-2), keepdim=True)
    
    if not isinstance(mean, (tuple, list, torch.Tensor)):
        t = t - mean 
    elif isinstance(mean, torch.Tensor):
        while mean.dim() < 3: mean = mean[...,None]
        t = t - mean
    else:
        assert len(mean) == t.shape[-3]
        t = torch.stack([t.select(-3,i) - v for i,v in enumerate(mean)]) 
    
    if not isinstance(std, (tuple, list, torch.Tensor)):
        if std < eps: std = eps
        t = t / std
    elif isinstance(mean, torch.Tensor):
        while std.dim() < 3: std = std[...,None]
        t = t / std.clamp(eps, None)
    else:
        assert len(std) == t.shape[-3]
        t = torch.stack([t.select(-3,i) / (eps if v < eps else v) for i,v in enumerate(std)]) 
    
    if t.dtype not in (torch.cfloat, torch.cdouble):
        if clamp[0] is not None or clamp[1] is not None:
            t = t.clamp(*clamp)
    
    return t

def denorm(t:torch.Tensor, mean=0.0, std=1.0, clamp=(0,1), eps:float=1e-8) -> torch.Tensor:
    if t.dtype == torch.uint8: 
        t = t / 255
        
    if not isinstance(std, (tuple, list, torch.Tensor)):
        t = t * std 
    elif isinstance(mean, torch.Tensor):
        while std.dim() < 3: std = std[...,None]
        t = t * std 
    else:
        assert len(std) == t.shape[-3]
        t = torch.stack([t.select(-3,i) * v for i,v in enumerate(std)]) 
    
    if not isinstance(mean, (tuple, list, torch.Tensor)):
        t = t + mean 
    elif isinstance(mean, torch.Tensor):
        while mean.dim() < 3: mean = mean[...,None]
        t = t + mean
    else:
        assert len(mean) == t.shape[-3]
        t = torch.stack([t.select(-3,i) + v for i,v in enumerate(mean)]) 
    
    if t.dtype not in (torch.cfloat, torch.cdouble):
        if clamp[0] is not None or clamp[1] is not None:
            t = t.clamp(*clamp)
    return t

def disp(t: torch.Tensor, permute=False, numpy=False, clamp=True, channels=None, device=None, dtype=torch.float, rotations=1, brightness=1, saturation=1) -> torch.Tensor:
    t = t.detach()
    
    dim = t.dim()
    while t.dim() < 4: t = t[None]
    
    if channels is not None:
        if t.shape[-3] < channels:
            t = F.pad(t, [0,0,0,0,0,channels-t.shape[-3]], "constant", 0)
        else:
            t = t[:,:channels]
            
    elif channels is None and t.shape[-3] == 2:
        t = F.pad(t, [0,0,0,0,0,1], "constant", 0.5)
    
    if channels is not None or t.shape[-3] > 4:
        channels = t.shape[-3]
        s = torch.linspace(-0.5* math.pi, 0.5 * math.pi, channels).repeat(t.shape[0])
        s = torch.complex(s.cos(), s.sin()).to(t.device).view(t.shape[0],channels,1,1)
        t = torch.complex(t, torch.zeros_like(t)) * s
        t = t.sum(-3, keepdim=True)
        t = 0.5 * (t - t.mean()) / t.std()
        t.real = torch.nan_to_num(t.real)
        t.imag = torch.nan_to_num(t.imag)
        
    # convert to float
    if t.dtype == torch.uint8: 
        t = t / 255
    elif t.dtype in [torch.cfloat, torch.cdouble]:
        if rotations != 1:
            t = t ** rotations / t.abs() ** (rotations-1)
        y = t.abs()
        u,v = t.real, t.imag
        y = torch.cat((y,u,v), -3)
        t = yuv_to_rgb(y)
        
    if t.dtype in [torch.float, torch.double]:
        if brightness != 1:
            t = torch.nan_to_num(t ** (1/brightness))
        if saturation != 1:
            mean = t.mean(-3, keepdim=True)
            std = t.std(-3, keepdim=True)
            y = torch.nan_to_num((t - mean) / std)
            #mask = y < 0
            y = y * saturation
            t = y * std + mean
    else:
        assert False, f"input.dtype of {t.dtype} is not supported"
    
    if clamp:
        t = t.clamp(0,1)
    
    if dtype == torch.uint8:
        t = (t * 255).byte()
    
    if device is not None:
        t = t.to(device)

    if dim < 4: t = t[0]
    return t

class Constant(nn.Module):
    def __init__(self, tensor, requires_grad=False):
        super().__init__()
        if not isinstance(tensor, torch.Tensor):
            if isinstance(tensor, list):
                tensor = torch.tensor(tensor)
                while tensor.dim() < 4: tensor = tensor.unsqueeze(0)
            else:
                tensor = torch.Tensor((tensor,)).view(1,1,1,1)
        self.value = nn.parameter.Parameter(tensor, requires_grad=requires_grad)
        
    def forward(self, x):
        return self.value
    
    def extra_repr(self):
        return f"value={self.value}"

class Iterate(nn.Module):
    def __init__(self, *args, loops=1):
        super().__init__()
        self.body = seq(args)
        self.loops = loops

    def forward(self, x):
        for _ in range(self.loops):
            x = self.body(x)
        return x

    def extra_repr(self):
        return f"loops={self.loops}"

def to_tensor(value, mod=None, x=None, device=None):
    if isinstance(value, str):
        value = eval(value)
    if callable(value): 
        value = value(mod,x)
    if isinstance(value, torch.Tensor): 
        return value.to(device)
    else:
        return torch.tensor(value, device=device)

class Multiply(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
        
    def forward(self,x):
        x = x * to_tensor(self.value, mod=self, x=x, device=x.device)
        return x

class Add(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
        
    def forward(self,x):
        x = x * to_tensor(self.value, mod=self, x=x, device=x.device)
        return x

class Sum(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs  = kwargs

    def forward(self, x):
        return x.sum(*self.args, **self.kwargs)

class MLP(nn.Sequential):
    def __init__(self, in_channels, out_channels, hidden_channels=None, layers=1, expand=4, act=nn.ReLU, dropout=0.0):
        super().__init__(
                *[nn.Sequential(nn.Linear(c, hidden_channels or in_channels*expand), act()) \
                  for c in ([in_channels] + [hidden_channels or in_channels*expand] * (layers-1))],
                nn.Linear(in_channels*expand, out_channels))

class FlattenDims(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
    
    def forward(self, x):
        sorted_dims = sorted([d if d >= 0 else len(x)+d for d in self.dims])
        reshape = list(x.shape[:sorted_dims[0]]) + [-1] + list(x.shape[sorted_dims[-1]+1:])
        return x.reshape(*reshape)

    def extra_repr(self):
        return "dims=(" + ", ".join([str(v) for v in self.dims]) + ")"

class ReshapeDim(nn.Module):
    def __init__(self, *shape, dim=0):
        super().__init__()
        self.shape = shape
        self.dim = dim

    def forward(self, x):
        shape = x.shape
        reshape = shape[:self.dim] + self.shape + shape[self.dim+1:]
        return x.reshape(*reshape)

    def extra_repr(self):
        return "shape=(" + ", ".join([str(v) for v in self.shape]) + "), dim=" + str(self.dim)
class Affine(nn.Module):
    def __init__(self, multiply=1, add=0):
        super().__init__()
        self.multiply = multiply
        self.add = add
        
    def forward(self,x):
        x = x * to_tensor(self.multiply, mod=self, x=x, device=x.device) + to_tensor(self.add, mod=self, x=x, device=x.device)
        return x

class Contiguous(nn.Module):
    def forward(self, x):
        return x.contiguous()
    
    
class Conv2d(nn.Module):
    def __init__(self, weight, bias, *args, **kwargs):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.args = args
        self.kwargs = kwargs
    
    def forward(self, x):
        weight = self.weight
        bias = self.bias
        if callable(weight):
            weight = weight(x)
        if callable(bias):
            bias = bias(x)
        return torch.conv2d(x, weight, bias, *self.args, **self.kwargs)
    
class Ignore(nn.Module):
    def __init__(self, *args, traverse=False):
        super().__init__()
        self.body = seq(args)
        self.traverse = traverse
        
    def forward(self, x):
        if self.traverse:
            self.body(x.clone())
        return x
    
    def extra_repr(self):
        return f"traverse={self.traverse}"

class Skip(nn.Sequential):
    def __init__(self, *args, reduction="+", drop_path=0):
        body = seq(args, as_sequential=False)
        
        if isinstance(body, OrderedDict):
            super().__init__(body)
        else:
            super().__init__(*body)
        self.reduction = reduction
        self.drop_path = drop_path

    def forward(self, x):
        if self.training and self.drop_path > 0 and torch.rand((1,)) < self.drop_path: return x
        if self.reduction == None: return x
        
        y = super().forward(x)
        if self.reduction == "+": return x + y
        if self.reduction == "-": return x - y
        if self.reduction == "*": return x * y
        if self.reduction == "/": return x / y 
        assert False, "invalid reduction"

    def extra_repr(self):
        if self.reduction != "+":
            return  f"reduction=\"{self.reduction}\""
        else:
            return ""

        
class Learnable(nn.Module):
    def __init__(self, shape=None, init=None):
        if shape is None:
            assert isinstance(init, torch.Tensor)
        super().__init__()
        self.shape = shape or init.shape
        self.init = init or 0
        self.value = nn.parameter.Parameter(torch.empty(*self.shape))
        
        init = self.init
        if isinstance(init, str):
            init = eval(init)
        if callable(init):
            init(self.value)
        elif isinstance(init, torch.Tensor):
            with torch.no_grad():
                self.value.copy_(init)
        else:
            with torch.no_grad():
                self.value.copy_(torch.full(self.shape, init))
        
    def forward(self, x):
        return self.value
         
def nan_to_num(input, nan=0.0, posinf=None, neginf=None):
    if input.dtype in [torch.cfloat, torch.cdouble]:
        real = torch.nan_to_num(input.real, nan=nan, posinf=posinf, neginf=neginf)
        imag = torch.nan_to_num(input.imag, nan=nan, posinf=posinf, neginf=neginf)
        return torch.complex(real, imag)
    else:
        return torch.nan_to_num(input, nan=nan, posinf=posinf, neginf=neginf)
        
class NanToNum(nn.Module):
    def __init__(self, nan=0.0, posinf=None, neginf=None):
        super().__init__()
        self.nan = nan
        self.posinf = posinf
        self.neginf = neginf
        
    def forward(self, x):
        return nan_to_num(x, self.nan, self.posinf, self.neginf)
    
class Load(nn.Module):
    def __init__(self, key):
        super().__init__()
        self.key = key
    
    def forward(self, x):
        return x[self.key]

    def extra_repr(self):
        return repr(self.key)
        
class Store(nn.Module):
    def __init__(self, key):
        super().__init__()
        self.key = key
    
    def forward(self, x):
        return { self.key: x }

    def extra_repr(self):
        return repr(self.key)
    
class Select(nn.Module):
    def __init__(self, dim, index):
        super().__init__()
        self.dim = dim
        self.index = index
        
    def forward(self, x):
        if isinstance(self.index, (list, tuple)):
            return torch.stack([x.select(self.dim, idx) for idx in self.index], dim=self.dim)
        else:
            return x.select(self.dim, self.index)
    def extra_repr(self):
        return f"dim={self.dim}, index={self.index}"
    
class CenterCrop(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size if isinstance(size, tuple) else (size, size)
    
    def forward(self ,x):
        return TF.center_crop(x, self.size)

    def extra_repr(self):
        return "size=" + str(self.size)
    
class View(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args
        
    def forward(self, x):
        return x.view(*self.args)

    def extra_repr(self):
        r = ""
        s = ""
        for arg in self.args:
            r += s + str(arg)
            s = ", "
        return r
    
class Abs(nn.Module):
    def forward(self, x):
        return x.abs()

class Real(nn.Module):
    def forward(self, x):
        return x.real

class Imag(nn.Module):
    def forward(self, x):
        return x.imag

class Transpose(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args
    
    def forward(self, x):
        return x.transpose(*self.args)
    
    def extra_repr(self):
        r = ""
        s = ""
        for arg in self.args:
            r += s + str(arg)
            s = ", "
        return r

class Pad(nn.Module):
    def __init__(self, padding, mode="constant", pad_value=0):
        super().__init__()
        self.padding = padding
        self.mode = mode
        self.pad_value = pad_value

    def forward(self, x):
        return F.pad(x, self.padding, self.mode, self.pad_value)

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args
    
    def forward(self, x):
        return x.permute(*self.args)
    
    def extra_repr(self):
        r = ""
        s = ""
        for arg in self.args:
            r += s + str(arg)
            s = ", "
        return r
    
class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args
    
    def forward(self, x):
        return x.reshape(*self.args)

    def extra_repr(self):
        r = ""
        s = ""
        for arg in self.args:
            r += s + str(arg)
            s = ", "
        return r
    
class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode="nearest"):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        assert size is not None or scale_factor is not None
        
    def forward(self, x):
        return interpolate(x, self.size, self.scale_factor, self.mode)
        

    def extra_repr(self):
        a = f"size={self.size}, " if self.size is not None else ""
        b = f"scale_factor={self.scale_factor}, " if self.scale_factor is not None else ""
        return a + b + f"mode={self.mode}"
    
def interpolate(x, size=None, scale_factor=None, mode="bilinear"):
        if x.dtype in [torch.cfloat, torch.cdouble]:
            real = F.interpolate(x.real, size = size, scale_factor=scale_factor, mode=mode)
            imag = F.interpolate(x.imag, size = size, scale_factor=scale_factor, mode=mode)
            return torch.complex(real, imag)
        else:
            return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)

class Repeat(nn.Module):
    '''Repeats the input'''
    def __init__(self, *args):
        super().__init__()
        self.args = args
        
    def forward(self, x):
        return x.repeat(*self.args)

    def extra_repr(self):
        r = ""
        s = ""
        for arg in self.args:
            r += s + str(arg)
            s = ", "
        return r

class Autocontrast(nn.Module):
    def forward(self, x):
        return TF.autocontrast(x)

class RandomRotation(nn.Module):
    def __init__(self, angle=180, fill=0):
        super().__init__()
        self.angle = angle
        self.fill = fill
    
    def forward(self,x):
        angle = torch.rand(1).item() * self.angle * 2 - self.angle
        return TF.rotate(x, self.angle, fill=self.fill)

    def extra_repr(self):
        return f"angle={self.angle}, fill={self.fill}"
    
class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=3, sigma=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        
    def forward(self, x):
        y = []
        for i,p in enumerate(x):
            y.append(TF.gaussian_blur(p, self.kernel_size, sigma=self.sigma))
        return torch.stack(y)

    def extra_repr(self):
        return f"kernel_size={self.kernel_size}, sigma={self.sigma}"
    
class PoissonNoise(nn.Module):
    def __init__(self, well_capacity=10500):
        super().__init__()
        self.well_capacity = well_capacity
        
    def forward(self, x):
        discrete = x * self.well_capacity
        return (torch.normal(discrete, discrete.sqrt()) / self.well_capacity).clamp(0,1)

    def extra_repr(self):
        return f"well_capacity={self.well_capacity}"
    
class Clamp(nn.Module):
    def __init__(self, min=0, max=1):
        super().__init__()
        self.min = min
        self.max = max
    
    def forward(self, x):
        return x.clamp(self.min, self.max)
        
    def extra_repr(self):
        return f"min={self.min}, max={self.max}"
    
class Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim    
        
    def forward(self, x):
        return x.squeeze(self.dim)

    def extra_repr(self):
        return f"dim={self.dim}"
    
class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim    
    
    def forward(self, x):
        return x.unsqueeze(self.dim)
    
    def extra_repr(self):
        return f"dim={self.dim}"
class FailedCondition:
    def __init__(self, module, x):
        self.module = module
        self.x = x
    
class If(nn.Module):
    '''conditional branch executing a lambda (module, x): <bool>'''
    def __init__(self, cb, *args, on_true=None, on_false=None):
        super().__init__()
        if len(args) > 0: 
            assert on_true == None
            self.on_true = seq(args)
        else:
            self.on_true = seq(on_true)
        self.cb = cb
        self.on_false = seq(on_false)
        
    def forward(self, x):
        # Dodgy at best, but for distributed and torch.save
        cb = self.cb
        if isinstance(cb, str): cb = eval(cb)
        if callable(cb): b = cb(self, x)
        else: b = cb
        if b: 
            if self.on_true is None:
                return x
            else:
                return self.on_true(x)
        elif self.on_false is None:
            return FailedCondition(self, x)
        else:
            return self.on_false(x)
    
    def extra_repr(self):
        return str(self.cb)
    
class Else(nn.Module):
    '''conditional branch executing a lambda (module, x): <bool>'''
    def __init__(self, *args):
        super().__init__()
        self.body = seq(args)
        
    def forward(self, x):
        if isinstance(x, FailedCondition):
            return self.body(x.x)
        else: 
            return x.x
    
    
class Every(nn.Module):
    '''Will be called every <n>th pass through the model'''
    def __init__(self, n, *args):
        super().__init__()
        self.n = n
        self.i = 0
        self.body = seq(*args)
        
    def forward(self, x):
        self.i += 1
        if self.i == self.n:
            self.i = 0
            return self.body(x)
        else:
            return x

    def extra_repr(self):
        return f"n={self.n}"

class Lambda(nn.Module):
    def __init__(self, callback, *args, **kwargs):
        super().__init__()
        self.callback = callback
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        callback = self.callback
        if isinstance(callback, str) and callback.startswith("lambda"):
            callback = eval(callback)
        assert callable(callback)
        return callback(self, x, *self.args, **self.kwargs)
    
    def extra_repr(self):
        return f"{str(self.callback)}"
    
class Print(nn.Module):
    def __init__(self, value=None, end="\n"):
        super().__init__()
        self.value = value
        self.end = end
        
    def forward(self, x):
        value = self.value
        if isinstance(value, str) and value.startswith("lambda "):
            value = eval(value)
        if callable(value):
            value = value(self, x)
        value = str(value)
        print(value, end=self.end)
        return x

    def extra_repr(self):
        return f"{str(self.value)}, end={self.end}"

class Hyper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        for name, value in kwargs.items():
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)
            self.register_buffer(name, value)

    def extra_repr(self):
        s = []
        for name, b in self.named_buffers():
            if b.numel() == 1:
                s.append(f"{name}: {b.item()}")
            else:
                s.append(f"{name}: {b}")
        return "\n".join(s)

def plot(t, title="", width=20, cols=None, axis="off", normalize=False, **kwargs):
    while t.dim() < 4: t = t[None]
    n,c,h,w = t.shape
    if normalize:
        t = norm(t)
    t = disp(t, **kwargs)
    grid = make_grid(t, cols or n)
    plt.figure(figsize = (width, width * grid.shape[-2] / grid.shape[-1]))
    plt.title(title)
    plt.axis(axis)
    plt.imshow(grid.permute(1,2,0).cpu().numpy())
    plt.show()

def save(t, filename, cols=None, normalize=False, **kwargs):
    while t.dim() < 4: t = t[None]
    n,c,h,w = t.shape
    if normalize:
        t = norm(t)
    t = disp(t, dtype=torch.uint8, **kwargs)
    if n > 1:
        t = make_grid(t, cols or n)
    else:
        t = t[0]
    torchvision.io.write_png(t.cpu(), filename)

class Plot(nn.Module):
    def __init__(self, title="", width=20, cols=None, axis="off", **kwargs):
        super().__init__()
        self.title = title
        self.width = width
        self.axis = axis
        self.cols = cols
        self.kwargs = kwargs
        
    def forward(self, x):
        plot(x, self.title, self.width, self.cols, self.axis,**self.kwargs)
        return x

    def extra_repr(self):
        return f"title={self.title}, width={self.width}, axis={self.axis}, cols={self.cols}"

class Save(nn.Module):
    def __init__(self, filename, cols=None, **kwargs):
        super().__init__()
        self.filename = filename
        self.cols = cols
        self.kwargs = kwargs
        self.count = 0

    def forward(self, x):
        fn = self.filename.format(self.count)
        self.count += 1
        save(x, fn, self.cols, **self.kwargs)
        return x

    def extra_repr(self):
        return f"title={self.title}, width={self.width}, axis={self.axis}, cols={self.cols}"
    
class NoGrad(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.body = seq(args)
    
    def forward(self, x, *args):
        with torch.no_grad():
            return self.body(x, *args)

class Mean(nn.Module):
    def __init__(self, dim, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        
    def forward(self, x):
        y = x.mean(self.dim, keepdim=self.keepdim)
        return y

    def extra_repr(self):
        return f"{self.dim}, keepdim={self.keepdim}"
    
class Std(nn.Module):
    def __init__(self, dim, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
    def forward(self, x):
        y = x.std(self.dim, keepdim=self.keepdim)
        return y

    def extra_repr(self):
        return f"{self.dim}, keepdim={self.keepdim}"
    
class Log(nn.Module):
    def forward(self,x):
        return x.log()

class Exp(nn.Module):
    def forward(self,x):
        return x.exp()

class Conjugate(nn.Module):
    def forward(self, x):
        return x.conj()

    
class FFT(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        y = torch.fft.fft(x, dim=self.dim)
        return y

    def extra_repr(self):
        return f"dim={self.dim}"
    
class ChannelAttention(nn.Module):
    def __init__(self, num_channels, kernel_size=3, reduction=8, padding=None, bias=False, dropout=0,  act=nn.ReLU):
        super().__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.bias = bias
        self.padding = padding if padding is not None else kernel_size // 2
        rc = max(num_channels//reduction, 1)
        
        self.channel_max = nn.Sequential(
            nn.MaxPool2d(kernel_size, stride=1, padding=self.padding),
            nn.Conv2d(num_channels, rc, 1),
            act(),
            nn.Conv2d(rc, num_channels, 1, bias=bias))
        
        self.channel_avg = nn.Sequential(
            nn.MaxPool2d(kernel_size, stride=1, padding=self.padding),
            nn.Conv2d(num_channels, rc, 1),
            act(),
            nn.Conv2d(rc, num_channels, 1, bias=bias))
            
        self.dropout = nn.Dropout2d(dropout)
    
    def forward(self, x):
        a = self.dropout(x)
        a = self.channel_max(a) + self.channel_avg(a)
        return x * torch.sigmoid(a)

class SqueezeExcitation(nn.Module):
    def __init__(self, num_channels, reduction=None, derotate=True, act=nn.ReLU):
        super().__init__()
        self.reduction =  reduction or int(math.ceil(num_channels ** 0.5))
        self.squeeze = Conv2d(num_channels, num_channels // self.reduction, 1)
        self.excitation = Conv2d(num_channels // self.reduction, num_channels, 1)
        self.act = act()
        self.derotate = derotate
        self.pool = AdaptiveAvgPool2d(1)
        self.num_channels = num_channels
        
    def forward(self, x):
        if self.derotate:
            r = x[:,[0],:,:]
            r = r / (r.abs() + 1e-6)
            x = x * r.conj()
            
        s = self.pool(x)
        s = self.act(self.squeeze(s))
        s = torch.sigmoid(self.excitation(s))
    
        if self.derotate:
            x = x * r
        return x * s

    def extra_repr(self):
        r = f"num_channels={self.num_channels}, reduction={self.reduction}, act={self.act}"
        if self.derotate:
            r += ", derotate=True"
        return r

class SpatialAttention(nn.Module):
    def __init__(self, groups=1, kernel_size=3, padding=None, bias=False, dropout=0):
        super().__init__()
        self.groups = groups
        self.kernel_size = kernel_size
        self.bias = bias
        self.padding = padding if padding is not None else kernel_size // 2
        
        self.maxpool = nn.AdaptiveMaxPool3d((groups, None, None))
        self.avgpool = nn.AdaptiveAvgPool3d((groups, None, None))
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(2 * groups, 1, kernel_size, padding=self.padding, bias=bias)
        
    def forward(self, x):
        a = torch.cat((self.maxpool(x), self.avgpool(x)), dim=-3)
        a = self.dropout(a)
        a = self.conv(a)
        return x * torch.sigmoid(a)

