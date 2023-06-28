import math
import os 
import sys

from re import U
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial
from torch.nn.parameter import Parameter

from einops import rearrange, repeat

from megatron import mpu
from megatron.ops.fftconv import fftconv_func, _mul_sum
from megatron.model.init_functions import xavier_normal_init_method

def fftconv_heads_ref(k, ssm_kernel, D, q, v, head_dim=1, ssm_kernel_rev=None):

    seqlen = k.shape[-1]
    fft_size = 2 * seqlen
    kv = (rearrange(k, 'b (h d1) l -> b d1 1 h l', d1=head_dim)
            * rearrange(v, 'b (h d2) l -> b 1 d2 h l', d2=head_dim))  # b d1 d2 h l
    kv_f = torch.fft.rfft(kv.to(dtype=ssm_kernel.dtype), n=fft_size) / fft_size
    ssm_kernel_f = torch.fft.rfft(ssm_kernel, n=fft_size)  # h L+1
    if ssm_kernel_rev is not None:
        ssm_kernel_rev_f = torch.fft.rfft(ssm_kernel_rev, n=fft_size)  # h L+1
        ssm_kernel_f = ssm_kernel_f + ssm_kernel_rev_f.conj()

    y = torch.fft.irfft(kv_f * ssm_kernel_f, n=fft_size, norm='forward')[..., :seqlen]  # b d1 d2 h l
    out = y + kv * D.unsqueeze(-1)  # b d1 d2 h l
    q = rearrange(q, 'b (h d1) l -> b d1 1 h l', d1=head_dim)
    if head_dim > 1:
        out = _mul_sum(out, q)
        return rearrange(out, 'b d2 h l -> b (h d2) l').to(dtype=k.dtype)
    else:
        return rearrange(out * q, 'b 1 1 h l -> b h l').to(dtype=k.dtype)
    

def initialize_affine_weight_gpu(weight, init_method, partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU."""

    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.partition_stride = stride

    with mpu.get_cuda_rng_tracker().fork():
        init_method(weight)


def get_activation_from_str(act_str):
    if act_str.lower() == "relu":
        return nn.ReLU()
    elif act_str.lower() == "gelu":
        return nn.GELU()
    elif act_str.lower() == "silu":
        return nn.SiLU()
    elif act_str.lower() == "identity":
        return nn.Identity()
    else:
        raise ValueError(f"Activation {act_str} not supported.")


class Sin(nn.Module):
    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = (
            nn.Parameter(w * torch.ones(1, dim))
            if train_freq
            else w * torch.ones(1, dim)
        )

    def forward(self, x):
        return torch.sin(self.freq * x)


class PositionalEmbedding(nn.Module):
    def __init__(self, emb_dim: int, seq_len: int, **kwargs):
        """Complex exponential positional embeddings for Hyena filters."""
        super().__init__()

        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        self.register_buffer("t", torch.linspace(0, 1, self.seq_len)[None, :, None])  # 1, L, 1

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len  # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        #self.z = nn.Parameter(torch.cat([self.t, z.real, z.imag], dim=-1))
        # fix to non-learnable
        z = torch.cat([self.t, z.real, z.imag], dim=-1)
        self.register_buffer("z", z)
        

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class RandomFourierPositionalEmbedding(torch.nn.Module):
    def __init__(self, emb_dim: int, seq_len: int, omega_0: float, use_bias: bool=False, **kwargs):
        if emb_dim % 2 != 0:
            raise ValueError(f"emb_dim must be even. Current {emb_dim}")
        super().__init__()

        linear_out_channels = emb_dim // 2
        self.linear = torch.nn.Linear(
            in_features=1, out_features=linear_out_channels, bias=use_bias
        )
        # initialize with xavier normal rescaled by 0.02
        torch.nn.init.xavier_normal_(self.linear.weight, gain=0.02)

        # Initialize:
        self.linear.weight.data.normal_(0.0, 2 * torch.pi * omega_0)
        if use_bias:
            torch.nn.init.constant_(self.linear.bias, 0.0)
        
        t = torch.linspace(-1, 1, seq_len)[None, :, None]
        self.register_buffer("t", t)

    def forward(self, L):
        out = self.linear(self.t[:, :L])
        return torch.cat([torch.cos(out), torch.sin(out)], dim=-1), (self.t + 1) / 2



class ParallelExponentialModulation(nn.Module):
    def __init__(
        self,
        neox_args,
        d_model,
        hidden_size_per_partition,
        mp_rank,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        shift: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct

        self.weight = Parameter(
            torch.empty(1, 1, hidden_size_per_partition,
                        dtype=neox_args.params_dtype)
        )

        self.weight.model_parallel = True
        self.weight.partition_dim = 2
        self.weight.partition_stride = 1

        master_weight = torch.linspace(min_decay, max_decay, d_model)[None, None].to(neox_args.params_dtype)

        weight_list = torch.split(master_weight, hidden_size_per_partition, dim=-1)
        rank = mpu.get_model_parallel_rank()
        world_size = mpu.get_model_parallel_world_size()
        my_weight_list = weight_list[rank::world_size]

        with torch.no_grad():
            torch.cat(my_weight_list, dim=self.weight.partition_dim, out=self.weight)

    def forward(self, t, x):
        decay = torch.exp(-t * self.weight.abs())
        x = x * (decay + self.shift)
        return x


class ParallelModalHyenaFilter(nn.Module):
    def __init__(
        self,
        neox_args,
        d_model: int,
        order: int,
        mimo: bool = False,
        theta_max: float = 2 * torch.pi,
    ):
        """
            Filters of SSM in modal form. Improved diagonal SSM parametrization using insights from LRU.
            We parametrize the poles and residues, i.e. the matrices A and C, of the state-space model. B is fixed to a one-vector and D is learned as a bias.
            The poles are parametrized as complex numbers in polar form, i.e. r * exp(i * theta), where r is the radius and theta the phase.
            Stability of the system in encouraged by init but not enforced.
            Args:
                d_model: dimension of the input and output (number of channels)
                order: order of the filter (number of states of underlying state-space model)
                mimo: whether the filter is MIMO or SISO
                r_min: minimum radius of the poles (init hyperparameter)
                r_max: maximum radius of the poles (init hyperparameter)
                theta_max: maximum phase of the poles (init hyperparameter)
        """
        super().__init__()
        self.order = order
        self.d_model = d_model
        self.mimo = mimo

        # Init poles and residues
        self.register_parameter("r", nn.Parameter(torch.ones(order // 2, d_model)))
        self.register_parameter("theta", nn.Parameter(torch.ones(order // 2, d_model)))

        if mimo:
            # TODO: implement MIMO case where R is a tensor of shape (order // 2, d_model, d_model)
            raise NotImplementedError
        self.register_parameter("R_re", nn.Parameter(torch.ones(order // 2, d_model)))
        self.register_parameter("R_im", nn.Parameter(torch.ones(order // 2, d_model)))

        self.register_parameter("h_0", nn.Parameter(torch.ones(1, d_model)))
        r_min, r_max = neox_args.hyena_filter_r_min, neox_args.hyena_filter_r_max
        self._init_params(r_max, r_min, theta_max)

    def _init_params(self, r_max, r_min, theta_max):
        # Init poles distributed uniformly of ring of the complex plane
        # between r_min and r_max and phase between 0 and theta_max
        u1 = torch.rand(self.order // 2, self.d_model)
        u2 = torch.rand(self.order // 2, self.d_model)
        self.r.data = r_min + (r_max - r_min) * u1
        self.theta.data = theta_max * u2
        # Init residues with Glorot initialization
        self.R_re.data = torch.randn(self.order // 2, self.d_model) * math.sqrt(
            2 / self.order
        )
        self.R_im.data = torch.randn(self.order // 2, self.d_model) * math.sqrt(
            2 / self.order
        )

    def _get_poles_and_residues(self):
        # poles
        p = self.r * torch.exp(1j * self.theta)
        # residues
        R = self.R_re + 1j * self.R_im
        return p, R

    def compute_filter(self, L):
        p, R = self._get_poles_and_residues()
        t = torch.arange(L-1).unsqueeze(1).unsqueeze(2).to(p)
        h = torch.sum(R * p**t, dim=1).real
        return h

    def forward(self, L, *args, **kwargs):
        # evaluate filter for t = 1, ..., L
        h = self.compute_filter(L)
        # stack h_0 to the beginning of the filter
        h = torch.cat([self.h_0.to(h), h], dim=0)
        h = rearrange(h, 'L D -> D L')
        return h

class ParallelHyenaFilter(nn.Module):
    def __init__(
        self,
        neox_args,
        init_method,
        d_model,
        emb_dim=3,  # dim of input to MLP, augments with positional encoding
        order=16,  # width of the implicit MLP
        seq_len=1024,
        w=1,  # frequency of periodic activations
        omega_0=1,  # frequency of positional embeddings
        wd=0,  # weight decay of kernel parameters
        num_inner_mlps=2,
        modulate: bool = True,
        normalized=False,
        **kwargs,
    ):
        """
        Implicit long filter with modulation.

        Args:
            d_model: number of channels in the input
            emb_dim: dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP

        Note:
            filter_dropout is not implemented
        """
        super().__init__()
        self.d_model = d_model
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.modulate = modulate

        world_size = mpu.get_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(d_model, world_size)

        self.act = Sin(dim=order, w=w)

        if neox_args.hyena_pos_emb == "fourier_fixed":
            assert (
                emb_dim % 2 != 0 and emb_dim >= 3
            ), "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
            self.pos_emb = PositionalEmbedding(emb_dim, seq_len)

        elif neox_args.hyena_pos_emb == "random_fourier":
            self.pos_emb = RandomFourierPositionalEmbedding(emb_dim, seq_len, omega_0)

        # uses a variable number of inner linear layers
        self.implicit_filter = nn.Sequential(
            nn.Linear(emb_dim, order),
            self.act,
        )
        for i in range(num_inner_mlps):
            self.implicit_filter.append(nn.Linear(order, order, bias=True))
            self.implicit_filter.append(self.act)
        
        final_init_method = torch.nn.init.xavier_normal_
        
        
        # final linear layer

        self.final_filter = nn.Linear(order, d_model, bias=False)
        torch.nn.init.xavier_normal_(self.final_filter.weight, gain=1)
        # self.final_filter = mpu.ColumnParallelLinear(
        #     neox_args=neox_args,
        #     input_size=order,
        #     output_size = d_model,
        #     gather_output=False,
        #     init_method = init_method,
        #     bias = False
        # )
        fast_decay_pct, slow_decay_pct = neox_args.hyena_filter_fast_decay, neox_args.hyena_filter_slow_decay
        self.modulation = ParallelExponentialModulation(
            neox_args, d_model, self.hidden_size_per_partition, mpu.get_model_parallel_rank(), 
            fast_decay_pct=fast_decay_pct, slow_decay_pct=slow_decay_pct, **kwargs)

        self.normalized = normalized

    def forward(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        h = self.final_filter(h)

        if self.modulate:
            h = self.modulation(t, h)

        if self.normalized:
            h = h / torch.norm(h, dim=-1, p=1, keepdim=True)
        h = rearrange(h, '1 L D -> D (1 L)')
        return h



class ParallelHyenaConv(nn.Module):
    """
    Inner action for a parallel Hyena Conv (replacing attention).
    Testing some new improvements...

    Inputs: Q, K, V
    Operation: 1D Conv on each Q, K, V (independently)
    Long Conv(Q * K) * V, independently on each H
    """

    def __init__(
        self,
        neox_args,
        init_method,
        layer_number=0,
    ):
        super().__init__()

        self.layer_number = layer_number
        self.fp16 = neox_args.precision == "fp16"
        self.bf16 = neox_args.precision == "bfloat16"
        self.act = get_activation_from_str(neox_args.gating_act) 

        self.use_hyena_filter = neox_args.use_hyena_filter
        self.use_fast_heads = neox_args.use_fast_heads
        self.use_slow_heads = neox_args.use_slow_heads

        self.model_parallel_size = mpu.get_model_parallel_world_size()
        self.model_parallel_rank = mpu.get_model_parallel_rank()
        
        world_size = mpu.get_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(neox_args.hidden_size, world_size)

        self.L = neox_args.seq_length
        self.num_heads = neox_args.num_heads
        self.head_dim = self.hidden_size_per_partition //  self.num_heads
        self.short_conv_L = neox_args.short_conv_L


        self.alpha1 = nn.Parameter(
            torch.empty(
                1,
                self.hidden_size_per_partition,
                1,
                device=torch.cuda.current_device(),
                dtype=neox_args.params_dtype,
            )
        )
        self.alpha2 = nn.Parameter(
            torch.empty(
                1,
                self.hidden_size_per_partition,
                1,
                device=torch.cuda.current_device(),
                dtype=neox_args.params_dtype,
            )
        )
        
        initialize_affine_weight_gpu(self.alpha1, torch.nn.init.ones_, 1)
        initialize_affine_weight_gpu(self.alpha2, torch.nn.init.ones_, 1)

        self.short_conv_weight = nn.Parameter(
            torch.empty(
                3, # Q K V
                self.hidden_size_per_partition,
                1,
                self.short_conv_L,
                device=torch.cuda.current_device(),
                dtype=neox_args.params_dtype,
            )
        )

        # self.short_conv_bias = nn.Parameter(
        #     torch.empty(
        #         3,
        #         self.hidden_size_per_partition,
        #         device=torch.cuda.current_device(),
        #         dtype=neox_args.params_dtype,
        #     )
        # )
        # self.short_conv_bias.model_parallel = True
        # self.short_conv_bias.partition_dim = 1
        # self.short_conv_bias.stride = 1

        # initialize_affine_weight_gpu(self.short_conv_weight, init_method, partition_dim=1)

        if neox_args.hyena_filter_cls == "implicit":
            self.filter = ParallelHyenaFilter(
                neox_args,
                init_method,
                d_model=self.num_heads,
                emb_dim=neox_args.hyena_filter_emb_dim,
                order=neox_args.hyena_filter_order,
                num_inner_mlps=neox_args.hyena_filter_num_inner_mlps,
                seq_len=self.L,
                w=neox_args.hyena_filter_w,
                omega_0=neox_args.hyena_filter_omega_0,
            )
        else:
            self.filter = ParallelModalHyenaFilter(
                neox_args,
                self.num_heads,
                neox_args.hyena_filter_order,
                mimo=False,
                theta_max=2 * torch.pi,
            )

        if self.use_slow_heads:
            self.long_conv_bias = nn.Parameter(
                torch.empty(
                    self.num_heads,
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
            )
        else:
            self.long_conv_bias = nn.Parameter(
                torch.empty(
                    self.hidden_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
            )

        self.long_conv_bias.model_parallel = True
        self.long_conv_bias.partition_dim = 0
        self.long_conv_bias.stride = 1
    
    def forward(self, query_layer, key_layer, value_layer):
        
        # input sizes: [sq, b, np, hn]
        # seqlen, batch, tensor parallel, hidden size per tensor parallel
        np = query_layer.shape[-2]

        query = rearrange(query_layer, 'sq b np hn -> b (np hn) sq')
        key = rearrange(key_layer, 'sq b np hn -> b (np hn) sq')
        value = rearrange(value_layer, 'sq b np hn -> b (np hn) sq')
        q,k,v = query[...,:self.L], key[...,:self.L], value[...,:self.L]

        filter = self.filter(self.L)


        if self.use_fast_heads:
            filter = filter.repeat_interleave(self.head_dim, dim=0) 
            z = k * v

            head_dim = self.head_dim
            seqlen = k.shape[-1]
            fft_size = 2 * seqlen

            k_heads = rearrange(k, 'b (h d1) l -> b d1 1 h l', d1=head_dim)
            
            v_heads = rearrange(v, 'b (h d2) l -> b 1 d2 h l', d2=head_dim)
            kv = _mul_sum(k_heads, v_heads) / math.sqrt(head_dim) # b d2 h l
            
            
            kv = rearrange(kv, 'b d2 h l -> b (h d2) l')

            # fft does not support fp16
            filter_f = torch.fft.rfft(filter.to(torch.float32), n=fft_size) / fft_size
            kv_f = torch.fft.rfft(kv.to(torch.float32), n=fft_size) 
            z = torch.fft.irfft(kv_f * filter_f, n=fft_size, norm='forward')[..., :seqlen]  # b d1 d2 h l
            z = z.to(dtype=k.dtype)
            out = z + kv * self.long_conv_bias.unsqueeze(-1)  # b d1 d2 h l
            
            q_heads = rearrange(q, 'b (h d1) l -> b d1 1 h l', d1=head_dim)
            z_heads = rearrange(z, 'b (h d2) l -> b 1 d2 h l', d2=head_dim)
            z = _mul_sum(q_heads, z_heads) / math.sqrt(head_dim) # b d2 h l
            z = rearrange(z, 'b d1 h l -> b (h d1) l')

            z = q * z

        elif self.use_slow_heads:
            z = fftconv_heads_ref(
                v.to(torch.float32),
                filter.to(torch.float32),
                self.long_conv_bias.to(torch.float32),
                v=k,
                head_dim=self.head_dim,
                q=q,
            )
            z = z.to(value.dtype)
        else:
            filter = filter.repeat_interleave(self.head_dim, dim=0) 
            z = k * v
            with torch.autocast("cuda"):
                z = fftconv_func(z.to(torch.float32), filter.to(torch.float32), self.long_conv_bias, None, gelu=False)
                z = z.to(v.dtype)

            # TODO: update
            if q.size(-1) != z.size(-1):
                z = q * z[..., -1:]
            else:
                z = q * z
            
        z = rearrange(z, 'b (np hn) sq -> b np sq hn', np=np)
        
        # if self.layer_number == 0: print('z', z[0,0,:,0])
        
        return z



@torch.jit.script 
def mul_sum2(q, y):
    return (q * y).sum(dim=2)


@torch.jit.script
def work_prefix_sum(z):        
    L = int(z.shape[-1])
    iters = int(torch.log2(torch.tensor(L)).ceil())

    for stage in range(iters):
        delta = int(2 ** stage)
        z[..., delta:] += z[..., :-delta].clone()
    return z
    
class ParallelHyenaConv2(nn.Module):
    def __init__(self, neox_args, init_method):
        super().__init__()
        self.fp16 = neox_args.precision == "fp16"
        self.bf16 = neox_args.precision == "bfloat16"

        self.model_parallel_size = mpu.get_model_parallel_world_size()
        self.model_parallel_rank = mpu.get_model_parallel_rank()
        world_size = mpu.get_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(neox_args.hidden_size, world_size)

        self.L = neox_args.seq_length
        self.num_heads = neox_args.num_heads
        self.head_dim = self.hidden_size_per_partition //  self.num_heads
        self.short_conv_L = neox_args.short_conv_L

        self.filter = ParallelHyenaFilter(
            neox_args,
            init_method,
            d_model=self.num_heads,
            emb_dim=neox_args.hyena_filter_emb_dim,
            order=neox_args.hyena_filter_order,
            num_inner_mlps=neox_args.hyena_filter_num_inner_mlps,
            seq_len=self.L,
            w=neox_args.hyena_filter_w,
            omega_0=neox_args.hyena_filter_omega_0,
        )
    
    def forward(self, query_layer, key_layer, value_layer):        
        # input sizes: [sq, b, np, hn]
        np = value_layer.shape[-2]

        query = rearrange(query_layer, 'sq b np hn -> b (np hn) sq')
        key = rearrange(key_layer, 'sq b np hn -> b (np hn) sq')
        value = rearrange(value_layer, 'sq b np hn -> b (np hn) sq')
        q, k, v = query[...,:self.L], key[...,:self.L], value[...,:self.L]

        h = self.filter(self.L)
        h = rearrange(h, 'h l -> 1 h 1 1 l')
        
        q = rearrange(q, 'b (h d1) l -> b h d1 1 l', d1=self.head_dim)
        k = rearrange(k, 'b (h d1) l -> b h d1 1 l', d1=self.head_dim)
        v = rearrange(v, 'b (h d2) l -> b h 1 d2 l', d2=self.head_dim)
        
        kv = k * v
        y = mul_sum2(work_prefix_sum(kv * h), q) / math.sqrt(self.head_dim)
        y = rearrange(y, 'b h d2 l -> b (h d2) l')

        return rearrange(y, 'b (np hn) sq -> b np sq hn', np=np)




