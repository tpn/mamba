# Copyright (c) 2023, Tri Dao, Albert Gu.

from textwrap import dedent

import torch
import torch.nn.functional as F
from mamba_ssm.utils.torch import custom_bwd, custom_fwd
import os

# Disable PyTorch dynamo/compiler to avoid memory issues
if False:
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        if hasattr(torch, "_inductor"):
            torch._inductor.config.triton.cudagraphs = False
        # Set to 'eager' to disable compilation completely
        if hasattr(torch, "_dynamo"):
            torch._dynamo.config.cache_size_limit = 0
    except ImportError:
        pass

    # Force high precision for matmul operations
    torch.set_float32_matmul_precision('high')

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
    import causal_conv1d_cuda
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_cuda = None

from mamba_ssm.ops.triton.layer_norm import _layer_norm_fwd

# Get the scan implementation choice from environment variable
SCAN_OPTION = os.environ.get("MAMBA_SCAN_OPTION", "cuda")

is_torch_cuda_parallel_available = False
is_cuda_parallel_available = False
# Import the appropriate scan implementation based on SCAN_OPTION
if SCAN_OPTION == "cuda":
    import selective_scan_cuda
elif SCAN_OPTION == "cuda2":
    import selective_scan2_cuda as selective_scan_cuda
elif SCAN_OPTION in ["ref"]:
    pass
elif SCAN_OPTION in ["torch"]:
    pass
elif SCAN_OPTION in ["torch-cudaparallel"]:
    try:
        from torch._higher_order_ops.cuda_parallel_associative_scan import (
            associative_scan,
        )
        is_torch_cuda_parallel_available = True
    except ImportError:
        raise RuntimeError("torch._higher_order_ops.cuda_parallel_associative_scan is not available.")
elif SCAN_OPTION in ["cudaparallel"]:
    #import ipdb; ipdb.set_trace()
    try:
        import cuda
        import cuda.parallel
        import cuda.parallel.experimental
        import cuda.parallel.experimental.algorithms
        is_cuda_parallel_available = True
    except ImportError:
        raise RuntimeError("cuda.parallel.experimental.algorithms is not available.")
else:
    import selective_scan_cuda

class SelectiveScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False):
        if SCAN_OPTION == "ref":
            # For reference implementation, we don't use this autograd function
            raise NotImplementedError("Reference implementation doesn't use SelectiveScanFn")

        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        else:
            ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if SCAN_OPTION == "ref":
            # For reference implementation, we don't use this autograd function
            raise NotImplementedError("Reference implementation doesn't use SelectiveScanFn")

        if not ctx.has_z:
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        # Here we just pass in None and dz will be allocated in the C++ code.
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
            False  # option to recompute out_z, not used here
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC,
                dD if D is not None else None,
                dz,
                ddelta_bias if delta_bias is not None else None,
                None,
                None)


def rms_norm_forward(
    x,
    weight,
    bias,
    eps=1e-6,
    is_rms_norm=True,
):
    # x (b l) d
    if x.stride(-1) != 1:
        x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    y = _layer_norm_fwd(
        x, weight, bias, eps, None, residual_dtype=None, is_rms_norm=is_rms_norm
    )[0]
    # y (b l) d
    return y


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                     return_last_state=False):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """
    if SCAN_OPTION == "ref":
        return selective_scan_ref(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)
    elif SCAN_OPTION == "torch":
        return selective_scan_torch(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)
    elif SCAN_OPTION == "cudaparallel":
        return selective_scan_cudaparallel(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)
    else:
        return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)

SEEN_SCAN_REF = -3

def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    global SEEN_SCAN_REF
    if SEEN_SCAN_REF == -3:
        msg = dedent("""
            selective_scan_ref(
                u.shape: {u.shape} (dtype: {u.dtype}),
                delta.shape: {delta.shape} (dtype: {delta.dtype}),
                A.shape: {A.shape} (dtype: {A.dtype}),
                B.shape: {B.shape} (dtype: {B.dtype}),
                C.shape: {C.shape} (dtype: {C.dtype}),
                D.shape: {D.shape} (dtype: {D.dtype}),
                z.shape: {z.shape} (dtype: {z.dtype}),
                delta_bias.shape: {delta_bias.shape} (dtype: {delta_bias.dtype}),
                delta_softplus: {delta_softplus},
                return_last_state: {return_last_state}
            )
        """.format(u=u, delta=delta, A=A, B=B, C=C, D=D, z=z,
                   delta_bias=delta_bias, delta_softplus=delta_softplus,
                   return_last_state=return_last_state))
        print(msg)
        SEEN_SCAN_REF = -2
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    if SEEN_SCAN_REF == -2:
        msg = dedent(f'''
            selective_scan_ref: pre-loop (for i in range({u.shape[2]}))
                u.shape: {u.shape} (dtype: {u.dtype})
                deltaA.shape: {deltaA.shape} (dtype: {deltaA.dtype})
                deltaB_u.shape: {deltaB_u.shape} (dtype: {deltaB_u.dtype})
                x.shape: {x.shape} (dtype: {x.dtype})
            ''')
        print(msg)
        SEEN_SCAN_REF = -1
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    if SEEN_SCAN_REF == -1:
        msg = dedent(f'''
            selective_scan_ref: post-loop (for i in range({u.shape[2]}))
                len(out): {len(out)}
                len(last_state): {len(last_state)}
            ''')
        print(msg)
        SEEN_SCAN_REF = None
    return out if not return_last_state else (out, last_state)

def selective_scan_ref_simple(u, delta, A, B, C, D=None, z=None,
                       delta_bias=None, delta_softplus=False,
                       return_last_state=False):
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    delta = delta + delta_bias[..., None].float()
    delta = F.softplus(delta)
    batch = u.shape[0]
    dim = A.shape[0] # 5120 for 2.8b
    dstate = A.shape[1] # 16 for 2.8b
    assert B.dim() >= 3, B.dim()
    assert C.dim() >= 3, C.dim()
    B = B.float()
    C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    # Perform the discretization steps.
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
    last_state = None

    for i in range(u.shape[2]): # prompt length
        # The prefix scan:
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]

        y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        ys.append(y)
    # ys = 1000 elem array of torch.Size([16, 5120])
    y = torch.stack(ys, dim=2) # (batch dim L)
    # y.shape = torch.Size([16, 5120, 1000])
    out = rearrange(D, "d -> d 1")
    out = y + u * out
    out = out * F.silu(z)
    out = out.to(dtype=torch.float16)
    return (out, last_state)


def selective_scan_torch(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    from torch._higher_order_ops.associative_scan import (
        associative_scan,
    )

    def s5_operator(x, y):
        A_i, Bu_i = x
        A_j, Bu_j = y
        return A_j * A_i, A_j * Bu_i + Bu_j

    use_associative_scan = True

    def _selective_scan_torch(u, delta, A, B, C, D, z, delta_bias, delta_softplus):
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()

        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state_scan = None

        if use_associative_scan:
            _, x_scan = associative_scan(s5_operator, (deltaA, deltaB_u), 2, combine_mode=combine_mode)
            if not is_variable_C:
                raise NotImplementedError('This feature is not yet implemented!')
                y = torch.einsum('bdn,dn->bd', x_scan, C)
            else:
                if C.dim() == 3:
                    y_scan = torch.einsum('bdsn,bns->bds', x_scan, C)
                else:
                    raise NotImplementedError('This feature is not yet implemented!')
                    y = torch.einsum('bdns,bdns->bds', x_scan, C)
            last_state_scan = x_scan[:, :, -1, :]
            if y_scan.is_complex():
                y_scan = y_scan.real * 2
        else:
            pass

        out = y_scan if D is None else y_scan + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)

        return out, last_state_scan

    #comment = '_compile'
    # combine_mode = 'generic'
    combine_mode = 'pointwise'

    if 'compile' in comment:
        _selective_scan_torch_cmp = torch.compile(_selective_scan_torch, fullgraph=True, mode='reduce-overhead')
    else:
        _selective_scan_torch_cmp = _selective_scan_torch

    out, last_state = _selective_scan_torch_cmp(u, delta, A, B, C, D, z, delta_bias, delta_softplus)

    return out if not return_last_state else (out, last_state)

# torch cuda parallel
# ---------------------------------------------------------------------
#  ── 1.  point‑wise associative operator used by the scan  ─────────────────
# ---------------------------------------------------------------------
def s5_operator_2(x, y):
    """
    x, y are tuples (A_i, Bu_i) and (A_j, Bu_j)
    The binary op composes the two affine maps:
        h  ↦  A_i h + Bu_i     followed by     h ↦  A_j h + Bu_j
    returning the composed (A_j A_i ,  A_j Bu_i + Bu_j)
    """
    A_i, Bu_i = x
    A_j, Bu_j = y
    return A_j * A_i, A_j * Bu_i + Bu_j


# ---------------------------------------------------------------------
#  ── 2.  selective‑scan with CUDA‑parallel inclusive scan  ────────────
# ---------------------------------------------------------------------
def selective_scan_torch_cudaparallel(
    u, delta, A, B, C,
    D=None, z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
    combine_mode: str = "pointwise",  # forwarded to associative_scan
    ):
    """
    CUDA‑parallel version of the selective scan.
    Shape conventions (all real tensors, fp16/fp32 mix is fine)::

        u, delta, z     : (B, D, L)
        A               : (D, N)
        B, C            : (B, N, L)               # "variable‑B / variable‑C" path
        D               : (D,)
    """
    # ----  pre‑processing ----------------------------------------------------
    dtype_in = u.dtype
    u      = u.float()                 # do math in fp32 for numerical safety
    delta  = delta.float()
    B      = B.float()
    C      = C.float()

    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    Bsz, Dim, L = u.shape
    Dstate      = A.shape[1]           # N

    # ----  discretise A and (B ∘ u) -----------------------------------------
    # deltaA, deltaB_u : (B, D, L, N)
    deltaA   = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)

    # ----  parallel prefix scan over sequence dimension ---------------------
    #       (B, D, L, N)  ←  associative_scan( s5_operator, ... , dim=2 )
    #
    from torch._higher_order_ops.cuda_parallel_associative_scan import (
        associative_scan,
    )

    (_,  x_scan) = associative_scan(           # we only need the composed Bu
        s5_operator_2,
        (deltaA, deltaB_u),                    # two‑tuple input
        dim     = 2,                           # scan over sequence dimension
        reverse = False,
        combine_mode = combine_mode,           # "pointwise" is fastest
    )
    # x_scan : (B, D, L, N)
    last_state = x_scan[..., -1, :]            # (B, D, N)

    # ----  read‑out  y_t  ----------------------------------------------------
    # C is (B, N, L)          → einsum over N
    y_scan = torch.einsum("bdsn,bns->bds", x_scan, C)   # (B, D, L)

    # ----  residual / gating / cast back ------------------------------------
    out = y_scan if D is None else y_scan + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype_in)

    return (out, last_state) if return_last_state else out

# raw cuda.parallel
if not is_cuda_parallel_available:
    raise ImportError('cuda.parallel is not available')

# ---------------------------------------------------------------------------
#  selective_scan_cudaparallel.py
# ---------------------------------------------------------------------------
# N.B. Was experiencing some inexplicable issues with the import order, hence
#      this insanity below.

print('About to import cupy...')
import cupy as cp
print('cupy imported successfully')
print('About to import numba...')
import numba
print('numba imported successfully')
print('About to import algorithms...')
import cuda.parallel.experimental.algorithms as algorithms
print('algorithms imported successfully')
print('About to import iterators...')
import cuda.parallel.experimental.iterators as iterators
print('iterators imported successfully')
print('About to import make_ndarray_iterator...')
from cuda.parallel.experimental.iterators._strided import make_ndarray_iterator
print('make_ndarray_iterator imported successfully')
#print('About to import zip_iterator...')
#from cuda.parallel.experimental.iterators import zip_iterator
#print('zip_iterator imported successfully')

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def torch_to_cupy(t: torch.Tensor) -> cp.ndarray:
    """Zero‑copy view of a torch CUDA tensor as a CuPy array."""
    return cp.from_dlpack(torch.utils.dlpack.to_dlpack(t.contiguous()))

def cupy_to_torch(a: cp.ndarray) -> torch.Tensor:
    """Zero‑copy view of a CuPy array as a torch CUDA tensor."""
    return torch.utils.dlpack.from_dlpack(a.toDlpack())

# ─────────────────────────────────────────────────────────────────────────────
# device‑side combine operator  (A, Bu)  ∘  (A', Bu')
# ─────────────────────────────────────────────────────────────────────────────
if False:
    _pair_dtype = cp.dtype([("A",  "f4"),
                            ("Bu", "f4")])
    _pair_nbtype = numba.from_dtype(_pair_dtype)

    @numba.cuda.jit(device=True)
    def _s5_op(x, y):
        """
        Compose two affine maps  h ↦ A·h + Bu  and  h ↦ A'·h + Bu'
        returning  (A'A,  A'·Bu + Bu')
        """
        out = _pair_nbtype()
        out.A  = y.A * x.A
        out.Bu = y.A * x.Bu + y.Bu
        return out

pair_type = numba.types.Record.make_c_struct([
    ("A",  numba.types.float32),
    ("Bu", numba.types.float32),
])

@numba.cuda.jit(device=True)
def s5_op(x, y):
    """
    x,y : pair_type
    out : pair_type   =  (y.A * x.A ,  y.A * x.Bu + y.Bu)
    """
    out = numba.cuda.local.array(1, dtype=pair_type)[0]  # create empty record
    out.A  = y.A * x.A
    out.Bu = y.A * x.Bu + y.Bu
    return out

# Define the scan operator directly in the device code
# This is a new version that works with separate A and Bu arrays
@numba.cuda.jit(device=True)
def s5_op_separate(A_Bu_state, next_inputs):
    A_prev, Bu_prev = A_Bu_state
    A_next, Bu_next = next_inputs

    # Implement (A_j * A_i, A_j * Bu_i + Bu_j)
    new_A = A_next
    new_Bu = A_next * Bu_prev + Bu_next

    return (new_A, new_Bu)

# ─────────────────────────────────────────────────────────────────────────────
# main routine
# ─────────────────────────────────────────────────────────────────────────────
def selective_scan_cudaparallel(
    u, delta, A, B, C,
    D=None, z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
    ):
    """
    Same signature as selective_scan_ref, but the prefix scan along sequence
    length L is executed by cuda.parallel's inclusive_scan.
    Only the real‑valued (fp16 / fp32) code‑path is included for brevity.
    """
    # -------- pre‑processing -------------------------------------------------
    dtype_in = u.dtype
    u      = u.float()
    delta  = delta.float()
    B      = B.float()
    C      = C.float()

    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    Bsz, Dim, L = u.shape
    N           = A.shape[1]                    # SSM state dim per channel

    # -------- discretisation  ----------------------------------------------
    # deltaA, deltaB_u  :  (B, D, L, N)
    deltaA   = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)

    # -------- move to CuPy (zero‑copy via DLPack) ---------------------------
    dA_cp  = torch_to_cupy(deltaA)
    dBu_cp = torch_to_cupy(deltaB_u)

    # -------- reshape so each (b,d,n) stream is a contiguous 1‑D segment ---
    #
    #   (B, D, L, N)  →  (G, L)   where  G = B·D·N
    #
    G = Bsz * Dim * N
    dA_flat  = dA_cp.transpose(0, 1, 3, 2).reshape(G, L)
    dBu_flat = dBu_cp.transpose(0, 1, 3, 2).reshape(G, L)

    # ------------------------------------------------------------------
    # 1.  Setup for the CUDA parallel scan
    # ------------------------------------------------------------------

    A_in = dA_flat    # (G, L) float32
    Bu_in = dBu_flat  # (G, L) float32
    A_out = cp.empty_like(A_in)
    Bu_out = cp.empty_like(Bu_in)


    # Initialize values
    init_val = (1.0, 0.0)  # Identity value for (A, Bu)

    # -------- run cuda.parallel inclusive_scan using s5_op_separate ---------------
    for g in range(G):
        # Create iterators for this row
        A_in_it = make_ndarray_iterator(
            A_in[g],
            (0,),
            iterators._iterators.IteratorIO.INPUT,
            "A_in",
        )
        Bu_in_it = make_ndarray_iterator(
            Bu_in[g],
            (0,),
            iterators._iterators.IteratorIO.INPUT,
            "Bu_in",
        )
        A_out_it = make_ndarray_iterator(
            A_out[g],
            (0,),
            iterators._iterators.IteratorIO.OUTPUT,
            "A_out",
        )
        Bu_out_it = make_ndarray_iterator(
            Bu_out[g],
            (0,),
            iterators._iterators.IteratorIO.OUTPUT,
            "Bu_out",
        )

        # Create zip iterators to handle (A, Bu) pairs
        input_it = zip_iterator([A_in_it, Bu_in_it], "input")
        output_it = zip_iterator([A_out_it, Bu_out_it], "output")

        # Use CUDA parallel inclusive scan
        scanner = algorithms.inclusive_scan(
            input_it, output_it, s5_op_separate, init_val
        )

        # Run the scan
        tmp_sz = scanner(None, input_it, output_it, L, init_val)
        tmp_buf = cp.empty((tmp_sz,), dtype=cp.uint8)
        scanner(tmp_buf, input_it, output_it, L, init_val)

    # ------------------------------------------------------------------
    # 2.  After the scan, use the Bu component
    # ------------------------------------------------------------------
    Bu_scan_t = cupy_to_torch(Bu_out)  # Convert to torch tensor

    # reshape back to (B, D, L, N)
    x_scan     = Bu_scan_t.reshape(Bsz, Dim, N, L).permute(0, 1, 3, 2)
    last_state = x_scan[:, :, -1, :]

    # -------- read‑out & residual path -------------------------------------
    y_scan = torch.einsum("bdsn,bns->bds", x_scan, C)      # (B,D,L)

    out = y_scan if D is None else y_scan + u * rearrange(D.float(), "d -> d 1")
    if z is not None:
        out = out * F.silu(z.float())
    out = out.to(dtype_in)

    return (out, last_state) if return_last_state else out


class MambaInnerFn(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                out_proj_weight, out_proj_bias,
                A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
                C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, b_rms_weight=None, c_rms_weight= None, dt_rms_weight= None, b_c_dt_rms_eps=1e-6):
        """
             xz: (batch, dim, seqlen)
        """
        assert causal_conv1d_cuda is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        assert checkpoint_lvl in [0, 1]
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_weight = out_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_bias = (out_proj_bias.to(dtype=torch.get_autocast_gpu_dtype())
                             if out_proj_bias is not None else None)
        if xz.stride(-1) != 1:
            xz = xz.contiguous()
        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        x, z = xz.chunk(2, dim=1)
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
        conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
            x, conv1d_weight, conv1d_bias, None, None, None, True
        )
        # We're being very careful here about the layout, to avoid extra transposes.
        # We want delta to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
        delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l = L)
        ctx.is_variable_B = B is None
        ctx.is_variable_C = C is None
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None
        if B is None:  # variable B
            B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                # B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if B.stride(-1) != 1:
                B = B.contiguous()
        if C is None:  # variable C
            C = x_dbl[:, -d_state:]  # (bl dstate)
            if C_proj_bias is not None:
                C = C + C_proj_bias.to(dtype=C.dtype)
            if not A.is_complex():
                # C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if C.stride(-1) != 1:
                C = C.contiguous()
        if D is not None:
            D = D.contiguous()

        if b_rms_weight is not None:
            B = rearrange(B, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
            B = rms_norm_forward(B, b_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        if c_rms_weight is not None:
            C = rearrange(C, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
            C = rms_norm_forward(C, c_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        if dt_rms_weight is not None:
            delta = rearrange(delta, "b d l -> (b l) d", l=L).contiguous()
            delta = rms_norm_forward(delta, dt_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            delta = rearrange(delta, "(b l) d -> b d l", l=L).contiguous()

        out, scan_intermediates, out_z = selective_scan_cuda.fwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
        )
        ctx.delta_softplus = delta_softplus
        ctx.out_proj_bias_is_None = out_proj_bias is None
        ctx.checkpoint_lvl = checkpoint_lvl
        ctx.b_rms_weight = b_rms_weight
        ctx.c_rms_weight = c_rms_weight
        ctx.dt_rms_weight = dt_rms_weight
        ctx.b_c_dt_rms_eps = b_c_dt_rms_eps
        if checkpoint_lvl >= 1:  # Will recompute conv1d_out and delta in the backward pass
            conv1d_out, delta = None, None
        ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
                              delta_proj_weight, out_proj_weight, conv1d_out, delta,
                              A, B, C, D, delta_bias, scan_intermediates, b_rms_weight, c_rms_weight, dt_rms_weight, out)
        return F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        # dout: (batch, seqlen, dim)
        assert causal_conv1d_cuda is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, out_proj_weight,
         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, b_rms_weight, c_rms_weight, dt_rms_weight, out) = ctx.saved_tensors
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        x, z = xz.chunk(2, dim=1)
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if ctx.checkpoint_lvl == 1:
            conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
                x, conv1d_weight, conv1d_bias, None, None, None, True
            )
            delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
                              "d (b l) -> b d l", l = L)
            if dt_rms_weight is not None:
                delta = rearrange(delta, "b d l -> (b l) d", l=L).contiguous()
                delta = rms_norm_forward(delta, ctx.dt_rms_weight, None, ctx.b_c_dt_rms_eps)
                delta = rearrange(delta, "(b l) d -> b d l", l=L).contiguous()
            if b_rms_weight is not None:
                # Recompute & RMSNorm B
                B = rearrange(B, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
                B = rms_norm_forward(
                    B, ctx.b_rms_weight, None, ctx.b_c_dt_rms_eps
                )
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            if c_rms_weight is not None:
                # Recompute & RMSNorm C
                C = rearrange(C, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
                C = rms_norm_forward(
                    C, ctx.c_rms_weight, None, ctx.b_c_dt_rms_eps
                )
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()

        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
        dx, dz = dxz.chunk(2, dim=1)
        dout = rearrange(dout, "b l e -> e (b l)")
        dout_y = rearrange(out_proj_weight.t() @ dout, "d (b l) -> b d l", l=L)
        dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, dout_y, scan_intermediates, out, dz,
            ctx.delta_softplus,
            True  # option to recompute out_z
        )
        dout_proj_weight = torch.einsum("eB,dB->ed", dout, rearrange(out_z, "b d l -> d (b l)"))
        dout_proj_bias = dout.sum(dim=(0, 1)) if not ctx.out_proj_bias_is_None else None
        dD = dD if D is not None else None
        dx_dbl = torch.empty_like(x_dbl)
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
            dB = None
        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dx_dbl[:, -d_state:] = dC  # (bl d)
            dC = None
        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
        dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
        dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
        dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        dx, dconv1d_weight, dconv1d_bias, *_ = causal_conv1d_cuda.causal_conv1d_bwd(
            x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, None, dx, False, True
        )
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
                dout_proj_weight, dout_proj_bias,
                dA, dB, dC, dD,
                ddelta_bias if delta_bias is not None else None,
                # 6-None are delta_softplus, checkpoint_lvl, b_rms_weight, c_rms_weight, dt_rms_weight, b_c_dt_rms_eps
                dB_proj_bias, dC_proj_bias, None, None, None, None, None, None)


def mamba_inner_fn(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1,
    b_rms_weight=None, c_rms_weight=None, dt_rms_weight=None,
    b_c_dt_rms_eps=1e-6
):
    if SCAN_OPTION == "ref":
        return mamba_inner_ref(
            xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
            out_proj_weight, out_proj_bias,
            A, B, C, D, delta_bias, B_proj_bias,
            C_proj_bias, delta_softplus
        )
    elif SCAN_OPTION == "torch" or SCAN_OPTION == "torchscan":
        # For the torch implementation, we also use mamba_inner_ref which calls selective_scan_fn internally
        return mamba_inner_ref(
            xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
            out_proj_weight, out_proj_bias,
            A, B, C, D, delta_bias, B_proj_bias,
            C_proj_bias, delta_softplus
        )
    else:
        return MambaInnerFn.apply(
            xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
            out_proj_weight, out_proj_bias,
            A, B, C, D, delta_bias, B_proj_bias,
            C_proj_bias, delta_softplus, checkpoint_lvl, b_rms_weight, c_rms_weight, dt_rms_weight, b_c_dt_rms_eps
        )

GLOBAL_SEEN_INNER_REF = False

def mamba_inner_ref(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True
):
    global GLOBAL_SEEN_INNER_REF
    if not GLOBAL_SEEN_INNER_REF:
        msg = dedent("""
            mamba_inner_ref(
                xz.shape: {xz.shape},
                conv1d_weight.shape: {conv1d_weight.shape},
                conv1d_bias.shape: {conv1d_bias.shape},
                x_proj_weight.shape: {x_proj_weight.shape},
                delta_proj_weight.shape: {delta_proj_weight.shape},
                out_proj_weight.shape: {out_proj_weight.shape},
                out_proj_bias.shape: {out_proj_bias.shape},
                A.shape: {A.shape},
                B.shape: {B.shape},
                C.shape: {C.shape},
                D.shape: {D.shape},
                delta_bias.shape: {delta_bias.shape},
                B_proj_bias.shape: {B_proj_bias.shape},
                C_proj_bias.shape: {C_proj_bias.shape},
                delta_softplus: {delta_softplus}
            )
        """.format(xz=xz,
                   conv1d_weight=conv1d_weight,
                   conv1d_bias=conv1d_bias,
                   x_proj_weight=x_proj_weight,
                   delta_proj_weight=delta_proj_weight,
                   out_proj_weight=out_proj_weight,
                   out_proj_bias=out_proj_bias,
                   A=A, B=B, C=C, D=D,
                   delta_bias=delta_bias,
                   B_proj_bias=B_proj_bias,
                   C_proj_bias=C_proj_bias,
                   delta_softplus=delta_softplus))
        print(msg)
        GLOBAL_SEEN_INNER_REF = True
    assert causal_conv1d_fn is not None, "causal_conv1d_fn is not available. Please install causal-conv1d."
    L = xz.shape[-1]
    delta_rank = delta_proj_weight.shape[1]
    d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
    x, z = xz.chunk(2, dim=1)
    x = causal_conv1d_fn(x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias, activation="silu")
    # We're being very careful here about the layout, to avoid extra transposes.
    # We want delta to have d as the slowest moving dimension
    # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
    x_dbl = F.linear(rearrange(x, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
    delta = delta_proj_weight @ x_dbl[:, :delta_rank].t()
    delta = rearrange(delta, "d (b l) -> b d l", l=L)
    if B is None:  # variable B
        B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl d)
        if B_proj_bias is not None:
            B = B + B_proj_bias.to(dtype=B.dtype)
        if not A.is_complex():
            B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            B = rearrange(B, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    if C is None:  # variable B
        C = x_dbl[:, -d_state:]  # (bl d)
        if C_proj_bias is not None:
            C = C + C_proj_bias.to(dtype=C.dtype)
        if not A.is_complex():
            C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            C = rearrange(C, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    y = selective_scan_fn(x, delta, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=True)
    return F.linear(rearrange(y, "b d l -> b l d"), out_proj_weight, out_proj_bias)
