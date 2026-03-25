"""Mamba-2 block implemented with MLX + Metal kernels.

Reference: Dao & Gu, "Transformers are SSMs" (Mamba-2), arXiv 2405.21060.

Uses Metal kernels for conv1d+SiLU (8x speedup) and optimized scan chunk.
Set USE_METAL_KERNELS = False to fall back to pure MLX ops for debugging.
"""

import mlx.core as mx
import mlx.nn as nn

USE_METAL_KERNELS = True

try:
    from alloy.models.mamba_kernels import fused_conv1d_silu, scan_chunk_metal

    @mx.custom_function
    def _metal_conv1d_silu(x_padded, weight, bias, L_arr):
        """Metal conv1d+SiLU with autodiff support."""
        L = int(L_arr.item())
        return fused_conv1d_silu(x_padded, weight, bias, L)

    @_metal_conv1d_silu.vjp
    def _metal_conv1d_silu_vjp(primals, cotangent, output):
        x_padded, weight, bias, L_arr = primals
        dy = cotangent
        L = int(L_arr.item())
        d_conv = weight.shape[1]
        # Recompute pre-activation for SiLU backward
        out_pre = mx.zeros_like(x_padded[:, :L, :])
        for k in range(d_conv):
            out_pre = out_pre + x_padded[:, k : k + L, :] * weight[:, k]
        out_pre = out_pre + bias
        sig = mx.sigmoid(out_pre)
        dsilu = sig * (1 + out_pre * (1 - sig))
        dconv = dy * dsilu
        dx_padded = mx.zeros_like(x_padded)
        dweight = mx.zeros_like(weight)
        for k in range(d_conv):
            dx_padded = dx_padded.at[:, k : k + L, :].add(dconv * weight[:, k])
            dweight = dweight.at[:, k].add(
                (dconv * x_padded[:, k : k + L, :]).sum(axis=(0, 1))
            )
        dbias = dconv.sum(axis=(0, 1))
        return dx_padded, dweight, dbias, mx.array(0)

except ImportError:
    USE_METAL_KERNELS = False


class MambaBlock(nn.Module):
    """Mamba-2 selective state-space block.

    Supports two modes:
    - **Alloy mode** (combined_proj=False, default): Separate in_proj (x, z)
      and x_proj (B, C, dt) as in the original Alloy implementation.
    - **Zamba2 mode** (combined_proj=True): Single in_proj that produces
      x, z, B, C, dt in one projection (matching HF Zamba2 weights).
      Also supports D skip parameter and inner RMS norm.

    Args:
        d_model: Model dimension.
        d_state: SSM state dimension (N).
        d_conv: Causal convolution width.
        expand: Expansion factor for inner dimension.
        headdim: Dimension per SSM head.
        chunk_size: Chunk size for parallel scan.
        combined_proj: If True, use Zamba2-style combined projection.
        n_groups: Number of groups for B, C projections (Zamba2 default: 1).
        use_D: If True, include D skip connection parameter.
        use_inner_norm: If True, apply RMS norm before output projection.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: int = 256,
        combined_proj: bool = False,
        n_groups: int = 1,
        use_D: bool = False,
        use_inner_norm: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.chunk_size = chunk_size
        self.combined_proj = combined_proj
        self.n_groups = n_groups

        d_inner = d_model * expand
        self.d_inner = d_inner
        assert d_inner % headdim == 0, "d_inner must be divisible by headdim"
        self.n_heads = d_inner // headdim

        if combined_proj:
            # Zamba2-style: single projection for x, z, B, C, dt
            # in_proj output: [d_inner, d_inner, d_state*n_groups, d_state*n_groups, n_heads]
            self.d_ssm = 2 * d_state * n_groups + self.n_heads  # B + C + dt
            proj_dim = 2 * d_inner + self.d_ssm
            self.in_proj = nn.Linear(d_model, proj_dim, bias=False)
            self.x_proj = None  # not used

            # Conv operates on x + B + C (extended)
            conv_dim = d_inner + 2 * d_state * n_groups
            self.conv_weight = mx.random.normal((conv_dim, d_conv)) * 0.02
            self.conv_bias = mx.zeros((conv_dim,))
        else:
            # Alloy-style: separate projections
            self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
            self.conv_weight = mx.random.normal((d_inner, d_conv)) * 0.02
            self.conv_bias = mx.zeros((d_inner,))
            self.x_proj = nn.Linear(d_inner, self.n_heads * (d_state + d_state + 1), bias=False)

        # Learnable log(A) parameter — one per head (initialized to log(1) = 0)
        self.A_log = mx.zeros([self.n_heads])

        # dt bias
        self.dt_bias = mx.zeros([self.n_heads])

        # Optional D skip connection (Zamba2)
        if use_D:
            self.D = mx.zeros([self.n_heads])
        else:
            self.D = None

        # Optional inner RMS norm (Zamba2)
        if use_inner_norm:
            self.norm = nn.RMSNorm(d_inner)
        else:
            self.norm = None

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def _depthwise_conv1d(self, x_padded: mx.array, L: int) -> mx.array:
        """Causal depthwise 1D convolution.

        Args:
            x_padded: Input with left-padding, shape [B, L + d_conv - 1, d_inner].
            L: Output sequence length.

        Returns:
            Convolution output of shape [B, L, d_inner].
        """
        # Sliding-window dot product — d_conv is small (typically 4)
        out = mx.zeros_like(x_padded[:, :L, :])
        for k in range(self.d_conv):
            out = out + x_padded[:, k : k + L, :] * self.conv_weight[:, k]
        return out + self.conv_bias

    def _selective_scan_sequential(
        self,
        x: mx.array,
        A_disc: mx.array,
        B: mx.array,
        C: mx.array,
        cache=None,
    ) -> mx.array:
        """Sequential selective scan — used for single-step decoding with cache.

        Args:
            x: Input per head, shape [B, L, n_heads, headdim].
            A_disc: Discretized decay, shape [B, L, n_heads].
            B: Input-dependent B, shape [B, L, n_heads, d_state].
            C: Input-dependent C, shape [B, L, n_heads, d_state].
            cache: Optional MambaCache to read/write SSM state.

        Returns:
            Output per head, shape [B, L, n_heads, headdim].
        """
        # Promote to float32 for scan precision
        x = x.astype(mx.float32) if x.dtype != mx.float32 else x
        B = B.astype(mx.float32) if B.dtype != mx.float32 else B
        C = C.astype(mx.float32) if C.dtype != mx.float32 else C
        A_disc = A_disc.astype(mx.float32) if A_disc.dtype != mx.float32 else A_disc

        B_size, L, n_heads, headdim = x.shape

        if cache is not None and cache.ssm_state is not None:
            h = cache.ssm_state
        else:
            h = mx.zeros((B_size, n_heads, self.d_state, headdim))

        outputs = []
        for t in range(L):
            a = A_disc[:, t, :, None, None]
            bx = B[:, t, :, :, None] * x[:, t, :, None, :]
            h = a * h + bx
            y_t = (C[:, t, :, :, None] * h).sum(axis=2)
            outputs.append(y_t)

        if cache is not None:
            cache.ssm_state = h

        return mx.stack(outputs, axis=1)

    def _scan_chunk(
        self,
        x_c: mx.array,
        a_c: mx.array,
        B_c: mx.array,
        C_c: mx.array,
        h: mx.array,
        cs: int,
        log_a_direct: mx.array = None,
    ) -> tuple:
        """Process one chunk of the selective scan using parallel matmul.

        Args:
            x_c: [B, cs, n_heads, headdim]
            a_c: [B, cs, n_heads] — decay factors (exp(A*dt))
            B_c: [B, cs, n_heads, d_state]
            C_c: [B, cs, n_heads, d_state]
            h:   [B, n_heads, d_state, headdim] — incoming state
            cs:  chunk size (may differ from self.chunk_size for last chunk)
            log_a_direct: [B, cs, n_heads] — if provided, use directly as log(A_disc)
                to avoid exp→log roundtrip precision loss.

        Returns:
            (y_chunk, h_new) where y_chunk: [B, cs, n_heads, headdim],
            h_new: [B, n_heads, d_state, headdim]
        """
        B_size = x_c.shape[0]
        n_heads = self.n_heads
        headdim = self.headdim
        d_state = self.d_state

        # Promote to float32 for scan precision (critical for bf16 training)
        x_c = x_c.astype(mx.float32) if x_c.dtype != mx.float32 else x_c
        B_c = B_c.astype(mx.float32) if B_c.dtype != mx.float32 else B_c
        C_c = C_c.astype(mx.float32) if C_c.dtype != mx.float32 else C_c

        # b_bar = B outer x: [B, cs, n_heads, d_state, headdim]
        b_bar = B_c[..., None] * x_c[:, :, :, None, :]

        # Build transfer matrix in log-space for numerical stability
        if log_a_direct is not None:
            log_a = log_a_direct.astype(mx.float32) if log_a_direct.dtype != mx.float32 else log_a_direct
        else:
            log_a = mx.log(mx.clip(a_c.astype(mx.float32), a_min=1e-10, a_max=None))
        log_a_cum = mx.cumsum(log_a, axis=1)  # [B, cs, n_heads]

        # M[t, s] = exp(log_a_cum[t] - log_a_cum[s]) for t >= s
        # Reshape for broadcasting: [B, n_heads, cs, 1] - [B, n_heads, 1, cs]
        lac = log_a_cum.transpose(0, 2, 1)  # [B, n_heads, cs]
        # Apply causal mask BEFORE exp to avoid inf from upper-triangle
        M_log = lac[:, :, :, None] - lac[:, :, None, :]
        causal_mask = mx.where(
            mx.tril(mx.ones((cs, cs))) > 0, M_log, mx.array(float("-inf"))
        )
        M = mx.exp(causal_mask)  # [B, n_heads, cs, cs]

        # h_from_input = M @ b_bar (over the time dimension)
        # b_bar: [B, cs, n_heads, d_state, headdim] -> [B, n_heads, cs, d_state*headdim]
        b_flat = b_bar.transpose(0, 2, 1, 3, 4).reshape(B_size, n_heads, cs, d_state * headdim)
        h_input = (M @ b_flat).reshape(B_size, n_heads, cs, d_state, headdim)

        # h_from_init: decay of initial state across all positions in chunk
        # decay[t] = prod(a[0..t]) = exp(log_a_cum[t])
        decay = mx.exp(lac)  # [B, n_heads, cs]
        h_from_init = decay[:, :, :, None, None] * h[:, :, None, :, :]

        # Total hidden state at each position
        h_all = h_from_init + h_input  # [B, n_heads, cs, d_state, headdim]

        # Output: y[t] = sum_d_state C[t] * h[t]
        C_t = C_c.transpose(0, 2, 1, 3)  # [B, n_heads, cs, d_state]
        y_chunk = (C_t[:, :, :, :, None] * h_all).sum(axis=3)  # [B, n_heads, cs, headdim]

        # Transpose to [B, cs, n_heads, headdim]
        y_chunk = y_chunk.transpose(0, 2, 1, 3)

        # New state = last position's hidden state
        h_new = h_all[:, :, -1, :, :]  # [B, n_heads, d_state, headdim]

        return y_chunk, h_new

    def _selective_scan(
        self,
        x: mx.array,
        A_disc: mx.array,
        B: mx.array,
        C: mx.array,
        cache=None,
        log_a: mx.array = None,
    ) -> mx.array:
        """Chunked parallel selective scan.

        Uses matmul-based parallel computation within each chunk (O(C²) but
        fully parallel on Metal), with sequential state propagation between
        chunks. Falls back to sequential scan for L <= chunk_size or when
        decoding with cache (L is typically 1).

        Args:
            x: Input per head, shape [B, L, n_heads, headdim].
            A_disc: Discretized decay, shape [B, L, n_heads].
            B: Input-dependent B, shape [B, L, n_heads, d_state].
            C: Input-dependent C, shape [B, L, n_heads, d_state].
            cache: Optional MambaCache to read/write SSM state.
            log_a: If provided, [B, L, n_heads] log-space decay (A*dt) to
                avoid exp→log roundtrip.

        Returns:
            Output per head, shape [B, L, n_heads, headdim].
        """
        B_size, L, n_heads, headdim = x.shape

        # For single-step decoding, use sequential (no overhead)
        if L <= 1:
            return self._selective_scan_sequential(x, A_disc, B, C, cache)

        # Initialize state
        if cache is not None and cache.ssm_state is not None:
            h = cache.ssm_state
        else:
            h = mx.zeros((B_size, n_heads, self.d_state, headdim))

        cs = self.chunk_size
        outputs = []

        for start in range(0, L, cs):
            end = min(start + cs, L)
            chunk_len = end - start

            y_chunk, h = self._scan_chunk(
                x[:, start:end],
                A_disc[:, start:end],
                B[:, start:end],
                C[:, start:end],
                h,
                chunk_len,
                log_a_direct=log_a[:, start:end] if log_a is not None else None,
            )
            outputs.append(y_chunk)

        if cache is not None:
            cache.ssm_state = h

        return mx.concatenate(outputs, axis=1)  # [B, L, n_heads, headdim]

    def __call__(self, x: mx.array, cache=None) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, L, d_model].
            cache: Optional MambaCache for autoregressive generation.

        Returns:
            Output tensor of shape [B, L, d_model].
        """
        B_size, L, _ = x.shape

        if self.combined_proj:
            return self._forward_combined(x, B_size, L, cache)
        else:
            return self._forward_split(x, B_size, L, cache)

    def _forward_split(self, x, B_size, L, cache):
        """Alloy-style forward: separate in_proj and x_proj."""
        # 1. Input projection → x_branch (for SSM) and z (output gate)
        xz = self.in_proj(x)  # [B, L, 2 * d_inner]
        x_branch, z = xz[..., : self.d_inner], xz[..., self.d_inner :]

        # 2. Causal depthwise conv1d
        if cache is not None and cache.conv_state is not None:
            x_padded = mx.concatenate([cache.conv_state, x_branch], axis=1)
            cache.conv_state = x_padded[:, -(self.d_conv - 1) :, :]
        else:
            x_padded = mx.pad(x_branch, [(0, 0), (self.d_conv - 1, 0), (0, 0)])
            if cache is not None:
                cache.conv_state = x_padded[:, -(self.d_conv - 1) :, :]

        if USE_METAL_KERNELS and cache is None:
            x_conv = _metal_conv1d_silu(x_padded, self.conv_weight, self.conv_bias, mx.array(L))
        else:
            x_conv = self._depthwise_conv1d(x_padded, L)
            x_conv = nn.silu(x_conv)

        # 3. SSM parameter projections (input-dependent B, C, dt)
        ssm_params = self.x_proj(x_conv)  # [B, L, n_heads * (2*d_state + 1)]
        ssm_params = ssm_params.reshape(B_size, L, self.n_heads, 2 * self.d_state + 1)

        B_param = ssm_params[..., : self.d_state]
        C_param = ssm_params[..., self.d_state : 2 * self.d_state]
        dt = ssm_params[..., -1]

        # Discretize
        dt = nn.softplus(dt + self.dt_bias)
        A = -mx.exp(self.A_log)
        A_disc = mx.exp(dt * A)

        # 4. Selective scan
        x_heads = x_conv.reshape(B_size, L, self.n_heads, self.headdim)
        y = self._selective_scan(x_heads, A_disc, B_param, C_param, cache)

        # 5. Output gate (SiLU) and projection
        y = y.reshape(B_size, L, self.d_inner)
        y = y * nn.silu(z)
        return self.out_proj(y)

    def _forward_combined(self, x, B_size, L, cache):
        """Zamba2-style forward: combined in_proj with B, C, dt included.

        HF Zamba2 in_proj split order: [gate(z), conv_input(x+B+C), dt]
        Where conv_dim = d_inner + 2*d_state*n_groups
        """
        ng = self.n_groups
        conv_dim = self.d_inner + 2 * self.d_state * ng  # x + B + C for conv

        # 1. Combined projection
        proj = self.in_proj(x)  # [B, L, d_inner + conv_dim + n_heads]

        # Split matching HF order: [gate, conv_input, dt]
        gate = proj[..., : self.d_inner]                              # z / output gate
        conv_input = proj[..., self.d_inner : self.d_inner + conv_dim]  # x + B + C
        dt = proj[..., self.d_inner + conv_dim :]                     # [B, L, n_heads]

        # 2. Causal depthwise conv1d on conv_input
        if cache is not None and cache.conv_state is not None:
            x_padded = mx.concatenate([cache.conv_state, conv_input], axis=1)
            cache.conv_state = x_padded[:, -(self.d_conv - 1) :, :]
        else:
            x_padded = mx.pad(conv_input, [(0, 0), (self.d_conv - 1, 0), (0, 0)])
            if cache is not None:
                cache.conv_state = x_padded[:, -(self.d_conv - 1) :, :]

        conv_out = self._depthwise_conv1d(x_padded, L)
        conv_out = nn.silu(conv_out)

        # Split conv output: x (d_inner), B (d_state*ng), C (d_state*ng)
        x_conv = conv_out[..., : self.d_inner]
        B_conv = conv_out[..., self.d_inner : self.d_inner + self.d_state * ng]
        C_conv = conv_out[..., self.d_inner + self.d_state * ng :]

        # 3. Reshape B, C for SSM (expand groups to heads)
        B_param = B_conv.reshape(B_size, L, ng, self.d_state)
        C_param = C_conv.reshape(B_size, L, ng, self.d_state)
        if ng < self.n_heads:
            reps = self.n_heads // ng
            B_param = mx.repeat(B_param, reps, axis=2)
            C_param = mx.repeat(C_param, reps, axis=2)

        # 4. Discretize: HF does hidden_states * dt first, then exp(A * dt)
        dt = nn.softplus(dt + self.dt_bias)
        dt = mx.clip(dt, a_min=1e-4, a_max=None)  # time_step_min clamp (HF default)
        A = -mx.exp(self.A_log)

        # HF Mamba-2: D residual uses UNSCALED x, scan uses dt-scaled x
        x_heads = x_conv.reshape(B_size, L, self.n_heads, self.headdim)

        # D skip connection on UNSCALED x (before dt scaling, matching HF)
        if self.D is not None:
            D_residual = self.D[None, None, :, None] * x_heads

        x_heads = x_heads * dt[..., None]  # Scale by dt for scan
        log_a = A * dt  # log-space decay (pass directly to avoid exp→log roundtrip)
        A_disc = mx.exp(log_a)  # [B, L, n_heads]

        # 5. Selective scan (pass log_a for precision)
        y = self._selective_scan(x_heads, A_disc, B_param, C_param, cache, log_a=log_a)

        # 6. Add D residual
        if self.D is not None:
            y = y + D_residual

        # 7. Gate FIRST, then norm (Zamba2RMSNormGated order)
        y = y.reshape(B_size, L, self.d_inner)
        y = y * nn.silu(gate)
        if self.norm is not None:
            y = self.norm(y)

        return self.out_proj(y)
