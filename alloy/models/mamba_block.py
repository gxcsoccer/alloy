"""Mamba-2 block implemented with pure MLX ops.

Reference: Dao & Gu, "Transformers are SSMs" (Mamba-2), arXiv 2405.21060.

The selective scan is implemented using standard MLX operations for automatic
differentiation support. A Metal kernel version can be swapped in later for
inference speed.
"""

import mlx.core as mx
import mlx.nn as nn


class MambaBlock(nn.Module):
    """Mamba-2 selective state-space block.

    Args:
        d_model: Model dimension.
        d_state: SSM state dimension (N).
        d_conv: Causal convolution width.
        expand: Expansion factor for inner dimension.
        headdim: Dimension per SSM head.
        chunk_size: Chunk size for parallel scan.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.chunk_size = chunk_size

        d_inner = d_model * expand
        self.d_inner = d_inner
        assert d_inner % headdim == 0, "d_inner must be divisible by headdim"
        self.n_heads = d_inner // headdim

        # Input projection: x -> (x_branch, z_gate)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)

        # Causal depthwise conv1d (manual weights — MLX Conv1d lacks groups)
        self.conv_weight = mx.random.normal((d_inner, d_conv)) * 0.02
        self.conv_bias = mx.zeros((d_inner,))

        # SSM parameter projections (input-dependent B, C, dt)
        self.x_proj = nn.Linear(d_inner, self.n_heads * (d_state + d_state + 1), bias=False)

        # Learnable log(A) parameter — one per head (initialized to log(1) = 0)
        self.A_log = mx.zeros([self.n_heads])

        # dt bias
        self.dt_bias = mx.zeros([self.n_heads])

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
    ) -> tuple:
        """Process one chunk of the selective scan using parallel matmul.

        Args:
            x_c: [B, cs, n_heads, headdim]
            a_c: [B, cs, n_heads] — decay factors
            B_c: [B, cs, n_heads, d_state]
            C_c: [B, cs, n_heads, d_state]
            h:   [B, n_heads, d_state, headdim] — incoming state
            cs:  chunk size (may differ from self.chunk_size for last chunk)

        Returns:
            (y_chunk, h_new) where y_chunk: [B, cs, n_heads, headdim],
            h_new: [B, n_heads, d_state, headdim]
        """
        B_size = x_c.shape[0]
        n_heads = self.n_heads
        headdim = self.headdim
        d_state = self.d_state

        # b_bar = B outer x: [B, cs, n_heads, d_state, headdim]
        b_bar = B_c[..., None] * x_c[:, :, :, None, :]

        # Build transfer matrix in log-space for numerical stability
        # log(a): a_c is already exp(A*dt), A<0 so 0 < a_c <= 1, log is safe
        log_a = mx.log(mx.clip(a_c, a_min=1e-10, a_max=None))  # [B, cs, n_heads]
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
                # Save last d_conv-1 positions from padded sequence (includes zero-padding)
                cache.conv_state = x_padded[:, -(self.d_conv - 1) :, :]

        x_conv = self._depthwise_conv1d(x_padded, L)
        x_conv = nn.silu(x_conv)

        # 3. SSM parameter projections (input-dependent B, C, dt)
        ssm_params = self.x_proj(x_conv)  # [B, L, n_heads * (2*d_state + 1)]
        ssm_params = ssm_params.reshape(B_size, L, self.n_heads, 2 * self.d_state + 1)

        B_param = ssm_params[..., : self.d_state]  # [B, L, n_heads, d_state]
        C_param = ssm_params[..., self.d_state : 2 * self.d_state]
        dt = ssm_params[..., -1]  # [B, L, n_heads]

        # Discretize: A_disc = exp(A * dt), where A = -exp(A_log)
        dt = nn.softplus(dt + self.dt_bias)  # [B, L, n_heads]
        A = -mx.exp(self.A_log)  # [n_heads]
        A_disc = mx.exp(dt * A)  # [B, L, n_heads]

        # 4. Selective scan
        x_heads = x_conv.reshape(B_size, L, self.n_heads, self.headdim)
        y = self._selective_scan(x_heads, A_disc, B_param, C_param, cache)

        # 5. Output gate (SiLU) and projection
        y = y.reshape(B_size, L, self.d_inner)
        y = y * nn.silu(z)
        return self.out_proj(y)
