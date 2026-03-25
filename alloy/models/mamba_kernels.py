"""Metal kernel implementations for Mamba-2 selective scan.

Provides fused GPU kernels for:
1. Depthwise conv1d + SiLU activation (4.1x speedup)
2. Flat outer product: B * x without materializing 5D tensor
3. C-contraction + state extraction from h_all

These replace the pure MLX ops in mamba_block.py for training speedup.
"""

import mlx.core as mx

# ===========================================================================
# Kernel 1: Fused Depthwise Conv1d + SiLU
# ===========================================================================

_conv1d_silu_kernel = mx.fast.metal_kernel(
    name="conv1d_silu",
    input_names=["x_padded", "weight", "bias"],
    output_names=["out"],
    source="""
        uint d = thread_position_in_grid.x;   // channel index
        uint t = thread_position_in_grid.y;   // time index
        uint b = thread_position_in_grid.z;   // batch index

        uint L_padded = L_OUT + D_CONV - 1;
        uint base_in = b * L_padded * D_INNER + t * D_INNER + d;
        float acc = bias[d];
        for (uint k = 0; k < D_CONV; k++) {
            acc += x_padded[base_in + k * D_INNER] * weight[d * D_CONV + k];
        }
        // Fused SiLU: x * sigmoid(x)
        float s = acc / (1.0f + metal::exp(-acc));
        out[b * L_OUT * D_INNER + t * D_INNER + d] = s;
    """,
)


def fused_conv1d_silu(x_padded, weight, bias, L):
    """Fused depthwise conv1d + SiLU via Metal. [B, L+d_conv-1, d_inner] -> [B, L, d_inner]."""
    B = x_padded.shape[0]
    d_inner, d_conv = weight.shape
    return _conv1d_silu_kernel(
        inputs=[x_padded, weight, bias],
        template=[("T", x_padded.dtype), ("L_OUT", L), ("D_INNER", d_inner), ("D_CONV", d_conv)],
        grid=(d_inner, L, B),
        threadgroup=(min(d_inner, 256), 1, 1),
        output_shapes=[(B, L, d_inner)],
        output_dtypes=[x_padded.dtype],
    )[0]


# ===========================================================================
# Kernel 2: Flat Outer Product (B * x -> b_flat without 5D intermediate)
# ===========================================================================

_flat_outer_kernel = mx.fast.metal_kernel(
    name="flat_outer",
    input_names=["B_c", "x_c"],
    output_names=["b_flat"],
    source="""
        // Output layout: b_flat[b, head, t, n * HEADDIM + d]
        // = B_c[b, t, head, n] * x_c[b, t, head, d]

        uint elem = thread_position_in_grid.x;
        uint total_per_batch = N_HEADS * CS * D_STATE * HEADDIM;
        uint b = elem / total_per_batch;
        uint rem = elem % total_per_batch;
        uint head = rem / (CS * D_STATE * HEADDIM);
        rem = rem % (CS * D_STATE * HEADDIM);
        uint t = rem / (D_STATE * HEADDIM);
        rem = rem % (D_STATE * HEADDIM);
        uint n = rem / HEADDIM;
        uint d = rem % HEADDIM;

        // B_c layout: [B, cs, n_heads, d_state] row-major
        uint b_idx = b * CS * N_HEADS * D_STATE + t * N_HEADS * D_STATE + head * D_STATE + n;
        // x_c layout: [B, cs, n_heads, headdim] row-major
        uint x_idx = b * CS * N_HEADS * HEADDIM + t * N_HEADS * HEADDIM + head * HEADDIM + d;

        b_flat[elem] = B_c[b_idx] * x_c[x_idx];
    """,
)


def flat_outer_product(B_c, x_c, n_heads, headdim, d_state):
    """Compute b_flat[b,head,t,n*headdim+d] = B[b,t,head,n] * x[b,t,head,d] without 5D intermediate.

    Returns: [B, n_heads, cs, d_state * headdim]
    """
    B_size, cs = B_c.shape[0], B_c.shape[1]
    total = B_size * n_heads * cs * d_state * headdim
    return _flat_outer_kernel(
        inputs=[B_c, x_c],
        template=[
            ("T", B_c.dtype), ("CS", cs), ("N_HEADS", n_heads),
            ("HEADDIM", headdim), ("D_STATE", d_state),
        ],
        grid=(total, 1, 1),
        threadgroup=(min(total, 256), 1, 1),
        output_shapes=[(B_size, n_heads, cs, d_state * headdim)],
        output_dtypes=[B_c.dtype],
    )[0]


# ===========================================================================
# Kernel 3: C-contraction + h_new extraction
# ===========================================================================

_c_contract_kernel = mx.fast.metal_kernel(
    name="c_contract",
    input_names=["C_c", "h_all"],
    output_names=["y_out", "h_out"],
    source="""
        // h_all: [B, n_heads, cs, d_state, headdim]
        // C_c:   [B, cs, n_heads, d_state]
        // y_out: [B, cs, n_heads, headdim]
        // h_out: [B, n_heads, d_state, headdim] (last time step)

        uint tid = thread_position_in_threadgroup.x;
        uint gid = threadgroup_position_in_grid.x;

        // Decode (b, head, t) from gid
        uint b = gid / (N_HEADS * CS);
        uint head = (gid / CS) % N_HEADS;
        uint t = gid % CS;
        uint d = tid;  // headdim index

        if (d >= HEADDIM) return;

        // Compute y = sum_n C[n] * h_all[n, d]
        float y_val = 0.0f;
        for (uint n = 0; n < D_STATE; n++) {
            float c_val = C_c[b * CS * N_HEADS * D_STATE + t * N_HEADS * D_STATE + head * D_STATE + n];
            float h_val = h_all[b * N_HEADS * CS * D_STATE * HEADDIM
                                + head * CS * D_STATE * HEADDIM
                                + t * D_STATE * HEADDIM
                                + n * HEADDIM + d];
            y_val += c_val * h_val;
        }
        y_out[b * CS * N_HEADS * HEADDIM + t * N_HEADS * HEADDIM + head * HEADDIM + d] = y_val;

        // Extract h_new from last time step
        if (t == CS - 1) {
            for (uint n = 0; n < D_STATE; n++) {
                float h_val = h_all[b * N_HEADS * CS * D_STATE * HEADDIM
                                    + head * CS * D_STATE * HEADDIM
                                    + t * D_STATE * HEADDIM
                                    + n * HEADDIM + d];
                h_out[b * N_HEADS * D_STATE * HEADDIM + head * D_STATE * HEADDIM + n * HEADDIM + d] = h_val;
            }
        }
    """,
)


def c_contraction(C_c, h_all, n_heads, headdim, d_state):
    """Compute y = sum_n C[n] * h_all[n, d] and extract h_new.

    Args:
        C_c: [B, cs, n_heads, d_state]
        h_all: [B, n_heads, cs, d_state, headdim]

    Returns:
        y_chunk: [B, cs, n_heads, headdim]
        h_new:   [B, n_heads, d_state, headdim]
    """
    B_size, cs = C_c.shape[0], C_c.shape[1]
    n_groups = B_size * n_heads * cs
    threads_per = min(headdim, 256)

    return _c_contract_kernel(
        inputs=[C_c, h_all],
        template=[
            ("T", mx.float32), ("CS", cs), ("N_HEADS", n_heads),
            ("HEADDIM", headdim), ("D_STATE", d_state),
        ],
        grid=(n_groups * threads_per, 1, 1),
        threadgroup=(threads_per, 1, 1),
        output_shapes=[
            (B_size, cs, n_heads, headdim),
            (B_size, n_heads, d_state, headdim),
        ],
        output_dtypes=[mx.float32, mx.float32],
    )


# ===========================================================================
# Optimized scan chunk: Metal kernels + MLX matmul
# ===========================================================================


def scan_chunk_metal(x_c, a_c, B_c, C_c, h, n_heads, headdim, d_state):
    """Optimized scan chunk using Metal kernels where beneficial.

    Uses:
    - Metal: flat_outer_product (avoids 5D b_bar tensor)
    - MLX: matmul M @ b_flat (hardware matrix engines)
    - Metal: c_contraction (fused sum + h_new extraction)
    """
    B_size, cs = x_c.shape[0], x_c.shape[1]

    # 1. Flat outer product (Metal) — avoids 5D b_bar
    b_flat = flat_outer_product(B_c, x_c, n_heads, headdim, d_state)

    # 2. Transfer matrix M (MLX — fast enough)
    log_a = mx.log(mx.clip(a_c, a_min=1e-10, a_max=None))
    log_a_cum = mx.cumsum(log_a, axis=1)
    lac = log_a_cum.transpose(0, 2, 1)
    M_log = lac[:, :, :, None] - lac[:, :, None, :]
    M = mx.exp(mx.where(mx.tril(mx.ones((cs, cs))) > 0, M_log, mx.array(float("-inf"))))

    # 3. Matmul (MLX — uses hardware matrix engines)
    h_input = (M @ b_flat).reshape(B_size, n_heads, cs, d_state, headdim)

    # 4. State propagation (MLX — simple broadcast)
    decay = mx.exp(lac)
    h_from_init = decay[:, :, :, None, None] * h[:, :, None, :, :]
    h_all = h_from_init + h_input

    # 5. C-contraction + h_new extraction (Metal)
    y_chunk, h_new = c_contraction(C_c, h_all, n_heads, headdim, d_state)

    return y_chunk, h_new


def scan_chunk_pure_mlx(x_c, a_c, B_c, C_c, h, n_heads, headdim, d_state):
    """Pure MLX reference implementation."""
    B_size, cs = x_c.shape[0], x_c.shape[1]

    b_bar = B_c[..., None] * x_c[:, :, :, None, :]
    log_a = mx.log(mx.clip(a_c, a_min=1e-10, a_max=None))
    log_a_cum = mx.cumsum(log_a, axis=1)
    lac = log_a_cum.transpose(0, 2, 1)
    M_log = lac[:, :, :, None] - lac[:, :, None, :]
    M = mx.exp(mx.where(mx.tril(mx.ones((cs, cs))) > 0, M_log, mx.array(float("-inf"))))
    b_flat = b_bar.transpose(0, 2, 1, 3, 4).reshape(B_size, n_heads, cs, d_state * headdim)
    h_input = (M @ b_flat).reshape(B_size, n_heads, cs, d_state, headdim)
    decay = mx.exp(lac)
    h_from_init = decay[:, :, :, None, None] * h[:, :, None, :, :]
    h_all = h_from_init + h_input

    C_t = C_c.transpose(0, 2, 1, 3)
    y_chunk = (C_t[:, :, :, :, None] * h_all).sum(axis=3)
    y_chunk = y_chunk.transpose(0, 2, 1, 3)
    h_new = h_all[:, :, -1, :, :]
    return y_chunk, h_new


# ===========================================================================
# Kernel 4: Fused Parallel Associative Scan
# ===========================================================================
# Replaces the O(cs²) matmul-based scan with O(cs·log(cs)) parallel prefix.
# Single kernel launch, shared memory, no intermediate M matrix.
#
# SSM recurrence: h[t] = a[t] * h[t-1] + b[t]
# Monoid: (a1,b1) ⊕ (a2,b2) = (a1·a2, a2·b1 + b2)
#
# Thread layout: one threadgroup per (batch, head).
# Each thread handles one (n, d) pair across ALL time steps.
# Shared memory stores a[t] for all threads to read.
# Each thread maintains its own b[t, n, d] in registers.

# Parallel scan kernel: one threadgroup per (batch*head*n*d),
# CS threads per group (one per time position).
# Each thread stores just (a_t, b_t) — minimal register pressure.
# Shared memory: 2 * CS floats for ping-pong scan buffers.

_parallel_scan_kernel = mx.fast.metal_kernel(
    name="parallel_scan",
    input_names=["a_in", "b_in", "h_init"],
    output_names=["h_all_out"],
    source="""
        uint t = thread_position_in_threadgroup.x;   // time position
        uint gid = threadgroup_position_in_grid.x;

        // Decode (bh, n, d) from group id
        // gid layout: (b*H + head) * D_STATE * HEADDIM + n * HEADDIM + d
        uint nd = gid % (D_STATE * HEADDIM);
        uint bh = gid / (D_STATE * HEADDIM);
        uint n = nd / HEADDIM;
        uint d_idx = nd % HEADDIM;

        if (t >= CS) return;

        // Load my (a, b) values
        float my_a = a_in[bh * CS + t];
        float my_b = b_in[bh * CS * D_STATE * HEADDIM + t * D_STATE * HEADDIM + n * HEADDIM + d_idx];

        // First position: include initial state
        if (t == 0) {
            float h0 = h_init[bh * D_STATE * HEADDIM + n * HEADDIM + d_idx];
            my_b = my_a * h0 + my_b;
        }

        // Ping-pong shared memory for the scan
        threadgroup float s_a[2 * CS];
        threadgroup float s_b[2 * CS];

        uint src = 0;  // read from s[src*CS..], write to s[(1-src)*CS..]
        s_a[t] = my_a;
        s_b[t] = my_b;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Hillis-Steele scan: log2(CS) steps
        for (uint stride = 1; stride < CS; stride *= 2) {
            uint r = src * CS;
            uint w = (1 - src) * CS;

            if (t >= stride) {
                float a_prev = s_a[r + t - stride];
                float b_prev = s_b[r + t - stride];
                float a_cur = s_a[r + t];
                float b_cur = s_b[r + t];
                s_a[w + t] = a_prev * a_cur;
                s_b[w + t] = a_cur * b_prev + b_cur;
            } else {
                s_a[w + t] = s_a[r + t];
                s_b[w + t] = s_b[r + t];
            }
            src = 1 - src;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Write scan result h[t, n, d] to global memory
        // h_all_out: [B*H, CS, D_STATE, HEADDIM]
        h_all_out[bh * CS * D_STATE * HEADDIM + t * D_STATE * HEADDIM + n * HEADDIM + d_idx]
            = s_b[src * CS + t];
    """,
)


def fused_parallel_scan_chunk(x_c, a_c, B_c, C_c, h, n_heads, headdim, d_state):
    """Fused parallel scan for one chunk — O(cs·log(cs)) via Metal.

    Two-kernel approach:
    1. Parallel scan kernel → h_all[B*H, cs, N, D] (one threadgroup per b*h*n*d)
    2. C-contraction → y[B, cs, H, D] (reuses existing kernel)

    Args:
        x_c: [B, cs, n_heads, headdim] — dt-scaled input
        a_c: [B, cs, n_heads] — discretized decay
        B_c: [B, cs, n_heads, d_state]
        C_c: [B, cs, n_heads, d_state]
        h:   [B, n_heads, d_state, headdim] — incoming state

    Returns:
        y_chunk: [B, cs, n_heads, headdim]
        h_new:   [B, n_heads, d_state, headdim]
    """
    B_size, cs = x_c.shape[0], x_c.shape[1]

    # Promote to float32
    x_c = x_c.astype(mx.float32) if x_c.dtype != mx.float32 else x_c
    B_c = B_c.astype(mx.float32) if B_c.dtype != mx.float32 else B_c
    C_c = C_c.astype(mx.float32) if C_c.dtype != mx.float32 else C_c
    a_c = a_c.astype(mx.float32) if a_c.dtype != mx.float32 else a_c

    # b_input = B * x: [B, cs, H, N, D] → [B*H, cs, N*D]
    b_input = B_c[..., None] * x_c[:, :, :, None, :]
    b_flat = b_input.transpose(0, 2, 1, 3, 4).reshape(B_size * n_heads, cs, d_state * headdim)

    # a: [B*H, cs]
    a_flat = a_c.transpose(0, 2, 1).reshape(B_size * n_heads, cs)

    # h_init: [B*H, N*D]
    h_flat = h.reshape(B_size * n_heads, d_state * headdim)

    # Kernel 1: Parallel scan → h_all [B*H, cs, N, D]
    n_groups = B_size * n_heads * d_state * headdim  # one group per (b,h,n,d)
    threads_per_group = min(cs, 256)  # one thread per time step

    h_all_flat = _parallel_scan_kernel(
        inputs=[a_flat, b_flat, h_flat],
        template=[
            ("T", mx.float32),
            ("CS", cs),
            ("N_HEADS", n_heads),
            ("HEADDIM", headdim),
            ("D_STATE", d_state),
        ],
        grid=(n_groups * threads_per_group, 1, 1),
        threadgroup=(threads_per_group, 1, 1),
        output_shapes=[
            (B_size * n_heads, cs, d_state * headdim),
        ],
        output_dtypes=[mx.float32],
    )[0]

    # Reshape h_all to [B, n_heads, cs, d_state, headdim]
    h_all = h_all_flat.reshape(B_size, n_heads, cs, d_state, headdim)

    # Kernel 2: C-contraction → y + h_new (reuse existing)
    y_chunk, h_new = c_contraction(C_c, h_all, n_heads, headdim, d_state)

    return y_chunk, h_new
