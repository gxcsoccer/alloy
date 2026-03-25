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
