"""Tests for Metal kernel implementations."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from alloy.models.mamba_kernels import (
    fused_conv1d_silu,
    scan_chunk_metal,
    scan_chunk_pure_mlx,
    flat_outer_product,
)


class TestConv1dSiLU:
    """Test fused conv1d + SiLU Metal kernel."""

    def test_matches_reference(self):
        mx.random.seed(42)
        B, L, d_inner, d_conv = 2, 128, 256, 4
        x_padded = mx.random.normal((B, L + d_conv - 1, d_inner))
        weight = mx.random.normal((d_inner, d_conv)) * 0.02
        bias = mx.zeros((d_inner,))

        # Metal
        y_metal = fused_conv1d_silu(x_padded, weight, bias, L)

        # Reference
        out = mx.zeros_like(x_padded[:, :L, :])
        for k in range(d_conv):
            out = out + x_padded[:, k : k + L, :] * weight[:, k]
        y_ref = nn.silu(out + bias)

        mx.eval(y_metal, y_ref)
        assert mx.allclose(y_metal, y_ref, atol=1e-5).item()

    def test_various_sizes(self):
        """Test with different sequence lengths and channel sizes."""
        for B, L, d_inner in [(1, 64, 128), (4, 2048, 1024), (2, 512, 512)]:
            d_conv = 4
            x_padded = mx.random.normal((B, L + d_conv - 1, d_inner))
            weight = mx.random.normal((d_inner, d_conv)) * 0.02
            bias = mx.zeros((d_inner,))

            y = fused_conv1d_silu(x_padded, weight, bias, L)
            mx.eval(y)
            assert y.shape == (B, L, d_inner)


class TestFlatOuterProduct:
    """Test flat outer product Metal kernel."""

    def test_matches_reference(self):
        mx.random.seed(42)
        B, cs, n_heads, headdim, d_state = 2, 64, 8, 64, 16
        B_c = mx.random.normal((B, cs, n_heads, d_state)) * 0.1
        x_c = mx.random.normal((B, cs, n_heads, headdim))

        # Metal
        b_flat = flat_outer_product(B_c, x_c, n_heads, headdim, d_state)

        # Reference
        b_bar = B_c[..., None] * x_c[:, :, :, None, :]
        b_flat_ref = b_bar.transpose(0, 2, 1, 3, 4).reshape(
            B, n_heads, cs, d_state * headdim
        )

        mx.eval(b_flat, b_flat_ref)
        assert mx.allclose(b_flat, b_flat_ref, atol=1e-5).item()


class TestScanChunk:
    """Test scan chunk Metal hybrid."""

    def test_matches_reference(self):
        mx.random.seed(42)
        B, cs, n_heads, headdim, d_state = 2, 64, 8, 64, 16
        x_c = mx.random.normal((B, cs, n_heads, headdim))
        a_c = mx.random.uniform(shape=(B, cs, n_heads)) * 0.5 + 0.25
        B_c = mx.random.normal((B, cs, n_heads, d_state)) * 0.1
        C_c = mx.random.normal((B, cs, n_heads, d_state)) * 0.1
        h = mx.random.normal((B, n_heads, d_state, headdim)) * 0.1

        y_m, h_m = scan_chunk_metal(x_c, a_c, B_c, C_c, h, n_heads, headdim, d_state)
        y_r, h_r = scan_chunk_pure_mlx(x_c, a_c, B_c, C_c, h, n_heads, headdim, d_state)
        mx.eval(y_m, h_m, y_r, h_r)

        assert mx.allclose(y_m, y_r, atol=1e-5).item()
        assert mx.allclose(h_m, h_r, atol=1e-5).item()

    def test_zero_initial_state(self):
        mx.random.seed(42)
        B, cs, n_heads, headdim, d_state = 4, 128, 16, 64, 16
        x_c = mx.random.normal((B, cs, n_heads, headdim))
        a_c = mx.random.uniform(shape=(B, cs, n_heads)) * 0.5 + 0.25
        B_c = mx.random.normal((B, cs, n_heads, d_state)) * 0.1
        C_c = mx.random.normal((B, cs, n_heads, d_state)) * 0.1
        h = mx.zeros((B, n_heads, d_state, headdim))

        y_m, h_m = scan_chunk_metal(x_c, a_c, B_c, C_c, h, n_heads, headdim, d_state)
        y_r, h_r = scan_chunk_pure_mlx(x_c, a_c, B_c, C_c, h, n_heads, headdim, d_state)
        mx.eval(y_m, h_m, y_r, h_r)

        assert mx.allclose(y_m, y_r, atol=1e-5).item()


class TestMetalInMambaBlock:
    """Test MambaBlock works with Metal kernels enabled."""

    def test_forward_matches(self):
        """Metal and pure MLX forward should produce same results."""
        import alloy.models.mamba_block as mbb

        mx.random.seed(42)
        mb = mbb.MambaBlock(d_model=256, d_state=16, d_conv=4, expand=2, headdim=64, chunk_size=64)
        mx.eval(mb.parameters())
        x = mx.random.normal((1, 256, 256))

        mbb.USE_METAL_KERNELS = True
        y_metal = mb(x)
        mx.eval(y_metal)

        mbb.USE_METAL_KERNELS = False
        y_mlx = mb(x)
        mx.eval(y_mlx)

        mbb.USE_METAL_KERNELS = True  # restore
        assert mx.allclose(y_metal, y_mlx, atol=1e-4).item()

    def test_backward_works(self):
        """Gradient computation should work with Metal kernels."""
        import alloy.models.mamba_block as mbb
        mbb.USE_METAL_KERNELS = True

        mx.random.seed(42)
        mb = mbb.MambaBlock(d_model=256, d_state=16, d_conv=4, expand=2, headdim=64, chunk_size=64)
        mx.eval(mb.parameters())
        x = mx.random.normal((1, 128, 256))

        loss_grad = nn.value_and_grad(mb, lambda m, x: m(x).sum())
        loss, grads = loss_grad(mb, x)
        mx.eval(loss, grads)
        assert not mx.isnan(loss).item()
