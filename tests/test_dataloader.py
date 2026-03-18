"""Tests for the data loading module."""

import json
import tempfile

import mlx.core as mx
import pytest

from alloy.data.dataloader import Dataset


class SimpleTokenizer:
    """Minimal tokenizer for testing."""

    def encode(self, text):
        return [ord(c) % 256 for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


@pytest.fixture
def data_file():
    """Create a temporary JSONL data file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for _ in range(100):
            f.write(json.dumps({"text": "hello world " * 50}) + "\n")
        return f.name


class TestDataset:
    """Tests for the Dataset class."""

    def test_yields_batches(self, data_file):
        """Dataset yields mx.array batches."""
        ds = Dataset(data_file, SimpleTokenizer(), seq_len=32, batch_size=2)
        batch = next(iter(ds))
        assert isinstance(batch, mx.array)

    def test_batch_shape(self, data_file):
        """Batches have shape [batch_size, seq_len]."""
        ds = Dataset(data_file, SimpleTokenizer(), seq_len=64, batch_size=4)
        batch = next(iter(ds))
        assert batch.shape == (4, 64)

    def test_multiple_batches(self, data_file):
        """Dataset yields multiple batches."""
        ds = Dataset(data_file, SimpleTokenizer(), seq_len=32, batch_size=2)
        batches = list(ds)
        assert len(batches) > 1

    def test_values_in_range(self, data_file):
        """Token IDs are within expected range."""
        ds = Dataset(data_file, SimpleTokenizer(), seq_len=32, batch_size=2)
        batch = next(iter(ds))
        mx.eval(batch)
        assert batch.min().item() >= 0
        assert batch.max().item() < 256

    def test_empty_lines_skipped(self):
        """Empty lines in JSONL are skipped."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("\n")
            f.write(json.dumps({"text": "x" * 500}) + "\n")
            f.write("\n")
            f.write(json.dumps({"text": "y" * 500}) + "\n")
            fname = f.name

        ds = Dataset(fname, SimpleTokenizer(), seq_len=16, batch_size=1)
        batches = list(ds)
        assert len(batches) > 0

    def test_incomplete_batch_dropped(self):
        """Final incomplete batch is not yielded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write just enough for ~1.5 batches
            f.write(json.dumps({"text": "a" * 100}) + "\n")
            fname = f.name

        ds = Dataset(fname, SimpleTokenizer(), seq_len=32, batch_size=2)
        batches = list(ds)
        for b in batches:
            assert b.shape == (2, 32)
