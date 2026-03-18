"""Streaming data loading with sequence packing for language model training.

Supports JSONL format: {"text": "..."} per line.
"""

import json
from pathlib import Path
from typing import Iterator

import mlx.core as mx


class Dataset:
    """Streaming dataset that loads JSONL, tokenizes, and packs sequences.

    Args:
        data_path: Path to JSONL file.
        tokenizer: Tokenizer with an `encode(text) -> list[int]` method.
        seq_len: Target sequence length for packing.
        batch_size: Number of sequences per batch.
        eos_token: EOS token ID inserted between documents.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        seq_len: int = 2048,
        batch_size: int = 4,
        eos_token: int = 0,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.eos_token = eos_token

    def _token_stream(self) -> Iterator[int]:
        """Yield token IDs one at a time from the JSONL file.

        Each document is followed by an EOS token. The file is read
        line-by-line to keep memory usage constant.
        """
        with open(self.data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                tokens = self.tokenizer.encode(doc["text"])
                yield from tokens
                yield self.eos_token

    def __iter__(self) -> Iterator[mx.array]:
        """Yield batches of packed token IDs of shape [batch_size, seq_len].

        Sequences are concatenated with an EOS separator and split into
        fixed-length chunks. No padding is used — partial final chunks
        are dropped.
        """
        chunk_size = self.seq_len
        tokens_needed = self.batch_size * chunk_size
        buffer = []

        for token_id in self._token_stream():
            buffer.append(token_id)
            if len(buffer) >= tokens_needed:
                batch = mx.array(buffer[:tokens_needed]).reshape(
                    self.batch_size, chunk_size
                )
                buffer = buffer[tokens_needed:]
                yield batch

        # Drop incomplete final batch
