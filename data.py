"""Data pipeline: reuse the tokenized dataset produced in Assignment 1.

Assignment 1 saved GPT-2 BPE token ids as blocks of 512 tokens
(data/tokenized_blocks.pt, shape (656119, 512)); the last block of each
document is padded with the EOS token. This module:

1. memory-maps that file and reads only the blocks that are needed,
2. samples blocks evenly across the whole file so every domain
   (Wikipedia, news, web) is represented,
3. removes EOS padding (keeping one EOS as a document separator),
4. splits train/validation by segment, so the two sets share no text,
5. cuts the token stream into (input, target) windows of seq_len tokens.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

GPT2_VOCAB_SIZE = 50257
GPT2_EOS_ID = 50256
DEFAULT_BLOCK_LEN = 512


def _torch_load(path: str | Path):
    """Load a .pt file, memory-mapping it when the installed torch supports it."""
    attempts = (
        {"mmap": True, "weights_only": True},  # reads rows lazily from disk
        {"weights_only": True},
        {"weights_only": False},  # our own trusted file from Assignment 1
    )
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return torch.load(path, map_location="cpu", **kwargs)
        except (TypeError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
            last_error = exc
    raise RuntimeError(f"Could not load {path}: {last_error}")


def _extract_blocks(obj) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return (token_ids, attention_mask or None) from the Assignment 1 file."""
    if isinstance(obj, dict):
        ids = None
        for key in ("input_ids", "blocks", "tokens", "token_ids", "data"):
            if isinstance(obj.get(key), torch.Tensor):
                ids = obj[key]
                break
        if ids is None:  # fall back to the first 2-D integer tensor in the dict
            for value in obj.values():
                if (
                    isinstance(value, torch.Tensor)
                    and value.dim() == 2
                    and not value.is_floating_point()
                ):
                    ids = value
                    break
        if ids is None:
            raise ValueError(f"No token tensor found in dataset. Keys: {list(obj)}")
        mask = obj.get("attention_mask")
        mask = mask if isinstance(mask, torch.Tensor) else None
    elif isinstance(obj, torch.Tensor):
        ids, mask = obj, None
    elif hasattr(obj, "__array__"):  # numpy array
        ids, mask = torch.as_tensor(obj), None
    else:
        raise ValueError(f"Unsupported dataset format: {type(obj).__name__}")

    if ids.dim() == 1:  # a flat token stream: view it as blocks
        n_blocks = ids.numel() // DEFAULT_BLOCK_LEN
        ids = ids[: n_blocks * DEFAULT_BLOCK_LEN].view(n_blocks, DEFAULT_BLOCK_LEN)
        mask = None
    if ids.dim() != 2:
        raise ValueError(f"Expected a 2-D tensor of token blocks, got shape {tuple(ids.shape)}")
    return ids, mask


def _segment_ranges(n_total: int, num_blocks: int, n_segments: int) -> list[tuple[int, int]]:
    """Pick num_blocks rows as n_segments evenly spaced runs of consecutive rows.

    Consecutive rows keep chunks of the same document together, while spreading
    the runs over the whole file covers every domain in the dataset.
    """
    num_blocks = min(num_blocks, n_total)
    n_segments = max(1, min(n_segments, num_blocks))
    rows_per_segment = num_blocks // n_segments
    spacing = n_total // n_segments
    return [(s * spacing, s * spacing + rows_per_segment) for s in range(n_segments)]


def _blocks_to_stream(
    ids: torch.Tensor, mask: torch.Tensor | None, eos_id: int
) -> torch.Tensor:
    """Concatenate blocks into one token stream, dropping EOS padding.

    A padded block is the last block of a document, so one EOS token is kept
    after it as a document separator.
    """
    block_len = ids.size(1)
    if mask is not None:
        real_lens = mask.to(torch.long).sum(dim=1)
    else:
        positions = torch.arange(block_len).repeat(ids.size(0), 1)
        last_real = positions.masked_fill(ids == eos_id, -1).max(dim=1).values
        real_lens = last_real + 1

    eos = torch.tensor([eos_id], dtype=torch.long)
    pieces = []
    for row, real_len in zip(ids, real_lens.tolist()):
        if real_len > 0:
            pieces.append(row[:real_len])
        if real_len < block_len:
            pieces.append(eos)
    return torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.long)


def prepare_token_streams(
    data_path: str,
    num_blocks: int = 4000,
    n_segments: int = 40,
    val_every: int = 10,
    eos_id: int = GPT2_EOS_ID,
    cache_dir: str = "data/cache",
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Build (train_stream, val_stream, info) from the Assignment 1 dataset.

    Every val_every-th segment goes to validation (10% by default), so the
    validation text never appears in training. Results are cached on disk.
    """
    data_path = str(data_path)
    cache_path = (
        Path(cache_dir)
        / f"streams_{Path(data_path).stem}_{num_blocks}_{n_segments}_{val_every}.pt"
    )
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        return cached["train"], cached["val"], cached["info"]

    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Pass --data_path pointing to tokenized_blocks.pt from Assignment 1."
        )

    ids, mask = _extract_blocks(_torch_load(data_path))
    n_total, block_len = ids.shape
    ranges = _segment_ranges(n_total, num_blocks, n_segments)
    eos = torch.tensor([eos_id], dtype=torch.long)

    train_parts, val_parts = [], []
    for s, (start, end) in enumerate(ranges):
        seg_ids = ids[start:end].to(torch.long)
        seg_mask = mask[start:end] if mask is not None else None
        stream = torch.cat([_blocks_to_stream(seg_ids, seg_mask, eos_id), eos])
        is_val = len(ranges) > 1 and (
            s % val_every == val_every - 1 or (len(ranges) < val_every and s == len(ranges) - 1)
        )
        (val_parts if is_val else train_parts).append(stream)

    train_stream = torch.cat(train_parts)
    if val_parts:
        val_stream = torch.cat(val_parts)
    else:  # a single segment: hold out the last 10% of its tokens
        cut = int(train_stream.numel() * 0.9)
        train_stream, val_stream = train_stream[:cut].clone(), train_stream[cut:].clone()

    info = {
        "data_path": data_path,
        "total_blocks_in_file": int(n_total),
        "block_len": int(block_len),
        "blocks_used": int(sum(end - start for start, end in ranges)),
        "n_segments": len(ranges),
        "train_tokens": int(train_stream.numel()),
        "val_tokens": int(val_stream.numel()),
        "max_token_id": int(max(train_stream.max(), val_stream.max())),
        "used_attention_mask": mask is not None,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"train": train_stream, "val": val_stream, "info": info}, cache_path)
    return train_stream, val_stream, info


class NextTokenDataset(Dataset):
    """Non-overlapping windows for next-token prediction.

    Sample i: x = tokens[i*L : i*L + L], y = tokens[i*L + 1 : i*L + L + 1],
    i.e. the target is the input shifted one position to the left.
    """

    def __init__(self, stream: torch.Tensor, seq_len: int):
        if stream.numel() <= seq_len:
            raise ValueError(
                f"Token stream ({stream.numel()} tokens) is shorter than seq_len={seq_len}."
            )
        self.stream = stream
        self.seq_len = seq_len
        self.n_samples = (stream.numel() - 1) // seq_len

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = i * self.seq_len
        chunk = self.stream[start : start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def make_dataloaders(
    train_stream: torch.Tensor,
    val_stream: torch.Tensor,
    seq_len: int,
    batch_size: int,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Shuffled training loader (seeded) and ordered validation loader."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        NextTokenDataset(train_stream, seq_len),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )
    val_loader = DataLoader(
        NextTokenDataset(val_stream, seq_len), batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader
