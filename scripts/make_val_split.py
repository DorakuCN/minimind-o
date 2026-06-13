#!/usr/bin/env python3
"""Create deterministic validation parquet splits for training monitoring."""

import argparse
import os
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def sample_parquet(src: Path, dst: Path, n_rows: int, seed: int, batch_size: int):
    parquet = pq.ParquetFile(src)
    total = parquet.metadata.num_rows
    if total == 0:
        raise ValueError(f"empty parquet: {src}")
    take = min(n_rows, total)
    indices = sorted(random.Random(seed).sample(range(total), take))
    chunks = []
    cursor = 0
    idx_pos = 0
    for batch in parquet.iter_batches(batch_size=batch_size):
        batch_len = batch.num_rows
        batch_end = cursor + batch_len
        local = []
        while idx_pos < len(indices) and indices[idx_pos] < batch_end:
            local.append(indices[idx_pos] - cursor)
            idx_pos += 1
        if local:
            chunks.append(pa.Table.from_batches([batch]).take(pa.array(local, type=pa.int64())))
        cursor = batch_end
        if idx_pos >= len(indices):
            break
    if not chunks:
        raise RuntimeError(f"no rows sampled from {src}")
    sampled = pa.concat_tables(chunks, promote_options="default")
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(sampled, dst)
    return take, total


def main():
    parser = argparse.ArgumentParser(description="Create deterministic val parquet splits")
    parser.add_argument("--dataset_dir", default="dataset")
    parser.add_argument("--output_dir", default="dataset/_val")
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8192, help="Parquet read batch size")
    parser.add_argument(
        "--sources",
        default="sft_t2a.parquet,sft_a2a.parquet,sft_i2t.parquet",
        help="Comma-separated parquet filenames under dataset_dir",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    summary = []
    for idx, name in enumerate(args.sources.split(",")):
        name = name.strip()
        if not name:
            continue
        src = dataset_dir / name
        stem = Path(name).stem
        dst = output_dir / f"{stem}_val.parquet"
        taken, total = sample_parquet(src, dst, args.rows, args.seed + idx, args.batch_size)
        summary.append((name, taken, total, dst))
        print(f"{name}: wrote {taken}/{total} rows -> {dst}")

    manifest = output_dir / "manifest.txt"
    with open(manifest, "w", encoding="utf-8") as f:
        f.write(f"seed={args.seed}\nrows_per_source={args.rows}\n")
        for name, taken, total, dst in summary:
            f.write(f"{name}\t{taken}/{total}\t{dst}\n")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
