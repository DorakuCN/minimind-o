#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def read_table_like_training(path: Path) -> pa.Table:
    table = pa.Table.from_batches(pq.ParquetFile(path).iter_batches())
    schema = pa.schema([
        field.with_type(pa.large_string()) if pa.types.is_string(field.type) else field
        for field in table.schema
    ])
    return table.cast(schema)


def shard_file(path: Path, output_dir: Path, num_shards: int, row_group_size: int) -> None:
    table = read_table_like_training(path)
    rows = table.num_rows
    stem = path.stem
    print(f"{path}: rows={rows}, columns={table.column_names}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    for rank in range(num_shards):
        start = rows * rank // num_shards
        end = rows * (rank + 1) // num_shards
        shard = table.slice(start, end - start)
        out_path = output_dir / f"{stem}.rank{rank:02d}-of{num_shards:02d}.parquet"
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        pq.write_table(shard, tmp_path, compression="zstd", row_group_size=row_group_size)
        os.replace(tmp_path, out_path)
        print(f"  rank {rank}: rows={shard.num_rows}, path={out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Shard parquet files for per-rank DDP loading.")
    parser.add_argument("--num_shards", type=int, default=3)
    parser.add_argument("--output_dir", type=Path, default=Path("dataset/_full_shards"))
    parser.add_argument("--row_group_size", type=int, default=8192)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    for path in args.paths:
        shard_file(path, args.output_dir, args.num_shards, args.row_group_size)


if __name__ == "__main__":
    main()
