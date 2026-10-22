from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin
import numpy as np

from emma.augment import HDF5Source


METHOD_CHOICES = {
    "none",
    "gzip",
    "lzf",
    "bitshuffle",
    "blosc",
    "bzip2",
    "lz4",
    "zstd",
}


@dataclass(frozen=True)
class CompressionConfig:
    name: str
    compression: Any | None
    compression_opts: Any | None = None
    shuffle: bool = False


@dataclass(frozen=True)
class SourceMetadata:
    keys: list[str]
    first_shape: tuple[int, ...]
    first_dtype: np.dtype
    logical_payload_bytes: int
    dataset_storage_bytes: int
    source_file_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark lossless HDF5 compression methods on real augmentation "
            "data using EMMa-Wrapper's HDF5Source."
        )
    )

    source = parser.add_argument_group("source data")
    source.add_argument(
        "--source-file",
        type=Path,
        required=True,
        help=(
            "Existing HDF5Source file containing representative real data. "
            "It may already be compressed: selected items are decoded and "
            "rewritten uncompressed before benchmarking."
        ),
    )
    source.add_argument(
        "--items",
        type=int,
        default=1000,
        help="Maximum number of source items to benchmark (default: 1000).",
    )
    source.add_argument(
        "--source-selection",
        choices=("random", "first"),
        default="random",
        help=(
            "How to choose --items entries from --source-file. Random selection "
            "is deterministic from --seed (default: random)."
        ),
    )
    source.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed for source selection and random-read request generation.",
    )
    source.add_argument(
        "--source-path",
        type=Path,
        default=None,
        help=(
            "Path for the uncompressed benchmark source. Defaults to "
            "<work-dir>/source_raw.h5."
        ),
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--work-dir",
        type=Path,
        default=Path("./compression_benchmarks"),
        help="Directory for source_raw.h5, benchmark files, and results.csv.",
    )
    output.add_argument(
        "--results-csv",
        type=Path,
        default=None,
        help="CSV output path. Defaults to <work-dir>/results.csv.",
    )
    output.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing raw source and benchmark files.",
    )
    output.add_argument(
        "--delete-benchmark-files",
        action="store_true",
        help="Delete each compressed HDF5 benchmark file after measuring it.",
    )

    benchmark = parser.add_argument_group("benchmark grid")
    benchmark.add_argument(
        "--preset",
        choices=("quick", "full"),
        default="quick",
        help="Compression parameter grid size (default: quick).",
    )
    benchmark.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional subset of methods: " + " ".join(sorted(METHOD_CHOICES)),
    )

    reads = parser.add_argument_group("read benchmark")
    reads.add_argument(
        "--single-count",
        type=int,
        default=2000,
        help="Number of random single-item reads per configuration.",
    )
    reads.add_argument(
        "--batch-count",
        type=int,
        default=200,
        help="Number of random batch reads per configuration.",
    )
    reads.add_argument(
        "--read-batch-size",
        type=int,
        default=32,
        help="Number of items in each random read batch.",
    )
    reads.add_argument(
        "--validate-count",
        type=int,
        default=32,
        help="Number of sampled items checked for exact lossless equality.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.items <= 0:
        raise ValueError("--items must be positive")
    if args.single_count < 0 or args.batch_count < 0:
        raise ValueError("Read benchmark counts cannot be negative")
    if args.read_batch_size <= 0:
        raise ValueError("--read-batch-size must be positive")
    if args.validate_count < 0:
        raise ValueError("--validate-count cannot be negative")

    if args.methods is not None:
        unknown = sorted(set(args.methods) - METHOD_CHOICES)
        if unknown:
            raise ValueError(f"Unknown methods: {unknown}")


def build_configs(args: argparse.Namespace) -> list[CompressionConfig]:
    methods = set(args.methods or METHOD_CHOICES)
    configs: list[CompressionConfig] = []

    def add(
        name: str,
        compression: Any | None,
        compression_opts: Any | None = None,
        *,
        shuffle: bool = False,
    ) -> None:
        configs.append(
            CompressionConfig(
                name=name,
                compression=compression,
                compression_opts=compression_opts,
                shuffle=shuffle,
            )
        )

    if "none" in methods:
        add("none", None)

    if "gzip" in methods:
        levels = [5, 7, 9] if args.preset == "quick" else list(range(0, 10))
        for level in levels:
            add(f"gzip_l{level}", "gzip", level)
            add(f"gzip_l{level}_shuffle", "gzip", level, shuffle=True)

    if "lzf" in methods:
        add("lzf", "lzf")
        add("lzf_shuffle", "lzf", shuffle=True)

    if "zstd" in methods:
        levels = [9, 15, 22] if args.preset == "quick" else [1, 3, 5, 9, 15, 22]
        for level in levels:
            add(f"zstd_{level}", hdf5plugin.Zstd(clevel=level))

    if "lz4" in methods:
        block_sizes = (
            [0, 1 << 15]
            if args.preset == "quick"
            else [0, 1 << 12, 1 << 15, 1 << 18]
        )
        for block_size in block_sizes:
            add(f"lz4_n{block_size}", hdf5plugin.LZ4(nbytes=block_size))

    if "bzip2" in methods:
        block_sizes = [5, 7, 9] if args.preset == "quick" else [1, 3, 5, 7, 9]
        for block_size in block_sizes:
            add(f"bzip2_{block_size}", hdf5plugin.BZip2(blocksize=block_size))

    if "bitshuffle" in methods:
        candidates: list[tuple[str, Any]] = [
            ("bitshuffle_none", hdf5plugin.Bitshuffle(cname="none")),
            ("bitshuffle_lz4", hdf5plugin.Bitshuffle(cname="lz4")),
            ("bitshuffle_zstd_9", hdf5plugin.Bitshuffle(cname="zstd", clevel=9)),
            ("bitshuffle_zstd_15", hdf5plugin.Bitshuffle(cname="zstd", clevel=15)),
            ("bitshuffle_zstd_22", hdf5plugin.Bitshuffle(cname="zstd", clevel=22)),
        ]
        if args.preset == "full":
            candidates.append(
                (
                    "bitshuffle_nelems4096_lz4",
                    hdf5plugin.Bitshuffle(nelems=4096, cname="lz4"),
                )
            )
        for name, compression in candidates:
            add(name, compression)

    if "blosc" in methods:
        cnames = ["blosclz", "lz4", "lz4hc", "zlib", "zstd"]
        levels = [5, 7, 9] if args.preset == "quick" else [1, 3, 5, 7, 9]
        shuffle_modes = [
            ("shuffle", hdf5plugin.Blosc.SHUFFLE),
            ("bitshuffle", hdf5plugin.Blosc.BITSHUFFLE),
        ]
        if args.preset == "full":
            shuffle_modes.insert(0, ("noshuffle", hdf5plugin.Blosc.NOSHUFFLE))

        for cname in cnames:
            for level in levels:
                for shuffle_name, shuffle_mode in shuffle_modes:
                    add(
                        f"blosc_{cname}_l{level}_{shuffle_name}",
                        hdf5plugin.Blosc(
                            cname=cname,
                            clevel=level,
                            shuffle=shuffle_mode,
                        ),
                    )

    configs.sort(key=lambda config: config.name)
    return configs


def select_source_keys(
    keys: list[str],
    *,
    count: int,
    selection: str,
    seed: int,
) -> list[str]:
    if count >= len(keys):
        return keys
    if selection == "first":
        return keys[:count]

    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(keys)), k=count))
    return [keys[index] for index in selected_indices]


def materialize_raw_source(
    input_path: Path,
    raw_path: Path,
    *,
    item_count: int,
    selection: str,
    seed: int,
    overwrite: bool,
) -> Path:
    """Decode selected source items and rewrite them without compression."""
    if input_path.resolve() == raw_path.resolve():
        raise ValueError("--source-file and --source-path must be different files")

    if raw_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Uncompressed source already exists: {raw_path}. "
                "Pass --overwrite to replace it."
            )
        raw_path.unlink()

    raw_path.parent.mkdir(parents=True, exist_ok=True)

    with HDF5Source("benchmark_input", input_path, mode="r") as source:
        keys = list(source.keys())
        if not keys:
            raise RuntimeError(f"No datasets found in source file: {input_path}")

        selected_keys = select_source_keys(
            keys,
            count=min(item_count, len(keys)),
            selection=selection,
            seed=seed,
        )

        with HDF5Source(
            "benchmark_raw",
            raw_path,
            mode="w",
            compression=None,
            compression_opts=None,
            shuffle=False,
            fletcher32=False,
        ) as writer:
            for key in selected_keys:
                writer.save(key, source[key])

    print(
        f"Materialized {len(selected_keys)} decoded items from {input_path} "
        f"to uncompressed source {raw_path}"
    )
    return raw_path


def read_source_metadata(source_path: Path) -> SourceMetadata:
    with HDF5Source("benchmark", source_path, mode="r") as source:
        keys = list(source.keys())
        if not keys:
            raise RuntimeError(f"No datasets found in source file: {source_path}")
        first = np.asarray(source[keys[0]])

    logical_payload_bytes = 0
    stored_dataset_bytes = 0
    with h5py.File(source_path, "r") as h5:
        for key in keys:
            obj = h5[key]
            if not isinstance(obj, h5py.Dataset):
                raise ValueError(
                    "HDF5Source benchmark expects one top-level dataset per item; "
                    f"{key!r} is not a dataset"
                )
            logical_payload_bytes += int(obj.size * obj.dtype.itemsize)
            stored_dataset_bytes += int(obj.id.get_storage_size())

    return SourceMetadata(
        keys=keys,
        first_shape=tuple(first.shape),
        first_dtype=first.dtype,
        logical_payload_bytes=logical_payload_bytes,
        dataset_storage_bytes=stored_dataset_bytes,
        source_file_bytes=source_path.stat().st_size,
    )


def dataset_storage_bytes(path: Path, keys: list[str]) -> int:
    """Return raw HDF5 dataset storage bytes, excluding file metadata."""
    total = 0
    with h5py.File(path, "r") as h5:
        for key in keys:
            obj = h5[key]
            if isinstance(obj, h5py.Dataset):
                total += int(obj.id.get_storage_size())
    return total


def copy_source_to_benchmark(
    source_path: Path,
    dest_path: Path,
    metadata: SourceMetadata,
    config: CompressionConfig,
) -> float:
    if dest_path.exists():
        dest_path.unlink()

    start = time.perf_counter()
    with HDF5Source("benchmark", source_path, mode="r") as source, HDF5Source(
        "benchmark",
        dest_path,
        mode="w",
        compression=config.compression,
        compression_opts=config.compression_opts,
        shuffle=config.shuffle,
        fletcher32=False,
    ) as writer:
        for key in metadata.keys:
            writer.save(key, source[key])
    return time.perf_counter() - start


def validate_lossless(
    source_path: Path,
    benchmark_path: Path,
    keys: list[str],
) -> bool:
    with HDF5Source("benchmark", source_path, mode="r") as source, HDF5Source(
        "benchmark", benchmark_path, mode="r"
    ) as benchmark:
        for key in keys:
            if not np.array_equal(np.asarray(source[key]), np.asarray(benchmark[key])):
                return False
    return True


def benchmark_reads(
    benchmark_path: Path,
    metadata: SourceMetadata,
    single_keys: list[str],
    batch_groups: list[list[str]],
) -> tuple[float, float, float]:
    with HDF5Source("benchmark", benchmark_path, mode="r") as source:
        start = time.perf_counter()
        for key in metadata.keys:
            _ = source[key]
        sequential_seconds = time.perf_counter() - start

        start = time.perf_counter()
        for key in single_keys:
            _ = source[key]
        random_single_seconds = time.perf_counter() - start

        start = time.perf_counter()
        for group in batch_groups:
            _ = source.get_batch(group)
        random_batch_seconds = time.perf_counter() - start

    return sequential_seconds, random_single_seconds, random_batch_seconds


def dataset_layout_info(path: Path, first_key: str) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        dataset = h5[first_key]
        return {
            "dataset_chunks": dataset.chunks,
            "dataset_compression": dataset.compression,
            "dataset_compression_opts": dataset.compression_opts,
            "dataset_shuffle": bool(dataset.shuffle),
            "dataset_dtype": str(dataset.dtype),
            "dataset_shape": tuple(dataset.shape),
            "dataset_filter_ids": tuple(getattr(dataset, "filter_ids", ())),
            "dataset_filter_names": tuple(getattr(dataset, "filter_names", ())),
        }


def make_result_row_base(
    index: int,
    config: CompressionConfig,
    source_path: Path,
    benchmark_path: Path,
    metadata: SourceMetadata,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "config_index": index,
        "config_name": config.name,
        "compression_repr": repr(config.compression),
        "compression_opts_repr": repr(config.compression_opts),
        "shuffle": config.shuffle,
        "input_source_path": str(args.source_file),
        "source_path": str(source_path),
        "benchmark_path": str(benchmark_path),
        "n_items": len(metadata.keys),
        "first_item_shape": json.dumps(metadata.first_shape),
        "first_item_dtype": str(metadata.first_dtype),
        "logical_payload_bytes": metadata.logical_payload_bytes,
        "source_dataset_storage_bytes": metadata.dataset_storage_bytes,
        "source_file_bytes": metadata.source_file_bytes,
        "input_file_bytes": args.source_file.stat().st_size,
        "status": "ok",
        "error": "",
        "benchmark_file_bytes": "",
        "benchmark_dataset_storage_bytes": "",
        "bytes_per_item": "",
        "compression_ratio_vs_source": "",
        "payload_compression_ratio": "",
        "file_bytes_over_payload_bytes": "",
        "storage_bytes_over_payload_bytes": "",
        "write_seconds": "",
        "write_items_per_sec": "",
        "sequential_read_seconds": "",
        "sequential_items_per_sec": "",
        "random_single_read_seconds": "",
        "random_single_reads_per_sec": "",
        "random_batch_read_seconds": "",
        "random_batch_reads_per_sec": "",
        "random_batch_items_per_sec": "",
        "lossless_ok": "",
        "dataset_chunks": "",
        "dataset_compression": "",
        "dataset_compression_opts": "",
        "dataset_shuffle": "",
        "dataset_filter_ids": "",
        "dataset_filter_names": "",
    }


def main() -> None:
    args = parse_args()
    validate_args(args)

    if not args.source_file.exists():
        raise FileNotFoundError(f"Source file does not exist: {args.source_file}")

    work_dir = args.work_dir
    benchmark_dir = work_dir / "bench_files"
    work_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    source_path = args.source_path or (work_dir / "source_raw.h5")
    materialize_raw_source(
        args.source_file,
        source_path,
        item_count=args.items,
        selection=args.source_selection,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    results_csv = args.results_csv or (work_dir / "results.csv")
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    metadata = read_source_metadata(source_path)
    configs = build_configs(args)
    if not configs:
        raise ValueError("No benchmark configurations selected")

    rng = random.Random(args.seed)
    single_keys = [rng.choice(metadata.keys) for _ in range(args.single_count)]

    batch_groups: list[list[str]] = []
    for _ in range(args.batch_count):
        if args.read_batch_size <= len(metadata.keys):
            group = rng.sample(metadata.keys, k=args.read_batch_size)
        else:
            group = [rng.choice(metadata.keys) for _ in range(args.read_batch_size)]
        batch_groups.append(group)

    validate_keys = rng.sample(
        metadata.keys,
        k=min(args.validate_count, len(metadata.keys)),
    )

    first_path = benchmark_dir / f"0001__{configs[0].name}.h5"
    fieldnames = list(
        make_result_row_base(1, configs[0], source_path, first_path, metadata, args).keys()
    )

    with results_csv.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_file.flush()

        for index, config in enumerate(configs, start=1):
            benchmark_path = benchmark_dir / f"{index:04d}__{config.name}.h5"
            row = make_result_row_base(
                index,
                config,
                source_path,
                benchmark_path,
                metadata,
                args,
            )

            try:
                if benchmark_path.exists() and not args.overwrite:
                    raise FileExistsError(
                        f"Benchmark file already exists: {benchmark_path}. "
                        "Pass --overwrite to replace it."
                    )

                write_seconds = copy_source_to_benchmark(
                    source_path,
                    benchmark_path,
                    metadata,
                    config,
                )
                benchmark_bytes = benchmark_path.stat().st_size
                benchmark_storage_bytes = dataset_storage_bytes(
                    benchmark_path, metadata.keys
                )
                lossless_ok = validate_lossless(
                    source_path,
                    benchmark_path,
                    validate_keys,
                )
                sequential_s, random_single_s, random_batch_s = benchmark_reads(
                    benchmark_path,
                    metadata,
                    single_keys,
                    batch_groups,
                )
                layout = dataset_layout_info(benchmark_path, metadata.keys[0])

                row.update(
                    {
                        "benchmark_file_bytes": benchmark_bytes,
                        "benchmark_dataset_storage_bytes": benchmark_storage_bytes,
                        "bytes_per_item": benchmark_bytes / len(metadata.keys),
                        "compression_ratio_vs_source": (
                            metadata.source_file_bytes / benchmark_bytes
                            if benchmark_bytes > 0
                            else ""
                        ),
                        "payload_compression_ratio": (
                            metadata.logical_payload_bytes / benchmark_storage_bytes
                            if benchmark_storage_bytes > 0
                            else ""
                        ),
                        "file_bytes_over_payload_bytes": (
                            benchmark_bytes / metadata.logical_payload_bytes
                            if metadata.logical_payload_bytes > 0
                            else ""
                        ),
                        "storage_bytes_over_payload_bytes": (
                            benchmark_storage_bytes / metadata.logical_payload_bytes
                            if metadata.logical_payload_bytes > 0
                            else ""
                        ),
                        "write_seconds": write_seconds,
                        "write_items_per_sec": (
                            len(metadata.keys) / write_seconds if write_seconds > 0 else ""
                        ),
                        "sequential_read_seconds": sequential_s,
                        "sequential_items_per_sec": (
                            len(metadata.keys) / sequential_s if sequential_s > 0 else ""
                        ),
                        "random_single_read_seconds": random_single_s,
                        "random_single_reads_per_sec": (
                            len(single_keys) / random_single_s if random_single_s > 0 else ""
                        ),
                        "random_batch_read_seconds": random_batch_s,
                        "random_batch_reads_per_sec": (
                            len(batch_groups) / random_batch_s if random_batch_s > 0 else ""
                        ),
                        "random_batch_items_per_sec": (
                            len(batch_groups) * args.read_batch_size / random_batch_s
                            if random_batch_s > 0
                            else ""
                        ),
                        "lossless_ok": lossless_ok,
                        "dataset_chunks": json.dumps(layout["dataset_chunks"]),
                        "dataset_compression": layout["dataset_compression"],
                        "dataset_compression_opts": repr(
                            layout["dataset_compression_opts"]
                        ),
                        "dataset_shuffle": layout["dataset_shuffle"],
                        "dataset_filter_ids": json.dumps(layout["dataset_filter_ids"]),
                        "dataset_filter_names": json.dumps(
                            layout["dataset_filter_names"]
                        ),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                row.update(
                    {
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            finally:
                csv_writer.writerow(row)
                csv_file.flush()
                print(
                    f"[{index:>3}/{len(configs)}] {config.name:<45} "
                    f"{row['status']}"
                    + (f" - {row['error']}" if row["error"] else "")
                )

                if args.delete_benchmark_files and benchmark_path.exists():
                    benchmark_path.unlink()

    print(f"Results written to {results_csv}")


if __name__ == "__main__":
    main()
