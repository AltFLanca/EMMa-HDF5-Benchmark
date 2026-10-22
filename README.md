# HDF5 lossless compression benchmark for EMMa augmentations

A standalone benchmark for comparing **lossless HDF5 compression methods** on representative augmentation data written through EMMa-Wrapper's `HDF5Source`.

## Benchmark workflow

The input file may already be compressed. The benchmark does not compare candidate files against the compressed input bytes directly. Instead it:

1. Opens the input file through `HDF5Source`.
2. Selects up to `--items` real items.
3. Reads those items, transparently decoding any existing HDF5 compression.
4. Writes the decoded NumPy values to `<work-dir>/source_raw.h5` with compression and shuffle disabled.
5. Rewrites that exact raw source using every selected compression configuration.
6. Measures compression ratio, write throughput, sequential reads, random single-item reads, and random batch reads.
7. Checks sampled outputs with `numpy.array_equal`.

This gives every compression method the same uncompressed baseline regardless of how the original source file was stored.

## Usage

Benchmark 1,000 randomly selected items from a real augmentation file:

```bash
uv run src/benchmark_hdf5_lossless.py \
  --source-file /path/to/augmentation.h5 \
  --items 1000 \
  --source-selection random \
  --work-dir runs/real_1000 \
  --overwrite
```

Random source selection and random read requests are deterministic by default (`--seed 123`).

To use the first N items instead:

```bash
uv run src/benchmark_hdf5_lossless.py \
  --source-file /path/to/augmentation.h5 \
  --items 1000 \
  --source-selection first \
  --work-dir runs/real_1000 \
  --overwrite
```

## Compression methods

The available method families are:

- `none`
- `gzip`, with and without HDF5 byte shuffle
- `lzf`, with and without HDF5 byte shuffle
- `zstd` (`hdf5plugin.Zstd`)
- `lz4` (`hdf5plugin.LZ4`)
- `bzip2` (`hdf5plugin.BZip2`)
- `bitshuffle`, with no secondary compressor, LZ4, and Zstd
- `blosc`, using BloscLZ, LZ4, LZ4HC, zlib, and Zstd with multiple levels/shuffle modes

The default `quick` preset tests a representative parameter grid. Use `--preset full` for a larger sweep:

```bash
uv run src/benchmark_hdf5_lossless.py \
  --source-file /path/to/augmentation.h5 \
  --items 1000 \
  --preset full \
  --work-dir runs/full \
  --overwrite
```

Or restrict the benchmark to selected method families:

```bash
uv run src/benchmark_hdf5_lossless.py \
  --source-file /path/to/augmentation.h5 \
  --items 1000 \
  --methods none lzf gzip blosc zstd \
  --work-dir runs/core \
  --overwrite
```

There is no compatibility-based method skipping. Every configuration in the selected grid is attempted. If a configuration raises an exception, its CSV row is recorded with `status=error` and the benchmark continues to the next configuration.

## Compression-ratio metrics

`results.csv` reports two useful ratios:

- `compression_ratio_vs_source` = uncompressed `source_raw.h5` whole-file bytes / benchmark whole-file bytes.
- `payload_compression_ratio` = logical NumPy payload bytes / compressed HDF5 dataset storage bytes.

`payload_compression_ratio` isolates the dataset payload from fixed HDF5 metadata and is usually the cleaner metric for comparing compressors. `compression_ratio_vs_source` additionally reflects whole-file overhead and is useful for deployment/storage estimates.

The `none` configuration is a useful sanity check and should normally have a payload compression ratio close to `1.0`.

## Performance metrics

For each configuration the CSV records, among other fields:

- `write_seconds`
- `write_items_per_sec`
- `sequential_read_seconds`
- `sequential_items_per_sec`
- `random_single_read_seconds`
- `random_single_reads_per_sec`
- `random_batch_read_seconds`
- `random_batch_reads_per_sec`
- `random_batch_items_per_sec`
- `lossless_ok`

For ML training, `random_batch_items_per_sec` is generally the most directly interpretable batch-read metric because it measures how many augmentation items are delivered per second rather than how many batch calls are completed.

## Read benchmark controls

Defaults:

- `--single-count 2000`
- `--batch-count 200`
- `--read-batch-size 32`
- `--validate-count 32`

For example:

```bash
uv run src/benchmark_hdf5_lossless.py \
  --source-file /path/to/augmentation.h5 \
  --items 1000 \
  --single-count 5000 \
  --batch-count 500 \
  --read-batch-size 64 \
  --work-dir runs/batch64 \
  --overwrite
```

## Output files

A normal run creates:

```text
<work-dir>/
├── source_raw.h5
├── results.csv
└── bench_files/
    ├── 0001__....h5
    ├── 0002__....h5
    └── ...
```

Use `--delete-benchmark-files` if only the CSV results and raw baseline need to be retained.

## Paper results

The results in the paper were obtained using:
```bash
uv run src/benchmark_hdf5_lossless.py   
  --source-file /home/user/datasets/EMMa-GLP-Depth/glp_depth_uint16_2.63.h5 \
  --items 1000 \
  --source-selection random \
  --seed 123 \
  --preset quick \
  --single-count 2000 \
  --batch-count 200 \
  --read-batch-size 32 \
  --work-dir runs/glp_depth_uint16_2.63_1000 \
  --overwrite
```
