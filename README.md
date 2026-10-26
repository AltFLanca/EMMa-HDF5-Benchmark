
# EMMa-HDF5-Benchmark

Benchmark HDF5 compression methods on EMMa depth maps, masks, and embeddings, comparing storage efficiency and read/write performance across different compression configurations. The benchmark compares uncompressed storage against several lossless compression methods, including gzip, LZF, Zstd, LZ4, BZip2, Bitshuffle, and Blosc.

Dataset access is handled through [EMMa-Wrapper](https://github.com/AltFLanca/EMMa-Wrapper).

## Setup

Install the project dependencies using [uv](https://docs.astral.sh/uv/). From the repository root, run:

```bash
uv sync --locked
```

You need an existing HDF5 augmentation file to run the benchmark. No datasets are included in this repository.

## Repository Structure

- `src/benchmark_hdf5.py` - Benchmarks HDF5 compression configurations, measuring storage efficiency and read/write performance.

## Run Benchmark

Benchmarking requires an existing HDF5 augmentation file, provided through the `--source-file` argument. The input must follow the `HDF5Source` layout, with one top-level dataset per ASIN. The input file may already be compressed. Selected items are first decoded and written to a common uncompressed baseline to ensure consistent comparisons across configurations.

The benchmark can be configured using the following arguments:

- `--items` - Maximum number of source items to benchmark.
- `--source-selection` - Item selection strategy (`random` or `first`).
- `--seed` - Random seed for reproducible sampling and read requests.
- `--preset` - Compression configuration grid (`quick` or `full`).
- `--methods` - Optional subset of compression methods to evaluate.
- `--single-count` - Number of random single-item reads per configuration.
- `--batch-count` - Number of random batch reads per configuration.
- `--read-batch-size` - Number of items in each random read batch.
- `--validate-count` - Number of sampled items checked for exact lossless equality.
- `--work-dir` - Directory for benchmark outputs.
- `--overwrite` - Replaces existing benchmark outputs.
- `--delete-benchmark-files` - Removes compressed candidate files after measurement.

Run the benchmark with:

```bash
uv run src/benchmark_hdf5.py \
  --source-file /path/to/EMMa-DepthAnythingV3/da3_large1.1_uint16_504.h5 \
  --items 1000 \
  --source-selection random \
  --seed 123 \
  --preset quick \
  --single-count 2000 \
  --batch-count 200 \
  --read-batch-size 32 \
  --work-dir runs/da3_large1.1_uint16_504_1000 \
  --overwrite
```

## Benchmark Results

The benchmark generates the following outputs in the specified work directory:

- `source_raw.h5` - Uncompressed baseline containing the selected source items.
- `results.csv` - Performance metrics and validation results for each compression configuration.
- `bench_files/` - HDF5 files generated for each compression configuration.

The results include:

- **Write throughput** - Speed of writing compressed data.
- **Sequential read throughput** - Speed of reading items sequentially.
- **Random read throughput** - Speed of reading individual items and batches in random order.
- **Compression ratio** - Ratio between the uncompressed baseline file size and the compressed file size, including HDF5 overhead.
- **Payload compression ratio** - Ratio between the logical array size and the stored dataset size.

Configurations that fail during benchmarking are recorded in `results.csv` with `status=error`.