# benchscope

A wrapper around stock `llama-bench` (from
[llama.cpp](https://github.com/ggml-org/llama.cpp)) that keeps its output and
adds live system telemetry: GPU name, VRAM used/total, RAM used/total, GPU
and CPU utilisation, temperature, power — plus a GPU-fit max-context estimate
and an optional 16:9 summary card PNG.

## Features

- **Zero-loss passthrough** — everything after `--` goes straight to
  `llama-bench`; all its flags (`-m -p -n -pg -t -ngl -b -ctk -d -r …`) work.
- **Live telemetry** (background sampler, default 0.2 s):
  AMD via sysfs + `rocm-smi` (fast path, no subprocess per tick), NVIDIA via
  `nvidia-smi`, CPU/RAM via `psutil`. Adds per-row `telemetry` to the JSON:
  `vram_peak/total/delta`, `ram_peak/total`, `gpu_avg/max`, `cpu_sys_avg/max`,
  `temp_max`, `power_avg/max`, `duration_s`.
- **Single vs `--per-test`** — single mode runs one bench process
  (telemetry is run-global); `--per-test` runs one process per pp/tg/pg size
  so each row gets isolated telemetry (costs one model load per test).
  Understands comma lists, repeated flags and `a-b` / `a-b+s` / `a-b*m`
  ranges for `-p`/`-n` (incl. `--n-prompt=` spellings).
- **GPU-fit max context** — estimates how many context tokens fit on the GPU
  with full weight offload: `VRAM − weights − reserve` divided by KV-cache
  bytes/token derived from GGUF hparams (GQA heads, K/V dims, cache dtypes,
  sliding-window layers counted at window size only), capped at the model's
  native context. `*` marks native-limited models. Tune headroom with
  `--ctx-reserve-mib` (default 800).
- **Outputs** — enriched JSON, per-sample timeseries CSV (`run_idx,label,…`),
  markdown table, and `--out-png` 1920×1080 summary card (table + throughput /
  GPU% / power charts). Existing files are never overwritten (`bench.png` →
  `bench_01.png`, …). ROCm lib path (`/opt/rocm/lib`) is injected into the
  bench subprocess automatically.

## Requirements

- Python 3.10+ with `psutil` and `matplotlib` (`pip install psutil matplotlib`)
- A `llama-bench` binary (CPU, CUDA, ROCm/HIP, Vulkan all fine)
- Linux for GPU telemetry (AMD sysfs/`rocm-smi` or `nvidia-smi`)

## Usage

```bash
python3 benchscope.py --list-gpus

python3 benchscope.py --per-test \
  --llama-bench-bin /path/to/llama.cpp/build/bin/llama-bench \
  --out-json bench.json --timeseries bench.csv --out-png bench.png -- \
  -m model.gguf -p 512 -n 128 -r 3
```

| Flag | Default | Meaning |
|---|---|---|
| `--llama-bench-bin` | auto-detect | path to `llama-bench` |
| `--per-test` | off | isolate telemetry per pp/tg/pg test |
| `--interval` | 0.2 | telemetry sampling seconds |
| `--gpu-index` | 0 | which GPU to report (0 = largest VRAM) |
| `--ctx-reserve-mib` | 800 | VRAM headroom for the ctx-fit estimate |
| `--out-json` / `--timeseries` / `--out-md` / `--out-png` | `bench_enriched.json` / `bench_timeseries.csv` / — / — | outputs |
| `--list-gpus` | — | print detected GPUs and exit |

## Example

Spark-X2.5-4B quant comparison on AMD RX 6800M, ROCm full offload
(`-p 512 -n 128 -r 3`, per-test):

| quant | size | pp 512 (t/s) | tg 128 (t/s) | ctx fit | VRAM | power (tg) |
|---|---|---|---|---|---|---|
| BF16 | 7.66 GiB | 474.36 ± 16.58 | 39.11 ± 0.18 | 101673 | ~8.2 GiB | 120.7 W |
| Q8_0 | 4.07 GiB | 2148.70 ± 214.94 | 67.39 ± 0.54 | 206244 | ~4.6 GiB | 101.5 W |
| Q4_K_M | 2.42 GiB | 1510.64 ± 102.93 | 98.47 ± 1.35 | 254389 | ~2.9 GiB | 116.0 W |

Q8_0 leads prompt processing while Q4_K_M leads generation — the kind of
trade-off this tool's per-phase telemetry makes visible.

See `RUN_SPARK_X2.5_ON_RX6800M.txt` for the full story of getting that model
running (it needed a llama.cpp update for `spark2_5` support).

## Caveats

- The ctx-fit number is an **estimate**: it assumes all weights live on the
  reported GPU and ignores batch-size/compute-buffer growth. Validate with
  `-d` prefill before a real long-context run.
- With multiple GPUs, llama.cpp may split layers across devices, which raises
  the real ceiling above the single-GPU estimate.

## Disclaimer

benchscope is an independent community tool. It is **not affiliated with,
endorsed, or sponsored by Meta Platforms, Inc. or the ggml-org/llama.cpp
project**. "Llama" is a trademark of Meta Platforms, Inc.; references to
`llama-bench`/`llama.cpp` here are purely descriptive (nominative use) to
indicate interoperability. Model names, GPU names and benchmark figures are
property of their respective owners. All telemetry and context-fit figures
are best-effort estimates provided "as is" — see LICENSE (MIT).
