# Environment

Hardware tiers and software pins are defined in MASTERPLAN §7. This file records **what is actually provisioned**.

## T0 — daily development (this machine)

Captured 2026-08-13.

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM | 6144 MiB |
| SM | 8.6 (Ampere GA107) |
| Driver | 592.82 |
| CUDA toolkit | 12.6.20 (`nvcc`) |
| CPU | AMD Ryzen 5 8645HS (12 threads) |
| RAM | 7.4 GiB |
| Disk | ~895 GiB free |
| OS | Linux |
| System Python | 3.12.3 (`/usr/bin/python3`) |

T0 role (MASTERPLAN): daily iteration, correctness, kernel debugging. Dev model: **Qwen2.5-0.5B**. bf16 is supported.

**RAM is tight (7.4 GiB).** Keep the editor + one Python process. Do not run vLLM and Slipstream concurrently. Prefer CPU-only pytest for contract work; reserve the GPU for parity and kernel tests.

## T1 / T2 / T3 / T4 — cluster

**Not yet verified.** Phase 0 remaining work: confirm exclusive A100 allocation, record interconnect (NVLink vs PCIe), lock clocks where permitted, cache Llama-3.1-8B weights on local/shared storage.

Fill this table when first cluster job lands:

| Tier | Hostname / partition | GPU | Memory | Interconnect | Exclusive? | Clock lock | Notes |
|---|---|---|---|---|---|---|---|
| T1 A100 | — | — | — | — | — | — | canonical results |
| T2 H100 | — | — | — | — | — | — | FP8 / FA3 |
| T3 V100 | — | — | — | — | — | — | bonus only; fp16 |
| T4 multi | — | — | — | — | — | — | TP / PD |

## Software pins

```
Python        >= 3.11 (T0 is 3.12.3; MASTERPLAN named 3.11 — 3.12 is accepted)
PyTorch       >= 2.6   (install extra: .[gpu])
Triton        >= 3.x   (ships with recent PyTorch)
CUDA          12.4+    (T0 has 12.6)
transformers  baseline / parity only — extra: .[bench]
vLLM, SGLang  baselines only — never imported by slipstream/
```

T0 stack (installed 2026-08-13): Python 3.12.3, torch 2.13.0+cu126, triton 3.7.1, transformers 5.15.0. Contract tests still do not require GPU packages.

## Setup

```bash
cd /home/titoisalive/projects/slipstream
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"            # contracts, pytest, ruff, mypy
# GPU engine + HF baseline:
pip install -e ".[gpu,bench]" \
  --extra-index-url https://download.pytorch.org/whl/cu126
pre-commit install
pytest
```

Weight cache (do not re-download inside a benchmark):

```bash
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# After first pull:
#   $HF_HOME/hub/models--Qwen--Qwen2.5-0.5B
```

## Running the Gate 0 HF baseline

```bash
python -m benchmarks.baselines.hf_generate \
    --model Qwen/Qwen2.5-0.5B \
    --max-new-tokens 128 \
    --out benchmarks/results/phase0
```

Writes `hf_qwen25_0.5b.json` + `run_manifest.json`. Requires `. [gpu,bench]` and a free GPU.

## Cluster discipline (repeat of §7.4, operational)

- Exclusive GPU for any number that will be published.
- Record whether clocks were locked (`nvidia-smi -lgc`).
- Log `SLURM_JOB_ID` — `capture_run_manifest` already picks it up.
- Non-exclusive nodes **invalidate** results.
