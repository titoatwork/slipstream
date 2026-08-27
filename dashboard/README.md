# Slipstream dashboard

Open via the API server (no npm):

```bash
python -m slipstream.entrypoints.api_server --model Qwen/Qwen2.5-0.5B
# http://127.0.0.1:8000/dashboard/
```

The grid is the physical KV page table. Blue = one sequence, gold = shared prefix pages, red = refcount ≥ 3, dark = free. That is the picture that makes paging visible.
