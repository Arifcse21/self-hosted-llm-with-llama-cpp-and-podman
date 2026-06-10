# llama.cpp API Server — Podman Setup
OpenAI-compatible LLM API server. No daemon, rootless, CPU and GPU ready.

---

### Test the API
```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

---

## Connect from CrewAI
```python
from crewai import LLM
llm = LLM(
    model="openai/local",              # "openai/" prefix required
    base_url="http://localhost:8080/v1",
    api_key="none",                    # ignored but required
    temperature=0.1,
    max_tokens=512,
)
```

---

## Connect from Cline (VS Code)

| Field | Value |
|---|---|
| Provider | `OpenAI Compatible` |
| Base URL | `http://localhost:8080/v1` |
| API Key | `local` (any non-empty string) |
| Model ID | your `.gguf` filename exactly |

Cline regularly sends 15,000–30,000 token prompts (file contents + tool results + history). Set `CTX_SIZE` to at least `32768` for agentic coding tasks.

---

## Environment Variables (tune without rebuilding)

| Variable | Default | Notes |
|---|---|---|
| `MODEL` | *(empty)* | Path to .gguf file — required to serve |
| `CTX_SIZE` | `2048` | Context window — see table below |
| `THREADS` | `4` | Match your physical core count (not hyperthreaded) |
| `PARALLEL` | `1` | Concurrent requests — keep at 1 for personal use |
| `N_PREDICT` | `512` | Max output tokens per response |
| `N_GPU_LAYERS` | `0` | Layers offloaded to GPU — see GPU table below |

Override at runtime:
```sh
podman run ... -e CTX_SIZE=32768 -e THREADS=8 -e N_GPU_LAYERS=0 ...
```

---

## Context Size vs RAM

Context window determines how much text (code + history + instructions) the model can hold at once. Each doubling roughly adds 0.7–1.5 GB RAM depending on model size.

| `CTX_SIZE` | RAM overhead (7B Q4) | RAM overhead (14B Q4) | Use case |
|---|---|---|---|
| `2048` | ~1.8 GB | ~3.2 GB | Minimal, single-file edits |
| `4096` | ~2.0 GB | ~3.5 GB | Small projects |
| `8192` | ~2.1 GB | ~3.8 GB | Default — often not enough for Cline |
| `16384` | ~2.8 GB | ~5.0 GB | Medium projects |
| `32768` | ~4.2 GB | ~7.5 GB | Recommended for Cline agentic tasks |
| `65536` | ~7.0 GB | ~13.0 GB | Large codebases — needs 16 GB+ RAM |
| `131072` | ~13.0 GB | ~24.0 GB | Full model train context — 32 GB+ RAM only |

Rule of thumb: for Cline use `32768` as your baseline. Drop to `16384` only if RAM is tight.

---

## GPU Offloading (`--n-gpu-layers`)

llama.cpp offloads transformer layers to VRAM one at a time. You can mix CPU and GPU freely — only the layers that fit go to the GPU, the rest stay on CPU. A 7B Q4 model has **32 layers**; a 14B has **40 layers**; a 32B has **64 layers**.

**CPU-only build** (`-DGGML_CUDA=OFF`, default in this repo): `N_GPU_LAYERS` is ignored. Rebuild with CUDA support to use the GPU.

**CUDA build** (`-DGGML_CUDA=ON`): add `--device nvidia.com/gpu=all` to `podman run`.

### Layers to VRAM — 7B Q4_K_M model

| VRAM | Safe `N_GPU_LAYERS` | Remaining on CPU | Speed gain vs CPU-only |
|---|---|---|---|
| 2 GB | 4–6 | 26–28 | ~15–20% |
| 4 GB | 14–16 | 16–18 | ~40–50% |
| 6 GB | 22–24 | 8–10 | ~65–75% |
| 8 GB | 32 (all) | 0 | ~80–90% |
| 10 GB+ | 32 (all) + KV cache | 0 | max |

### Layers to VRAM — 14B Q4_K_M model

| VRAM | Safe `N_GPU_LAYERS` | Remaining on CPU | Speed gain vs CPU-only |
|---|---|---|---|
| 4 GB | 6–8 | 32–34 | ~15–20% |
| 8 GB | 18–20 | 20–22 | ~45–55% |
| 12 GB | 32–34 | 6–8 | ~70–80% |
| 16 GB | 40 (all) | 0 | ~85–90% |
| 24 GB+ | 40 (all) + KV cache | 0 | max |

### Layers to VRAM — 32B Q4_K_M model

| VRAM | Safe `N_GPU_LAYERS` | Remaining on CPU | Speed gain vs CPU-only |
|---|---|---|---|
| 8 GB | 10–12 | 52–54 | ~15–20% |
| 16 GB | 28–32 | 32–36 | ~45–55% |
| 24 GB | 48–52 | 12–16 | ~70–80% |
| 32 GB+ | 64 (all) | 0 | max |

Speed gains are approximate and depend on CPU, RAM bandwidth, and PCIe generation. Even partial offloading (a quarter of layers) meaningfully reduces CPU load.

### GPU build Containerfile

```dockerfile
FROM nvidia/cuda:12.4.1-devel-debian12

RUN apt-get update && apt-get install -y \
    build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp

WORKDIR /opt/llama.cpp

# sm_86 = Ampere (RTX 3000/A-series)
# sm_89 = Ada Lovelace (RTX 4000-series)
# sm_90 = Hopper (H100)
# sm_75 = Turing (RTX 2000-series)
# sm_61 = Pascal (GTX 1000-series)
# sm_50 = Maxwell (GTX 900 / MX940)
ARG CUDA_ARCH=86
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
    && cmake --build build --config Release -j$(nproc) --target llama-server

EXPOSE 8080
ENTRYPOINT ["/opt/llama.cpp/build/bin/llama-server"]
```

Build for your GPU architecture:
```sh
# RTX 3000/4000 series (most common desktop GPUs)
podman build --build-arg CUDA_ARCH=86 -t llama-cuda:latest -f Containerfile.gpu .

# RTX 4000 series (Ada)
podman build --build-arg CUDA_ARCH=89 -t llama-cuda:latest -f Containerfile.gpu .
```

Run with GPU offloading:
```sh
podman run -d \
  --name llama-server \
  --network=host \
  --device nvidia.com/gpu=all \
  -v llama_models:/models:ro,z \
  llama-cuda:latest \
    --host 127.0.0.1 \
    --port 8080 \
    --model /models/your-model.gguf \
    --ctx-size 32768 \
    --threads $(nproc) \
    --n-gpu-layers 999 \
    --parallel 1
```

`--n-gpu-layers 999` is a common convention meaning "offload as many layers as fit" — llama.cpp caps it at the actual layer count automatically.

Monitor VRAM usage while running:
```sh
watch -n 1 "nvidia-smi --query-gpu=temperature.gpu,memory.used,memory.free,utilization.gpu --format=csv,noheader"
```

---

## GPU Architecture Quick Reference

| GPU family | Examples | CUDA arch | `CUDA_ARCH` value |
|---|---|---|---|
| Maxwell | GTX 950, 960, MX940 | sm_50 | `50` |
| Pascal | GTX 1060, 1070, 1080 | sm_61 | `61` |
| Turing | RTX 2060, 2070, 2080 | sm_75 | `75` |
| Ampere | RTX 3060, 3070, 3080, 3090 | sm_86 | `86` |
| Ada Lovelace | RTX 4060, 4070, 4080, 4090 | sm_89 | `89` |
| Hopper | H100 | sm_90 | `90` |
| Apple Silicon | M1/M2/M3/M4 | Metal (not CUDA) | use `-DGGML_METAL=ON` |

---

## Model Volume

Models are stored in a named Podman volume `llama_models` — they persist across container stops, restarts, and rebuilds. You only download once.

```sh
# Volume location on host
podman volume inspect llama_models

# Copy a model into the volume
podman run --rm \
  -v llama_models:/models \
  -v $(pwd):/src:ro \
  busybox cp /src/your-model.gguf /models/
```

---

## Recommended Models by RAM

| Total RAM | Model | Quant | `CTX_SIZE` | Notes |
|---|---|---|---|---|
| 8 GB | Qwen2.5-Coder-7B-Instruct | Q4_K_M | 16384 | Tight but works |
| 16 GB | Qwen2.5-Coder-7B-Instruct | Q4_K_M | 32768 | Comfortable |
| 16 GB | Qwen2.5-Coder-14B-Instruct | Q4_K_M | 16384 | Better quality |
| 32 GB | Qwen2.5-Coder-14B-Instruct | Q4_K_M | 32768 | Sweet spot |
| 32 GB | Qwen2.5-Coder-32B-Instruct | Q4_K_M | 16384 | Best CPU quality |
| 64 GB+ | Qwen2.5-Coder-32B-Instruct | Q4_K_M | 65536 | No compromises |
