# llama.cpp API Server — Podman Setup

OpenAI-compatible LLM API server. No daemon, rootless, 4GB VPS friendly.
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

## Environment Variables (tune without rebuilding)

| Variable | Default | Notes |
|---|---|---|
| `MODEL` | *(empty)* | Path to .gguf file — required to serve |
| `CTX_SIZE` | `2048` | Context window — lower = less RAM |
| `THREADS` | `4` | Match your vCPU count |
| `PARALLEL` | `1` | Concurrent requests — keep at 1 |
| `N_PREDICT` | `512` | Max output tokens |

Override at runtime:
```sh
podman run ... -e CTX_SIZE=1024 -e THREADS=4 ...
```

---

## Model Volume

Models are stored in a named Podman volume `llama_models` — they **persist** across container stops, restarts, and rebuilds. You only download once.

```sh
# Volume location on host
podman volume inspect llama_models
```