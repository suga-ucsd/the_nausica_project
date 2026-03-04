# 🌿 Nausicaa v3 — llama.cpp edition

4 files. One GGUF. Zero model-loading complexity.

```
config.yaml    all settings
inference.py   LLM (llama-cpp-python) + STT (whisper.cpp) + TTS (Piper)
tools.py       Docker sandbox + all file/shell tools + ReAct agent loop
main.py        CLI REPL + voice pipeline entry point
```

---

## How llama.cpp inference works

```
 .gguf file (single binary)
 ├── architecture metadata  (n_layers, n_heads, rope_freq, n_ctx...)
 ├── tokenizer vocab        (BPE, 32000 tokens)
 ├── chat template          (Jinja2: <|system|>...</s><|user|>...</s>)
 └── quantized weights      (Q4_K_M: 4-bit, 256 weights per block)
         ↓
 Llama(model_path, n_gpu_layers=-1, n_ctx=2048)
         ↓
 mmap weights into memory → offload n_gpu_layers to VRAM
 allocate KV cache: n_ctx × n_layers × 2 × head_dim × fp16
         ↓
 create_chat_completion(messages)
   1. apply chat template → raw token string
   2. tokenize → token ids
   3. prefill: all prompt tokens in one batch → fills KV cache
   4. decode loop: one token per step
      - attention reads K,V from cache (O(1) per step)
      - sample: temperature → top-k → top-p → multinomial
      - stop at eos or stop string
   5. decode ids → text → return
```

**Why GGUF beats transformers for local inference:**
- One file contains everything (weights + tokenizer + chat template + arch)
- No `config.json`, `tokenizer.json`, `tokenizer_config.json` etc.
- Quantization baked in — no bitsandbytes, no `load_in_4bit=True`
- CUDA kernels compiled directly into llama-cpp-python — no `device_map`, no `accelerate`
- llama.cpp uses GGML's AVX2/CUDA int4 matmul — faster than PyTorch for inference

---

## Install

### 1. System deps
```bash
sudo apt install portaudio19-dev alsa-utils tesseract-ocr docker.io
sudo usermod -aG docker $USER && newgrp docker
```

### 2. Python packages
```bash
python3 -m venv .venv && source .venv/bin/activate

# llama-cpp-python WITH CUDA support (important — don't just pip install llama-cpp-python)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir

# Everything else
pip install soundfile pyaudio pyyaml rich huggingface-hub
```

### 3. Download TinyLlama GGUF
```bash
huggingface-cli download bartowski/TinyLlama-1.1B-Chat-v1.0-GGUF \
  TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf \
  --local-dir models/

# Q4_K_M = 668 MB, ~0.5 GB VRAM, good quality
# Q8_0   = 1.1 GB, ~1.0 GB VRAM, near-fp16 quality
```

### 4. Build whisper.cpp (STT)
```bash
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make WHISPER_CUDA=1 -j$(nproc)          # GPU
# make -j$(nproc)                       # CPU only
bash models/download-ggml-model.sh base.en
cd ..
```

### 5. Get Piper TTS
```bash
mkdir piper && cd piper
curl -L https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz \
  | tar -xz --strip-components=1
cd ..
huggingface-cli download rhasspy/piper-voices \
  en/en_US/lessac/medium/en_US-lessac-medium.onnx \
  en/en_US/lessac/medium/en_US-lessac-medium.onnx.json \
  --local-dir models/
```

### 6. Edit config.yaml
```yaml
workspace: "~/your/project"
```

### 7. Run
```bash
source .venv/bin/activate

# Text mode (no mic needed — good for testing)
python main.py --cli

# Voice mode
python main.py
```

---

## Docker sandbox

On startup a container is created:
```
docker run -d --network none --memory 512m --read-only
  -v /your/project:/workspace:rw
  debian:bookworm-slim sleep infinity
```

- `run_shell` commands → `docker exec <container> <cmd>`
- File I/O (read/write/patch) → Python-direct on mounted path (faster)
- `--network none` → tools cannot make network connections
- `--read-only` → container filesystem is immutable
- Auto-removed on exit via `atexit`

---

## ReAct agent loop

```
User: "find all TODO comments"
         ↓
messages = [system_prompt] + history
         ↓
LLM.generate(messages)  →  "TOOL: run_shell\nARGS: {"command": "grep -rn TODO ."}"
         ↓
parse TOOL/ARGS → execute inside Docker → get result
         ↓
inject TOOL_RESULT into history → LLM generates again
         ↓
"ANSWER: Found 3 TODOs in utils.py."
         ↓ (voice mode)
TTS.speak("Found 3 TODOs in utils.py.")
```

---

## i3blocks status bar

Add to `~/.config/i3blocks/config`:
```ini
[nausicaa]
command=cat /tmp/nausicaa_status
interval=1
color=#a9dc76
```
