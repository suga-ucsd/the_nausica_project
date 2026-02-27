"""
inference.py — all three inference backends in one file.

LLM  : llama-cpp-python  (wraps llama.cpp C library)
STT  : whisper.cpp       (subprocess — C binary you compiled)
TTS  : Piper             (subprocess — binary from GitHub releases)

How llama-cpp-python works under the hood
─────────────────────────────────────────
llama.cpp is a C/C++ transformer inference engine built on GGML, a custom
tensor library with vectorized int4 matrix multiply (AVX2 on CPU, CUDA on GPU).

When you call Llama(model_path, n_gpu_layers=-1):
  1. Opens the .gguf file (mmap — weights are NOT fully loaded into RAM)
  2. Reads architecture metadata from the GGUF header
     (n_layers, n_heads, rope_freq_base, vocab, chat_template — all in the file)
  3. Allocates the KV cache:
     n_ctx × n_layers × 2 × n_heads × head_dim × sizeof(float16)
     For TinyLlama Q4_K_M: 2048 × 22 × 2 × 32 × 64 × 2 bytes ≈ 360 MB
  4. Offloads n_gpu_layers weight tensors to VRAM via CUDA

When you call create_chat_completion(messages):
  1. Applies the Jinja2 chat template stored in the .gguf metadata
     → produces the raw token string with <|system|>…</s><|user|>…</s> etc.
  2. Tokenizes using the BPE vocab from the .gguf
  3. Prefill: runs all prompt tokens through the transformer in one batch
     → fills the KV cache for every prompt token
  4. Decode loop:
     → runs ONE new token per step through transformer
     → at each layer, attention only needs Q for the new token
       but K,V for ALL previous tokens (reads from KV cache) — O(1) per step
     → sample next token (temperature → top-k → top-p → multinomial)
     → stop if eos_token or a stop string is hit
  5. Returns decoded text

No transformers, accelerate, bitsandbytes, sentencepiece, safetensors needed.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  LLM — llama-cpp-python
# ══════════════════════════════════════════════════════════════════════════════

class LLM:
    """
    Wraps llama-cpp-python. One instance lives for the entire session.

    The generate() method accepts a list of chat messages (role/content dicts)
    and returns the assistant reply as a plain string.
    create_chat_completion() handles tokenisation, template, prefill and decode.
    """

    def __init__(self, cfg: dict) -> None:
        from llama_cpp import Llama  # pip install llama-cpp-python

        model_path = Path(cfg["model"]).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(
                f"GGUF model not found: {model_path}\n"
                "Download with:\n"
                "  pip install huggingface-hub\n"
                "  huggingface-cli download bartowski/TinyLlama-1.1B-Chat-v1.0-GGUF"
                " TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf --local-dir models/"
            )

        log.info("Loading GGUF model: %s", model_path.name)
        log.info("  n_gpu_layers=%d  n_ctx=%d", cfg.get("n_gpu_layers", -1), cfg.get("n_ctx", 2048))

        # verbose=False suppresses llama.cpp's own C-level logging
        self._llm = Llama(
            model_path    = str(model_path),
            n_gpu_layers  = cfg.get("n_gpu_layers", -1),
            n_ctx         = cfg.get("n_ctx", 2048),
            verbose       = False,
        )
        self._cfg = cfg
        log.info("LLM ready")

    def generate(self, messages: list[dict]) -> str:
        """
        messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
        Returns the assistant reply text.
        """
        resp = self._llm.create_chat_completion(
            messages       = messages,
            max_tokens     = self._cfg.get("max_tokens", 400),
            temperature    = self._cfg.get("temperature", 0.7),
            top_p          = self._cfg.get("top_p", 0.9),
            top_k          = self._cfg.get("top_k", 40),
            repeat_penalty = self._cfg.get("repeat_penalty", 1.1),
            stop           = ["USER:", "TOOL_RESULT:", "<|user|>"],
        )
        text = resp["choices"][0]["message"]["content"]
        return text.strip() if text else ""


# ══════════════════════════════════════════════════════════════════════════════
#  STT — whisper.cpp subprocess wrapper
# ══════════════════════════════════════════════════════════════════════════════

class STT:
    """
    Calls the whisper.cpp binary on a WAV file.

    whisper.cpp is a C++ port of OpenAI Whisper. It reads float32 audio,
    runs MFCC → encoder → decoder, and prints the transcript to stdout.
    We write the audio to a temp WAV, call the binary, parse stdout.

    Build whisper.cpp:
      git clone https://github.com/ggerganov/whisper.cpp
      cd whisper.cpp && make WHISPER_CUDA=1 -j$(nproc)
      bash models/download-ggml-model.sh base.en
    """

    def __init__(self, cfg: dict) -> None:
        self._binary = Path(cfg["binary"]).expanduser().resolve()
        self._model  = Path(cfg["model"]).expanduser().resolve()
        self._lang   = cfg.get("lang", "en")

        if not self._binary.exists():
            raise FileNotFoundError(f"whisper.cpp binary not found: {self._binary}")
        if not self._model.exists():
            raise FileNotFoundError(f"whisper.cpp model not found: {self._model}")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """audio: float32 numpy array at 16kHz mono. Returns transcript string."""
        if audio.size == 0:
            return ""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            sf.write(tmp, audio.astype(np.float32), sample_rate, subtype="PCM_16")
            r = subprocess.run(
                [str(self._binary),
                 "--model",    str(self._model),
                 "--language", self._lang,
                 "--file",     tmp,
                 "--no-timestamps",
                 "--print-special", "false"],
                capture_output=True, text=True, timeout=60,
            )
            lines = [
                l.strip() for l in r.stdout.splitlines()
                if l.strip() and not (l.strip().startswith("[") and l.strip().endswith("]"))
            ]
            return " ".join(lines).strip()
        finally:
            os.unlink(tmp)


# ══════════════════════════════════════════════════════════════════════════════
#  TTS — Piper subprocess wrapper
# ══════════════════════════════════════════════════════════════════════════════

class TTS:
    """
    Pipes text → Piper binary → raw PCM → aplay.

    Piper is a neural TTS engine by Rhasspy. It takes text on stdin,
    synthesises via an ONNX model, and writes raw int16 PCM to stdout.
    We pipe that directly to aplay (ALSA) so there's zero latency overhead.

    Get Piper: https://github.com/rhasspy/piper/releases
    Get models: https://huggingface.co/rhasspy/piper-voices
    """

    def __init__(self, cfg: dict) -> None:
        self._binary = Path(cfg["binary"]).expanduser().resolve()
        self._model  = Path(cfg["model"]).expanduser().resolve()
        self._rate   = cfg.get("rate", 1.0)

    def speak(self, text: str) -> None:
        """Synthesise and play. Blocks until audio is done."""
        text = text.strip()
        if not text or not self._binary.exists() or not self._model.exists():
            return
        try:
            piper = subprocess.Popen(
                [str(self._binary),
                 "--model",        str(self._model),
                 "--length_scale", str(self._rate),
                 "--output-raw"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            player = subprocess.Popen(
                ["aplay", "-q", "-r", "22050", "-f", "S16_LE", "-c", "1"],
                stdin=piper.stdout, stderr=subprocess.DEVNULL,
            )
            piper.stdout.close()
            piper.communicate(input=text.encode(), timeout=60)
            player.wait(timeout=60)
        except Exception as e:
            log.error("TTS error: %s", e)
