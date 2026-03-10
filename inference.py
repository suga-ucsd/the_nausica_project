"""
inference.py — all I/O backends.

LLM : HTTP → llama-server  (any model, any architecture, zero Python deps)
STT : whisper.cpp subprocess
TTS : Piper subprocess → aplay

Swapping models = restarting llama-server with -m different.gguf
No recompiling, no reinstalling, no version conflicts.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)


# ── LLM — llama-server HTTP client ───────────────────────────────────────────

class LLM:
    """
    Talks to a running llama-server via /v1/chat/completions.

    Multiple instances can point at different servers:
        fast  = LLM(cfg["servers"]["default"])
        coder = LLM(cfg["servers"]["coder"])
    """

    def __init__(self, server_cfg: dict) -> None:
        base = server_cfg.get("url", "http://localhost:8080").rstrip("/")
        self._url = base + "/v1/chat/completions"
        self._cfg = server_cfg
        log.info("LLM → %s", self._url)
        self._check()

    def _check(self) -> None:
        base = self._url.replace("/v1/chat/completions", "")
        try:
            urllib.request.urlopen(base + "/health", timeout=3)
        except Exception:
            log.warning("llama-server not reachable at %s — start it first:\n"
                        "  llama-server -m models/your.gguf --port 8080 --n-gpu-layers 99",
                        base)

    def generate(self, messages: list[dict], stop: list[str] | None = None) -> str:
        payload = json.dumps({
            "messages":       messages,
            "max_tokens":     self._cfg.get("max_tokens",     512),
            "temperature":    self._cfg.get("temperature",    0.7),
            "top_p":          self._cfg.get("top_p",          0.9),
            "top_k":          self._cfg.get("top_k",          40),
            "repeat_penalty": self._cfg.get("repeat_penalty", 1.1),
            "stop":           stop or ["USER:", "TOOL_RESULT:"],
        }).encode()

        req = urllib.request.Request(
            self._url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read())
        except Exception as e:
            log.error("LLM request failed: %s", e)
            return "ANSWER: Connection error — is llama-server running?"

        text = data["choices"][0]["message"]["content"] or ""
        # Strip Qwen3/deepseek <think> blocks — internal reasoning, not spoken
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text


# ── STT — whisper.cpp subprocess ─────────────────────────────────────────────

class STT:
    def __init__(self, cfg: dict) -> None:
        self._bin   = Path(cfg["binary"]).expanduser()
        self._model = Path(cfg["model"]).expanduser()
        self._lang  = cfg.get("lang", "en")

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        if not self._bin.exists() or audio.size == 0:
            return ""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            sf.write(tmp, audio.astype(np.float32), sr, subtype="PCM_16")
            r = subprocess.run(
                [str(self._bin), "--model", str(self._model),
                 "--language", self._lang, "--file", tmp, "--no-timestamps"],
                capture_output=True, text=True, timeout=60,
            )
            return " ".join(
                l.strip() for l in r.stdout.splitlines()
                if l.strip() and not (l.strip().startswith("[") and l.strip().endswith("]"))
            ).strip()
        finally:
            os.unlink(tmp)


# ── TTS — Piper → aplay ──────────────────────────────────────────────────────

class TTS:
    def __init__(self, cfg: dict) -> None:
        self._bin   = Path(cfg["binary"]).expanduser()
        self._model = Path(cfg["model"]).expanduser()
        self._rate  = cfg.get("rate", 1.0)

    def speak(self, text: str) -> None:
        if not text.strip() or not self._bin.exists():
            return
        try:
            p = subprocess.Popen(
                [str(self._bin), "--model", str(self._model),
                 "--length_scale", str(self._rate), "--output-raw"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            a = subprocess.Popen(
                ["aplay", "-q", "-r", "22050", "-f", "S16_LE", "-c", "1"],
                stdin=p.stdout, stderr=subprocess.DEVNULL,
            )
            p.stdout.close()
            p.communicate(input=text.encode(), timeout=60)
            a.wait(timeout=60)
        except Exception as e:
            log.error("TTS: %s", e)
