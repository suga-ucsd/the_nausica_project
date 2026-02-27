"""
main.py — Nausicaa entry point.

Usage:
  python main.py           # voice mode (mic → whisper.cpp → LLM → piper)
  python main.py --cli     # text REPL (no mic/speaker needed)

Requires:
  pip install llama-cpp-python soundfile pyaudio pyyaml rich
  (llama-cpp-python needs to be installed with CUDA support — see README)
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    import numpy as np

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("nausicaa.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("nausicaa")


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["workspace"] = Path(cfg["workspace"]).expanduser().resolve()
    return cfg


# ── Voice listener — energy-based VAD, no extra packages ─────────────────────

class Listener:
    """
    PyAudio mic capture with RMS energy Voice Activity Detection.

    How it works:
      - Captures 30ms frames from the mic
      - Computes RMS of each frame: sqrt(mean(samples²))
      - If RMS > threshold → speech frame, start collecting
      - N consecutive quiet frames → utterance done, yield audio array
    """

    def __init__(self, cfg: dict) -> None:
        self._rate    = cfg.get("sample_rate",    16000)
        self._thresh  = cfg.get("vad_threshold",  0.015)
        self._sil_n   = cfg.get("silence_frames", 20)
        self._chunk   = 480   # 30ms at 16kHz

    def listen_once(self) -> "np.ndarray":
        import queue
        import numpy as np
        import pyaudio

        pa  = pyaudio.PyAudio()
        q: queue.Queue[bytes] = queue.Queue()

        def cb(data, *_):
            q.put(data)
            return (None, pyaudio.paContinue)

        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=self._rate,
                         input=True, frames_per_buffer=self._chunk, stream_callback=cb)

        ring, speech, in_speech, sil = [], [], False, 0
        try:
            while True:
                frame = q.get(timeout=2)
                samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(samples ** 2)))
                is_speech = rms > self._thresh

                if not in_speech:
                    ring.append(frame)
                    if len(ring) > 8: ring.pop(0)
                    if is_speech:
                        in_speech = True
                        speech = list(ring)
                        sil = 0
                else:
                    speech.append(frame)
                    if not is_speech:
                        sil += 1
                        if sil >= self._sil_n and len(speech) >= 8:
                            raw = b"".join(speech)
                            return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    else:
                        sil = 0
        finally:
            stream.stop_stream(); stream.close(); pa.terminate()


# ── Status bar for i3blocks ───────────────────────────────────────────────────

def set_status(state: str) -> None:
    icons = {"idle":"  idle","listening":"  listening","thinking":"  thinking…","speaking":"  speaking"}
    try:
        Path("/tmp/nausicaa_status").write_text(icons.get(state, state) + "\n")
    except Exception:
        pass


# ── Voice mode ────────────────────────────────────────────────────────────────

def run_voice(cfg: dict) -> None:
    from inference import LLM, STT, TTS
    from tools import Agent

    log.info("Loading models…")
    set_status("idle")

    llm   = LLM(cfg["llm"])
    stt   = STT(cfg["stt"])
    tts   = TTS(cfg["tts"])
    agent = Agent(
        llm         = llm,
        workspace   = cfg["workspace"],
        docker_cfg  = cfg.get("docker", {}),
        whitelist   = cfg.get("shell_whitelist", []),
        max_iter    = cfg.get("max_iterations", 8),
    )

    listener = Listener(cfg.get("voice", {}))
    busy     = threading.Event()

    log.info("Voice pipeline ready. Speak to Nausicaa…")
    set_status("listening")

    def handle(audio):
        if busy.is_set(): return
        busy.set()
        try:
            set_status("thinking")
            text = stt.transcribe(audio)
            if not text.strip():
                set_status("listening")
                return
            log.info("Heard: %r", text)
            answer = agent.chat(text)
            log.info("Answer: %r", answer)
            set_status("speaking")
            tts.speak(answer)
            set_status("listening")
        except Exception as e:
            log.exception("Pipeline error: %s", e)
            set_status("listening")
        finally:
            busy.clear()

    try:
        while True:
            audio = listener.listen_once()
            if audio is not None:
                threading.Thread(target=handle, args=(audio,), daemon=True).start()
    except KeyboardInterrupt:
        log.info("Stopped.")
        set_status("idle")


# ── CLI (text REPL) ───────────────────────────────────────────────────────────

def run_cli(cfg: dict) -> None:
    from rich.console import Console
    from rich.prompt  import Prompt
    from inference import LLM
    from tools     import Agent

    console = Console()
    console.print("\n[bold green]🌿 Nausicaa CLI[/]  —  Ctrl+C to quit\n"
                  "[dim]Commands: /reset  /tree  exit[/]\n")

    log.info("Loading LLM…")
    llm   = LLM(cfg["llm"])
    agent = Agent(
        llm        = llm,
        workspace  = cfg["workspace"],
        docker_cfg = cfg.get("docker", {}),
        whitelist  = cfg.get("shell_whitelist", []),
        max_iter   = cfg.get("max_iterations", 8),
    )

    while True:
        try:
            user = Prompt.ask("[bold cyan]you[/]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]bye[/]"); break

        u = user.strip()
        if not u: continue
        if u.lower() in ("exit", "quit", "q"): break
        if u == "/reset":
            agent.reset(); console.print("[dim]Session cleared.[/]"); continue
        if u == "/tree":
            console.print(agent._sb.tree(depth=2)); continue

        try:
            answer = agent.chat(u)
            console.print(f"\n[bold yellow]nausicaa:[/] {answer}\n")
        except Exception as e:
            import traceback
            console.print(f"[bold red]error:[/] {e}\n{traceback.format_exc()}")


# ── Entry ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli",    action="store_true")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    (run_cli if args.cli else run_voice)(cfg)


if __name__ == "__main__":
    main()
