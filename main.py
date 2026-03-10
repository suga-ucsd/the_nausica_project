"""
main.py — Nausicaa entry point.

  python main.py --cli                   # default agent
  python main.py --cli --agent coder     # specific agent
  python main.py --cli --agent router    # auto-routes to best agent
  python main.py                         # voice mode
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path
from cli import run_cli

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler("nausicaa.log"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("nausicaa")


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _run_cli(cfg: dict, agent_name: str) -> None:
    from rich.console import Console
    from rich.prompt import Prompt
    from agents import build

    console = Console()
    console.print("\n[bold green]🌿 Nausicaa[/]  —  Ctrl+C to quit")
    console.print("[dim]/reset  /memory  /agents  /tree  exit[/]\n")

    agents, router = build(cfg)

    # Pick the active interface
    if agent_name == "router" and router:
        active = router
        console.print(f"[dim]Mode: router → auto-delegates to {list(agents.keys())}[/]\n")
    elif agent_name in agents:
        active = agents[agent_name]
        console.print(f"[dim]Agent: {agent_name}[/]\n")
    else:
        active = agents.get("default") or next(iter(agents.values()))
        console.print(f"[dim]Agent: default[/]\n")

    while True:
        try:
            user = Prompt.ask("[bold cyan]you[/]")
        except (KeyboardInterrupt, EOFError):
            active.reset()
            console.print("\n[dim]Session saved. bye[/]")
            break

        u = user.strip()
        if not u: continue
        if u.lower() in ("exit", "quit", "q"):
            active.reset(); break

        # ── Built-in commands ─────────────────────────────────────────
        if u == "/reset":
            active.reset()
            console.print("[dim]Session saved and cleared.[/]"); continue

        if u == "/agents":
            console.print(f"[dim]Available: {list(agents.keys())}[/]"); continue

        if u.startswith("/agent "):
            name = u.split(None, 1)[1].strip()
            if name in agents:
                active = agents[name]
                console.print(f"[dim]Switched to: {name}[/]")
            else:
                console.print(f"[dim]Unknown agent. Available: {list(agents.keys())}[/]")
            continue

        if u == "/memory":
            # Access memory from first agent (shared)
            a = next(iter(agents.values()))
            txt = a._memory.facts.read_text() if a._memory.facts.exists() else "(empty)"
            console.print(f"[dim]{txt}[/]"); continue

        if u == "/tree":
            from agents import Sandbox
            console.print("[dim]" + next(iter(agents.values()))._memory.root.parent.name + "[/]")
            continue

        # ── Chat ──────────────────────────────────────────────────────
        try:
            answer = active.chat(u)
            console.print(f"\n[bold yellow]nausicaa:[/] {answer}\n")
        except Exception as e:
            import traceback
            console.print(f"[bold red]error:[/] {e}\n{traceback.format_exc()}")


# ── Voice ─────────────────────────────────────────────────────────────────────

def run_voice(cfg: dict, agent_name: str) -> None:
    import numpy as np
    import pyaudio
    from inference import STT, TTS
    from agents import build

    agents, router = build(cfg)
    active = (router if agent_name == "router" and router
              else agents.get(agent_name) or agents.get("default")
              or next(iter(agents.values())))

    stt  = STT(cfg["stt"])
    tts  = TTS(cfg["tts"])
    vcfg = cfg.get("voice", {})

    rate    = vcfg.get("sample_rate",    16000)
    thresh  = vcfg.get("vad_threshold",  0.015)
    sil_n   = vcfg.get("silence_frames", 20)
    chunk   = 480

    def set_status(s):
        try: Path("/tmp/nausicaa_status").write_text(s + "\n")
        except: pass

    log.info("Voice mode ready (agent=%s) — speak to Nausicaa", agent_name)
    set_status("idle")

    busy = threading.Event()

    def handle(audio: "np.ndarray") -> None:
        if busy.is_set(): return
        busy.set()
        try:
            set_status("thinking")
            text = stt.transcribe(audio, rate)
            if not text.strip():
                set_status("idle"); return
            log.info("Heard: %r", text)
            answer = active.chat(text)
            set_status("speaking")
            tts.speak(answer)
            set_status("idle")
        except Exception as e:
            log.exception("Pipeline error: %s", e)
            set_status("idle")
        finally:
            busy.clear()

    import queue
    pa = pyaudio.PyAudio()
    q: queue.Queue[bytes] = queue.Queue()

    def cb(data, *_):
        q.put(data); return (None, pyaudio.paContinue)

    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=rate,
                     input=True, frames_per_buffer=chunk, stream_callback=cb)
    set_status("listening")

    ring, speech, in_speech, sil = [], [], False, 0
    try:
        while True:
            frame = q.get(timeout=2)
            import numpy as np
            samples = np.frombuffer(frame, np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(samples ** 2)))
            speaking = rms > thresh

            if not in_speech:
                ring.append(frame)
                if len(ring) > 8: ring.pop(0)
                if speaking:
                    in_speech = True; speech = list(ring); sil = 0
            else:
                speech.append(frame)
                if not speaking:
                    sil += 1
                    if sil >= sil_n and len(speech) >= 8:
                        raw = b"".join(speech)
                        audio = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
                        threading.Thread(target=handle, args=(audio,), daemon=True).start()
                        ring, speech, in_speech, sil = [], [], False, 0
                else:
                    sil = 0
    except KeyboardInterrupt:
        log.info("Stopped.")
    finally:
        stream.stop_stream(); stream.close(); pa.terminate()
        active.reset()


# ── Entry ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli",    action="store_true", help="text mode")
    ap.add_argument("--agent",  default="default",   help="agent name or 'router'")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    (run_cli if args.cli else run_voice)(cfg, args.agent)


if __name__ == "__main__":
    main()
